"""
This module contains the Shape class, which is used to set up a geometry/shape for solving PDE on with smoothed boundary method.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional,Callable, Literal
import diffrax as dfx
import scipy
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import null_space  # CPU-only; used once at setup

from .utils.derivatives import _gradx_c, _grady_c, _grad2x_c, _grad2y_c, _grad2xy_c

Array = jax.Array


@dataclasses.dataclass
class Shape:
    """Sets up a geometry/shape for solving PDE on with smoothed boundary method.

    The user creates a shape by providing a binary representation and an optional smoothing parameter.
    """

    binary: Array
    dx: Optional[Tuple[float, float]] = (1.0, 1.0)
    smooth_epsilon: float = 1.0
    smooth_curvature: float = 0.0
    smooth_dt: float = 0.1
    smooth_tf: float = 1.0

    def __post_init__(self):
        self.smooth = self.smooth_shape()
        self.smooth = jnp.where(self.smooth < 0.001, 0.001, self.smooth)
        self.smooth = jnp.where(self.smooth > 0.99, 1.0, self.smooth)

    def smooth_shape(self) -> Array:
        """Smooths the shape using the Allen-Cahn equation with curvature minimization."""

        def potential(u):
            return 18.0 / self.smooth_epsilon * u * (1.0 - u) * (1.0 - 2.0 * u)

        @jax.jit
        def rhs(t, u, args):
            gradx = _gradx_c(u, self.dx[0])
            grady = _grady_c(u, self.dx[1])
            grad2x = _grad2x_c(u, self.dx[0])
            grad2y = _grad2y_c(u, self.dx[1])
            grad2xy = _grad2xy_c(u, self.dx[0], self.dx[1])
            grad_norm_sq = gradx**2 + grady**2
            grad_norm_sq = jnp.where(grad_norm_sq < 1e-7, 1.0, grad_norm_sq)
            norm_laplace = (
                grad2x * gradx**2 + 2.0 * grad2xy * gradx * grady + grad2y * grady**2
            ) / grad_norm_sq
            laplace = grad2x + grad2y
            return (
                2.0
                * (
                    self.smooth_curvature * laplace
                    + (1.0 - self.smooth_curvature) * norm_laplace
                )
                - potential(u) / self.smooth_epsilon
            )

        solution = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            dfx.Tsit5(),
            t0=0.0,
            t1=self.smooth_tf,
            dt0=self.smooth_dt,
            y0=self.binary,
            stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
            saveat=dfx.SaveAt(t1=True),
            max_steps=1000000,
        )

        return solution.ys[-1]

    def laplacian_from_mask(self, periodic: bool = False):
        """
        Unnormalized graph Laplacian (4-neighbour) from a 0/1 mask.
        Nodes are entries where mask==1. Two nodes connect if they are
        up/down/left/right neighbours and both are 1.

        Returns:
            L  : (n_nodes, n_nodes) CSR Laplacian
            ids: (H, W) array, node index in [0, n_nodes) or -1 if not a node
        """
        mask = self.binary > 0
        H, W = mask.shape
        ids = -np.ones((H, W), dtype=np.int64)
        ids[mask] = np.arange(mask.sum(), dtype=np.int64)
        n = int(mask.sum())
        if n == 0:
            return csr_matrix((0, 0)), ids

        def undirected_edges(dy, dx):
            """Return endpoints (u,v) for each undirected edge, listed once."""
            if periodic:
                m_both = mask & np.roll(mask, (dy, dx), axis=(0, 1))
                if not m_both.any():
                    return np.empty(0, np.int64), np.empty(0, np.int64)
                u = ids[m_both]
                v = np.roll(ids, (dy, dx), axis=(0, 1))[m_both]
                return u, v
            else:
                y0, y1 = max(0, dy), H + min(0, dy)
                x0, x1 = max(0, dx), W + min(0, dx)
                m1 = mask[y0:y1, x0:x1]
                m2 = mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
                both = m1 & m2
                if not both.any():
                    return np.empty(0, np.int64), np.empty(0, np.int64)
                u = ids[y0:y1, x0:x1][both]
                v = ids[y0 - dy : y1 - dy, x0 - dx : x1 - dx][both]
                return u, v

        # Build edges once using right and down neighbours, then symmetrize
        ur, vr = undirected_edges(0, +1)  # right
        ud, vd = undirected_edges(+1, 0)  # down

        u_one = np.concatenate([ur, ud])
        v_one = np.concatenate([vr, vd])

        # Degree from unique undirected edges: each endpoint counted once
        deg = np.bincount(np.concatenate([u_one, v_one]), minlength=n).astype(
            np.float64
        )

        # Off-diagonals: symmetrize edges (u,v) and (v,u)
        rows_off = np.concatenate([u_one, v_one])
        cols_off = np.concatenate([v_one, u_one])
        data_off = -np.ones(rows_off.shape[0], dtype=np.float64)

        # Diagonal
        rows = np.concatenate([rows_off, np.arange(n)])
        cols = np.concatenate([cols_off, np.arange(n)])
        data = np.concatenate([data_off, deg])

        L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        return L, ids

    def get_shape_modes(self, N: Optional[int] = None):
        """Get the first N eigenvectors of the graph Laplacian of the binary mask.

        Creates a graph where nodes are the 1-valued pixels, with edges between
        adjacent pixels (left, right, top, bottom neighbors).

        Args:
            N: Number of eigenvectors to return. If None, returns all eigenvectors.
            downsampling_factor: If provided, downsample binary by this factor before
                computing modes, then upsample results back to original size.
                This can significantly reduce memory usage and computation time
                for large binary masks.

        Returns:
            Array of shape (num_nodes, N) containing the first N eigenvectors
        """

        laplacian, node_ids = self.laplacian_from_mask()
        # return laplacian, node_ids

        n = laplacian.shape[0]

        # Check if Laplacian matrix is symmetric
        is_symmetric = (laplacian != laplacian.T).nnz == 0
        if not is_symmetric:
            raise ValueError("Laplacian matrix is not symmetric")

        # A scale-aware tiny shift: ~ 1e-8 times a typical diagonal magnitude
        diag_mean = float(laplacian.diagonal().mean()) if n > 0 else 1.0
        sigma = max(diag_mean, 1.0) * 1e-8
        # Get only the first N eigenvectors (much faster than computing all)
        eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(
            laplacian,
            k=N,
            which="LM",
            sigma=sigma,
            tol=1e-8,
            maxiter=None,
        )

        # Initialize output array with zeros
        shape = self.binary.shape
        output = np.zeros((shape[0], shape[1], N))

        # Vectorized assignment using advanced indexing
        # Get valid node positions (where node_ids >= 0)
        valid_mask = node_ids >= 0
        valid_node_ids = node_ids[valid_mask]

        # print(valid_mask)
        # print(valid_node_ids)

        # Fill in eigenvector values at node locations
        for i in range(N):
            eigenvec = eigenvecs[:, i]
            output[valid_mask, i] = eigenvec[valid_node_ids]

        self.shape_basis = jnp.array(output)
        self.shape_basis_evals = eigenvals
        
    def get_mask_edge(self, connectivity: int = 4, side: str = "inner") -> jnp.ndarray:
        """
        Find edge pixels of the binary mask using a neighbor-count convolution.

        Args:
            connectivity: 4 or 8
            side: 'inner' (inside pixels touching background),
                'outer' (outside pixels touching shape),
                'both'  (union)

        Returns:
            Boolean array with the same shape as self.binary.
        """
        assert connectivity in (4, 8), "connectivity must be 4 or 8"
        assert side in ("inner", "outer", "both"), "side must be 'inner', 'outer', or 'both'"

        b = (self.binary > 0).astype(jnp.float32)  # (H, W)

        if connectivity == 4:
            kernel = jnp.array([[0., 1., 0.],
                                [1., 0., 1.],
                                [0., 1., 0.]], dtype=jnp.float32)
        else:  # 8-connectivity
            kernel = jnp.array([[1., 1., 1.],
                                [1., 0., 1.],
                                [1., 1., 1.]], dtype=jnp.float32)

        # Conv expects (N, C, H, W). Use N=C=1 and SAME padding to preserve shape.
        def conv2(x):
            return lax.conv_general_dilated(
                lhs=x[None, None, ...],
                rhs=kernel[None, None, ...],
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )[0, 0]

        # Number of "shape" neighbors each pixel has
        shape_neighbors = conv2(b)  # sum of b in neighborhood (excluding center)

        # Number of available neighbors (accounts for image borders)
        available_neighbors = conv2(jnp.ones_like(b))

        # Inside pixels touching background: have at least one neighbor different
        inner_edge = (b == 1) & (shape_neighbors < available_neighbors)

        # Outside pixels touching shape: at least one neighbor is inside
        outer_edge = (b == 0) & (shape_neighbors > 0)

        if side == "inner":
            return inner_edge
        elif side == "outer":
            return outer_edge
        else:
            return inner_edge | outer_edge


@dataclasses.dataclass
class SpatialHeterogeneity(Shape):
    """
    A Shape subclass that augments a given Shape with a truncated basis of shape modes
    and a parameterized spatial heterogeneity field.

    Usage:
        sh = SpatialHeterogeneity.from_shape(
            base_shape, n_basis=16, prior_fn=jax.exp, params=jnp.zeros(16)
        )
        field = sh.evaluate()                # (H, W)
        field2 = sh.evaluate(new_params)     # (H, W) using provided params
    """

    n_basis: int = 0
    prior_fn: Callable[[Array], Array] = lambda x: x

    _basis: Array = dataclasses.field(init=False, repr=False)   # (H, W, n_basis)
    _evals: Array = dataclasses.field(init=False, repr=False)   # (n_basis,)
    params: Array = dataclasses.field(init=False, repr=False)   # (n_basis,)

        # ------------- Construction helpers -------------
    @classmethod
    def from_shape(
        cls,
        shape: Shape,
        n_basis: int,
        prior_fn: Optional[Callable[[Array], Array]] = None,
        params: Optional[Array] = None,
    ) -> "SpatialHeterogeneity":
        """
        Create a SpatialHeterogeneity subclass instance from an *existing* Shape.
        Copies over all Shape fields, runs Shape's __post_init__ (smoothing),
        then builds the shape modes and installs parameters.
        """
        # Build the subclass by copying Shape state
        self = cls(
            binary=shape.binary,
            dx=shape.dx,
            smooth_epsilon=shape.smooth_epsilon,
            smooth_curvature=shape.smooth_curvature,
            smooth_dt=shape.smooth_dt,
            smooth_tf=shape.smooth_tf,
            n_basis=n_basis,
            prior_fn=prior_fn or (lambda x: x),
        )
        # __post_init__ will run here (from Shape), building self.smooth

        # Build basis
        basis, evals = self._compute_basis(n_basis)
        self._basis = basis  # (H, W, n_basis)
        self._evals = evals  # (n_basis,)

        # Install params
        if params is None:
            params = jnp.zeros((n_basis,), dtype=self._basis.dtype)
        else:
            params = jnp.asarray(params, dtype=self._basis.dtype)
            if params.shape != (n_basis,):
                raise ValueError(f"params must have shape ({n_basis},), got {params.shape}")
        self.params = params
        return self

    def __post_init__(self):
        # Run Shape's smoothing routine first
        super().__post_init__()
        # Note: _basis/_evals/params are populated in from_shape() to avoid
        # rebuilding modes every time Shape.__post_init__ runs.

    # ------------- Basis & evaluation -------------
    def _compute_basis(self, N: int) -> Tuple[Array, Array]:
        """
        Compute and return (basis, evals) where:
          basis: (H, W, N) eigenmodes placed back on the image grid
          evals: (N,) eigenvalues
        Uses Shape.laplacian_from_mask() and scipy.sparse.linalg.eigsh as in your code.
        """
        # Compute Laplacian and node ids
        L, node_ids = self.laplacian_from_mask()
        n = L.shape[0]
        if n == 0:
            H, W = self.binary.shape
            return jnp.zeros((H, W, N)), jnp.zeros((N,))

        # Symmetry check
        if (L != L.T).nnz != 0:
            raise ValueError("Laplacian matrix is not symmetric")

        import scipy.sparse.linalg as spla

        diag_mean = float(L.diagonal().mean()) if n > 0 else 1.0
        sigma = max(diag_mean, 1.0) * 1e-8

        # Compute first N eigenpairs near zero (shift-invert for smoothest modes)
        evals, vecs = spla.eigsh(L, k=N, which="LM", sigma=sigma, tol=1e-8, maxiter=None)

        # Scatter back to (H, W, N)
        H, W = self.binary.shape
        out = np.zeros((H, W, N), dtype=np.float64)
        valid_mask = node_ids >= 0
        valid_ids = node_ids[valid_mask]
        for i in range(N):
            out[valid_mask, i] = vecs[valid_ids, i]

        # Convert to JAX arrays
        return jnp.asarray(out), jnp.asarray(evals)

    def _select_mask(self, mask_type: Literal["binary", "smooth"]) -> Array:
        """Return the requested mask array."""
        if mask_type == "binary":
            return self.binary
        elif mask_type == "smooth":
            return self.smooth
        else:
            raise ValueError(f"mask_type must be 'binary' or 'smooth', got {mask_type!r}")
    

    def set_params(self, params: Array) -> None:
        params = jnp.asarray(params, dtype=self._basis.dtype)
        if params.shape != (self.n_basis,):
            raise ValueError(f"params must have shape ({self.n_basis},), got {params.shape}")
        self.params = params

    # ---------------------- EVALUATION (mask optional) ----------------------
    def evaluate(
        self,
        params: Optional[Array] = None,
        *,
        apply_mask: bool = False,
        mask_type: Literal["binary", "smooth"] = "binary",
    ) -> Array:
        """
        Evaluate the heterogeneity field:
          1) linear combo: field = sum_i params[i] * basis[..., i]
          2) prior transform
          3) (optional) mask by 'binary' or 'smooth'

        Returns: (H, W) JAX array
        """
        coeffs = self.params if params is None else jnp.asarray(params, dtype=self._basis.dtype)
        if coeffs.shape != (self.n_basis,):
            raise ValueError(f"params must have shape ({self.n_basis},), got {coeffs.shape}")

        field = jnp.tensordot(self._basis, coeffs, axes=([-1], [0]))
        field = self.prior_fn(field)

        if apply_mask:
            mask = self._select_mask(mask_type)
            field = field * mask
        return field

    def make_evaluator(
        self,
        *,
        apply_mask: bool = False,
        mask_type: Literal["binary", "smooth"] = "binary",
    ) -> Callable[[Array], Array]:
        basis = self._basis
        prior = self.prior_fn
        mask = self._select_mask(mask_type) if apply_mask else None

        @jax.jit
        def f(theta: Array) -> Array:
            fld = jnp.tensordot(basis, theta, axes=([-1], [0]))
            fld = prior(fld)
            if mask is not None:
                fld = fld * mask
            return fld

        return f

    @property
    def shape_modes(self) -> Array:
        return self._basis

    @property
    def shape_mode_evals(self) -> Array:
        return self._evals

    # ---------------------- MEAN-CONSTRAINED SETUP ----------------------
    def build_mean_constrained(
        self,
        *,
        mask_type: Literal["binary", "smooth"] = "binary",
    ) -> None:
        """
        Precompute U (N x (N-1)) so that for any p_het, theta = U @ p_het
        yields zero change in the mean defined by 'mask_type'.
        """
        basis = np.asarray(self._basis)  # (H,W,N)
        H, W, N = basis.shape
        mask = np.asarray(self._select_mask(mask_type))
        mask_bool = mask > 0

        # Fill outside-mask entries with the in-mask mean (per your Julia logic)
        basis_filled = np.empty_like(basis)
        for i in range(N):
            mode = basis[..., i]
            avg_in = mode[mask_bool].mean() if mask_bool.any() else 0.0
            mcopy = mode.copy()
            mcopy[~mask_bool] = avg_in
            basis_filled[..., i] = mcopy

        # Flatten to (HW, N) and make 1×N mean operator row
        S = basis_filled.reshape(H * W, N)
        A = S.mean(axis=0, keepdims=True)  # (1, N)

        U = null_space(A)  # (N, N-1) for rank-1 A
        if U.shape[1] != max(N - 1, 0):
            raise RuntimeError(
                f"Unexpected nullspace dimension {U.shape[1]} for N={N}. "
                "Check that basis columns are not constant/degenerate."
            )

        self._mean_constrained_U = jnp.asarray(U)            # (N, N-1)
        self._mean_constrained_mask_type = mask_type         # remember how the mean was defined

    @property
    def mean_constrained_U(self) -> jax.Array:
        U = getattr(self, "_mean_constrained_U", None)
        if U is None:
            raise AttributeError("Call build_mean_constrained() first.")
        return U

    # ---------------------- MEAN-CONSTRAINED EVALUATION (mask optional) ----------------------
    def evaluate_mean_constrained(
        self,
        p_het: jax.Array,
        *,
        apply_prior: bool = True,
        apply_mask: bool = False,
        mask_type: Literal["binary", "smooth"] = "binary",
    ) -> jax.Array:
        """
        theta = U @ p_het; field = prior(basis @ theta); (optionally) mask.
        Returns: (H, W)
        """
        U = self.mean_constrained_U
        if p_het.shape != (U.shape[1],):
            raise ValueError(f"p_het must have shape ({U.shape[1]},), got {p_het.shape}")

        theta = U @ p_het
        field = jnp.tensordot(self._basis, jnp.asarray(theta, self._basis.dtype), axes=([-1], [0]))
        if apply_prior:
            field = self.prior_fn(field)

        if apply_mask:
            mask = self._select_mask(mask_type)
            field = field * mask
        return field

    def make_mean_constrained_evaluator(
        self,
        *,
        apply_prior: bool = True,
        apply_mask: bool = False,
        mask_type: Literal["binary", "smooth"] = "binary",
    ) -> Callable[[jax.Array], jax.Array]:
        U = self.mean_constrained_U
        basis = self._basis
        prior = self.prior_fn
        mask = self._select_mask(mask_type) if apply_mask else None

        @jax.jit
        def f(p_het: jax.Array) -> jax.Array:
            theta = U @ p_het
            fld = jnp.tensordot(basis, theta, axes=([-1], [0]))
            if apply_prior:
                fld = prior(fld)
            if mask is not None:
                fld = fld * mask
            return fld

        return f