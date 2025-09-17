"""Custom numerical solvers for partial differential equations.

This module provides specialized numerical solvers that extend the diffrax library
for solving specific types of PDEs. The solvers are designed to work with the
equation classes defined in the numerics.equations module.

Available Solvers:
    SemiImplicitFourierSpectral: Semi-implicit Fourier spectral method.

    StrangSplitting: Strang splitting method for equations with separable operators
        (e.g., Gross-Pitaevskii equation).

All solvers inherit from diffrax.AbstractSolver and are compatible with the
diffrax integration framework.
"""

import diffrax as dfx
import jax
import jax.numpy as jnp
from typing import Callable
import equinox as eqx
from jax.flatten_util import ravel_pytree

from .rock2_tableau import get_tableau_jax
from equinox.internal import while_loop

class SemiImplicitFourierSpectral(dfx.AbstractSolver):
    """Semi-implicit Fourier spectral method.

    This solver implements a semi-implicit Fourier spectral method for phase-field simulations with variable mobility.

    Required Equation Attributes:
        fourier_symbol: Fourier space representation of the highest order differential operator.
        fft: Forward Fourier transform function.
        ifft: Inverse Fourier transform function.

    Parameters:
        A (float): Constant for splitting the mobility term.

    References:
        Zhu, Jingzhi, et al. "Coarsening kinetics from a variable-mobility Cahn-Hilliard
        equation: Application of a semi-implicit Fourier spectral method." Physical Review E
        60.4 (1999): 3564.
    """

    required_equation_attrs = ["fourier_symbol", "fft", "ifft"]
    A: float
    fourier_symbol: jax.Array
    fft: Callable
    ifft: Callable
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0
        f0 = terms.vf(t0, y0, args)

        euler_y1 = y0 + δt * f0
        tmp = 1.0 + self.A * δt * self.fourier_symbol
        y1 = y0 + δt * self.ifft(self.fft(f0) / tmp).real

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class StrangSplitting(dfx.AbstractSolver):
    """Strang splitting method for time-dependent PDEs with separable operators.

    References:
        Bao, Weizhu, and Yongyong Cai. "Mathematical theory and numerical methods for
        Bose-Einstein condensation." arXiv preprint arXiv:1212.5341 (2012).
    """

    required_equation_attrs = ["A_term", "dx", "fft", "ifft"]
    A_term: jax.Array
    dx: float
    fft: Callable
    ifft: Callable
    time_scale: float
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = (t1 - t0) * self.time_scale

        y0_ = y0[..., 0] + 1j * y0[..., 1]

        exp_A_term = jnp.exp(self.A_term * 0.5 * δt)

        tmp = self.fft(y0_) * exp_A_term  # 1
        tmp = self.ifft(tmp)  # 2
        b_term = terms.vf(t0, y0, args)
        tmp = tmp * jnp.exp((b_term[..., 0] + 1j * b_term[..., 1]) * δt)  # 3
        tmp /= jnp.sqrt(jnp.sum(jnp.abs(tmp) ** 2) * self.dx**2)  # 4
        tmp = self.fft(tmp)  # 5
        tmp = tmp * exp_A_term  # 6
        y1_ = self.ifft(tmp)  # 7
        y1 = jnp.stack([y1_.real, y1_.imag], axis=-1)
        # TODO: I should be able to change order and reduce number of ffts

        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, None, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return NotImplementedError
# Load tableau once. (We won't mutate these.)
_MS, _FP1, _FP2, _RECF, _OFFSETS, _SIZES = get_tableau_jax()
_MS_BOUNDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    22, 24, 26, 28, 30, 33, 36, 39, 43, 47, 51, 56, 61, 66, 72, 78, 85, 93,
    102, 112, 123, 135, 148, 163, 180, 198
)

# Maximum supported stages (from your ms grid).
MAX_S = 198  # keep static for JIT (matches ms[-1])


def _estimate_spectral_radius_fd(term: dfx.ODETerm, t, y, args, eps=1e-6, n_iters=3):
    """
    Matrix-free spectral radius proxy using finite differences in a
    power-iteration-like loop, but implemented on a flat PyTree.
    """
    y_flat, unravel = ravel_pytree(y)

    def f_flat(yf):
        y_ = unravel(yf)
        fy = term.vf(t, y_, args)
        fy_flat, _ = ravel_pytree(fy)
        return fy_flat

    fy0 = f_flat(y_flat)
    v = jnp.ones_like(y_flat)
    v = v / (jnp.linalg.norm(v) + 1e-30)

    def body(_, v):
        Jv = (f_flat(y_flat + eps * v) - fy0) / eps
        nrm = jnp.linalg.norm(Jv)
        v = jnp.where(nrm > 0, Jv / nrm, v)
        return v

    v = jax.lax.fori_loop(0, n_iters, body, v)
    Jv = (f_flat(y_flat + eps * v) - fy0) / eps
    return jnp.linalg.norm(Jv)


def _analytic_params(s):
    """
    Analytic RKC2 params shared across stages (Chebyshev damping).
    Returns (c0, fp1, fp2, omega0, omega1).
    """
    s = jnp.maximum(s, 1)
    delta = jnp.array(0.2)
    theta = jnp.arccosh(1.0 + delta / (s * s))
    omega0 = jnp.cosh(s * theta) / jnp.cosh(jnp.maximum(s - 1, 1) * theta)
    omega1 = jnp.where(s >= 3, jnp.cosh((s - 1) * theta) / jnp.cosh((s - 2) * theta), 1.0)
    c0 = 1.0 / omega0
    fp1 = jnp.array(0.36792)
    fp2 = jnp.array(0.38433)
    return c0, fp1, fp2, omega0, omega1


def _analytic_mu_kappa(i, omega0, omega1):
    """Per-stage (μ_i, κ_i) for analytic fallback."""
    a_i = jnp.where(i == 1, 1.0 / omega0, 2.0 * omega0 / jnp.where(omega1 == 0, 1.0, omega1))
    b_i = jnp.where(i <= 2, 0.0, (omega0 / jnp.where(omega1 == 0, 1.0, omega1)) ** 2)
    # We only ever use i>=2 inside the loop; still fine to return both.
    return a_i, b_i

class ROCK2JAX(dfx.AbstractSolver):
    # Diffrax interface
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 2

    # User-settable knobs (Equinox-style fields; no manual __init__)
    min_stages: int = 0
    max_stages: int = 200
    eigen_estimator: object = None  # optional: (terms, t, y, args) -> scalar

    # Cached arrays (fresh copies via default_factory)
    ms: jax.Array = eqx.field(default_factory=lambda: _MS)
    fp1_tab: jax.Array = eqx.field(default_factory=lambda: _FP1)
    fp2_tab: jax.Array = eqx.field(default_factory=lambda: _FP2)
    recf: jax.Array | None = eqx.field(default_factory=lambda: _RECF)
    offsets: jax.Array | None = eqx.field(default_factory=lambda: _OFFSETS)
    sizes: jax.Array | None = eqx.field(default_factory=lambda: _SIZES)

    # Required by some diffrax versions
    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def init(self, terms, t0, t1, y0, args):
        return None

    def _choose_stage_index(self, dt, eigen_est):
        # Raw stage suggestion
        mdeg = jnp.floor(jnp.sqrt((1.5 + jnp.abs(dt) * eigen_est) / 0.811)) + 1.0
        s = jnp.clip(mdeg, self.min_stages, self.max_stages)
        s = jnp.maximum(s, 3.0).astype(jnp.int32)
        # Snap to discrete grid ms
        idx = jnp.searchsorted(self.ms, s, side="left")
        idx = jnp.clip(idx, 0, self.ms.shape[0] - 1)
        s_eff = self.ms[idx].astype(jnp.int32)
        return s_eff, idx

    def _get_c0_fp(self, s_eff, idx):
        """
        Return (c0, fp1, fp2) for chosen stage-count s_eff from tableau or analytic fallback.
        """
        if self.recf is None:
            # print("Using analytic ROCK2 coefficients")
            c0, fp1, fp2, _, _ = _analytic_params(s_eff)
            return c0, fp1, fp2
        start = self.offsets[idx]
        c0 = self.recf[start]
        fp1 = self.fp1_tab[idx]
        fp2 = self.fp2_tab[idx]
        return c0, fp1, fp2
    # def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
    #     del made_jump
    #     dt = t1 - t0

    #     # Eigenvalue bound
    #     eigen_est = (self.eigen_estimator(terms, t0, y0, args)
    #                  if self.eigen_estimator is not None
    #                  else _estimate_spectral_radius_fd(terms, t0, y0, args))

    #     # Choose stage count and its ms index
    #     s_eff, idx = self._choose_stage_index(dt, eigen_est)

    #     # First-stage abscissa + finishing weights
    #     c0, fp1, fp2 = self._get_c0_fp(s_eff, idx)

    #     f = terms.vf

    #     # ----- Stage 1 (Euler-like) -----
    #     fy = f(t0, y0, args)
    #     u_im2 = y0
    #     u_im1 = jax.tree_util.tree_map(lambda yy, ff: yy + (dt * c0) * ff, y0, fy)
    #     u = u_im1
    #     t_im1 = t0 + dt * c0
    #     t_im2 = t0 + dt * c0
    #     t_im3 = t0

    #     # ----- Coefficient accessors for internal stages -----
    #     use_recf = (self.recf is not None)  # Python bool (static to trace)

    #     if use_recf:
    #         start = self.offsets[idx]  # tracer scalar tied to idx

    #         def get_mu_kap(i):
    #             # recf layout: [c0, μ₂, κ₂, μ₃, κ₃, ...] per stage
    #             pos = start + 1 + 2 * (i - 2)
    #             mu_i = self.recf[pos]
    #             kap_i = self.recf[pos + 1]
    #             return mu_i, kap_i
    #     else:
    #         # Analytic fallback depends on s_eff
    #         _, _, _, omega0, omega1 = _analytic_params(s_eff)

    #         def get_mu_kap(i):
    #             a_i = jnp.where(i == 1, 1.0 / omega0,
    #                             2.0 * omega0 / jnp.where(omega1 == 0, 1.0, omega1))
    #             b_i = jnp.where(i <= 2, 0.0,
    #                             (omega0 / jnp.where(omega1 == 0, 1.0, omega1)) ** 2)
    #             return a_i, b_i

    #     # ----- Internal stages: reverse-mode–safe static loop via lax.switch -----
    #     # Build 46 branches locally; each has compile-time-constant bounds [2, s_bound+1)
    #     def _make_branch(s_bound: int):
    #         def _run_branch(carry):
    #             def _stage_body(i, carry_):
    #                 u_im2_c, u_im1_c, u_c, t_im1_c, t_im2_c, t_im3_c = carry_
    #                 mu_i, kap_i = get_mu_kap(i)
    #                 nu_i = -1.0 - kap_i

    #                 # one RHS per stage
    #                 fu_im1 = f(t_im1_c, u_im1_c, args)

    #                 # time-like recurrence
    #                 t_next = dt * mu_i - nu_i * t_im2_c - kap_i * t_im3_c

    #                 # state recurrence
    #                 u_next = jax.tree_util.tree_map(
    #                     lambda fi, u1, u2: (dt * mu_i) * fi - nu_i * u1 - kap_i * u2,
    #                     fu_im1, u_im1_c, u_im2_c
    #                 )

    #                 # slide window
    #                 return (u_im1_c, u_next, u_next, t_next, t_im1_c, t_im2_c)

    #             # Runs i=2..s_bound inclusive; if s_bound < 2, zero iters
    #             return jax.lax.fori_loop(2, s_bound + 1, _stage_body, carry)
    #         return _run_branch

    #     branches = tuple(_make_branch(int(s)) for s in _MS_BOUNDS)
    #     carry0 = (u_im2, u_im1, u, t_im1, t_im2, t_im3)

    #     # Select and execute the correct static-bounded loop (idx matches your snapped grid)
    #     u_im2, u_im1, u, t_im1, t_im2, t_im3 = jax.lax.switch(idx, branches, carry0)

    #     # ----- Two-stage finishing -----
    #     dt1 = dt * fp1
    #     dt2 = dt * fp2

    #     f_u = f(t_im1, u, args)
    #     u_im1 = jax.tree_util.tree_map(lambda uu, fu: uu + dt1 * fu, u, f_u)
    #     t_im1 = t_im1 + dt1
    #     f_u2 = f(t_im1, u_im1, args)

    #     tmp = jax.tree_util.tree_map(lambda fu2, fu: dt2 * (fu2 - fu), f_u2, f_u)
    #     y1 = jax.tree_util.tree_map(lambda uim1, fu2, tm: uim1 + dt1 * fu2 + tm, u_im1, f_u2, tmp)
    #     y_error = tmp

    #     dense_info = {"y0": y0, "y1": y1}
    #     result = dfx.RESULTS.successful
    #     return y1, y_error, dense_info, solver_state, result
    
    # def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
    #     del made_jump
    #     dt = t1 - t0

    #     # Eigenvalue bound
    #     eigen_est = (self.eigen_estimator(terms, t0, y0, args)
    #                 if self.eigen_estimator is not None
    #                 else _estimate_spectral_radius_fd(terms, t0, y0, args))

    #     # Choose stage count s and its ms index
    #     s_eff, idx = self._choose_stage_index(dt, eigen_est)
    #     # jax.debug.print(
    #     #     "Using s_eff={} stages (idx={}) for dt={}, eigen_est={}",
    #     #     s_eff, idx, dt, eigen_est
    #     # )
    #     # First-stage abscissa + finishing weights
    #     c0, fp1, fp2 = self._get_c0_fp(s_eff, idx)

    #     f = terms.vf

    #     # Stage 1 (Euler-like)
    #     fy = f(t0, y0, args)
    #     u_im2 = y0
    #     u_im1 = jax.tree_util.tree_map(lambda yy, ff: yy + (dt * c0) * ff, y0, fy)
    #     u = u_im1
    #     t_im1 = t0 + dt * c0
    #     t_im2 = t0 + dt * c0
    #     t_im3 = t0

    #     # Prepare constants for loop
    #     if self.recf is not None:
    #         start = self.offsets[idx]  # JAX scalar

    #         def get_mu_kap(i):
    #             pos = start + 1 + 2 * (i - 2)
    #             mu_i = self.recf[pos]
    #             kap_i = self.recf[pos + 1]
    #             return mu_i, kap_i
    #     else:
    #         # Analytic: precompute omega0/omega1 once for this s_eff
    #         _, _, _, omega0, omega1 = _analytic_params(s_eff)

    #         def get_mu_kap(i):
    #             a_i = jnp.where(i == 1, 1.0 / omega0,
    #                             2.0 * omega0 / jnp.where(omega1 == 0, 1.0, omega1))
    #             b_i = jnp.where(i <= 2, 0.0,
    #                             (omega0 / jnp.where(omega1 == 0, 1.0, omega1)) ** 2)
    #             return a_i, b_i

    #     # -----------------------------
    #     # Internal Chebyshev stages: i = 2 .. s_eff  (inclusive)
    #     # -> fori_loop runs [lower, upper), so use upper = s_eff + 1
    #     # -----------------------------
    #     def _stage_body(i, carry):
    #         u_im2_c, u_im1_c, u_c, t_im1_c, t_im2_c, t_im3_c = carry
    #         mu_i, kap_i = get_mu_kap(i)
    #         nu_i = -1.0 - kap_i

    #         fu_im1 = f(t_im1_c, u_im1_c, args)  # one RHS per internal stage
    #         t_next = dt * mu_i - nu_i * t_im2_c - kap_i * t_im3_c
    #         u_next = jax.tree_util.tree_map(
    #             lambda fi, u1, u2: (dt * mu_i) * fi - nu_i * u1 - kap_i * u2,
    #             fu_im1, u_im1_c, u_im2_c
    #         )

    #         # Slide window (match while_loop semantics)
    #         return (u_im1_c, u_next, u_next, t_next, t_im1_c, t_im2_c)

    #     carry0 = (u_im2, u_im1, u, t_im1, t_im2, t_im3)
    #     u_im2, u_im1, u, t_im1, t_im2, t_im3 = jax.lax.fori_loop(
    #         2, s_eff + 1, _stage_body, carry0
    #     )
    #     i = s_eff + 1  # matches while_loop's i after exit

    #     # Two-stage finishing
    #     dt1 = dt * fp1
    #     dt2 = dt * fp2

    #     f_u = f(t_im1, u, args)
    #     u_im1 = jax.tree_util.tree_map(lambda uu, fu: uu + dt1 * fu, u, f_u)
    #     t_im1 = t_im1 + dt1
    #     f_u2 = f(t_im1, u_im1, args)

    #     tmp = jax.tree_util.tree_map(lambda fu2, fu: dt2 * (fu2 - fu), f_u2, f_u)
    #     y1 = jax.tree_util.tree_map(lambda uim1, fu2, tm: uim1 + dt1 * fu2 + tm, u_im1, f_u2, tmp)
    #     y_error = tmp

    #     dense_info = {"y0": y0, "y1": y1}
    #     result = dfx.RESULTS.successful
    #     return y1, y_error, dense_info, solver_state, result

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del made_jump
        dt = t1 - t0

        # Eigenvalue bound
        eigen_est = (self.eigen_estimator(terms, t0, y0, args)
                     if self.eigen_estimator is not None
                     else _estimate_spectral_radius_fd(terms, t0, y0, args))

        # Choose stage count s and its ms index
        s_eff, idx = self._choose_stage_index(dt, eigen_est)
        # jax.debug.print(
        #     "Using s_eff={} stages (idx={}) for dt={}, eigen_est={}",
        #     s_eff, idx, dt, eigen_est
        # )

        # First-stage abscissa + finishing weights
        c0, fp1, fp2 = self._get_c0_fp(s_eff, idx)

        f = terms.vf

        # Stage 1 (Euler-like)
        fy = f(t0, y0, args)
        u_im2 = y0
        u_im1 = jax.tree_util.tree_map(lambda yy, ff: yy + (dt * c0) * ff, y0, fy)
        u = u_im1
        t_im1 = t0 + dt * c0
        t_im2 = t0 + dt * c0
        t_im3 = t0

        # Prepare constants for loop
        if self.recf is not None:
            start = self.offsets[idx]  # JAX scalar

            def get_mu_kap(i):
                pos = start + 1 + 2 * (i - 2)
                mu_i = self.recf[pos]
                kap_i = self.recf[pos + 1]
                return mu_i, kap_i
        else:
            # Analytic: precompute omega0/omega1 once for this s_eff
            _, _, _, omega0, omega1 = _analytic_params(s_eff)

            def get_mu_kap(i):
                # i is int32 JAX scalar
                a_i = jnp.where(i == 1, 1.0 / omega0, 2.0 * omega0 / jnp.where(omega1 == 0, 1.0, omega1))
                b_i = jnp.where(i <= 2, 0.0, (omega0 / jnp.where(omega1 == 0, 1.0, omega1)) ** 2)
                return a_i, b_i

        # Internal Chebyshev stages: i = 2 .. (s_eff - 2)
        # def cond_fun(carry):
        #     i, *_ = carry
        #     return i <= (s_eff - 2)
        def cond_fun(carry):
            i, *_ = carry
            return i <= s_eff


        def body_fun(carry):
            i, u_im2, u_im1, u, t_im1, t_im2, t_im3 = carry
            mu_i, kap_i = get_mu_kap(i)
            nu_i = -1.0 - kap_i

            fu_im1 = f(t_im1, u_im1, args)  # one RHS per internal stage
            t_next = dt * mu_i - nu_i * t_im2 - kap_i * t_im3
            u_next = jax.tree_util.tree_map(
                lambda fi, u1, u2: (dt * mu_i) * fi - nu_i * u1 - kap_i * u2,
                fu_im1, u_im1, u_im2
            )

            # Slide window
            return (i + 1, u_im1, u_next, u_next, t_next, t_im1, t_im2)

        (i, u_im2, u_im1, u, t_im1, t_im2, t_im3) = while_loop(
            cond_fun,
            body_fun,
            (jnp.array(2, dtype=jnp.int32), u_im2, u_im1, u, t_im1, t_im2, t_im3),
            kind="bounded",
            max_steps=MAX_S,
            base=16,
        )

        # Two-stage finishing
        dt1 = dt * fp1
        dt2 = dt * fp2

        f_u = f(t_im1, u, args)
        u_im1 = jax.tree_util.tree_map(lambda uu, fu: uu + dt1 * fu, u, f_u)
        t_im1 = t_im1 + dt1
        f_u2 = f(t_im1, u_im1, args)

        tmp = jax.tree_util.tree_map(lambda fu2, fu: dt2 * (fu2 - fu), f_u2, f_u)
        y1 = jax.tree_util.tree_map(lambda uim1, fu2, tm: uim1 + dt1 * fu2 + tm, u_im1, f_u2, tmp)
        y_error = tmp

        dense_info = {"y0": y0, "y1": y1}
        result = dfx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

#----------------------------------------------------------------------
# # Load tableau once. (We won't mutate these.)
# _MS, _FP1, _FP2, _RECF, _OFFSETS, _SIZES = get_tableau_jax()

# # Maximum supported stages (from your ms grid).
# MAX_S = 198  # keep static for JIT (matches ms[-1])


# def _estimate_spectral_radius_fd(term: dfx.ODETerm, t, y, args, eps=1e-6, n_iters=3):
#     """
#     Matrix-free spectral radius proxy using finite differences in a
#     power-iteration-like loop, but implemented on a flat PyTree.
#     """
#     y_flat, unravel = ravel_pytree(y)

#     def f_flat(yf):
#         y_ = unravel(yf)
#         fy = term.vf(t, y_, args)
#         fy_flat, _ = ravel_pytree(fy)
#         return fy_flat

#     fy0 = f_flat(y_flat)
#     v = jnp.ones_like(y_flat)
#     v = v / (jnp.linalg.norm(v) + 1e-30)

#     def body(_, v):
#         Jv = (f_flat(y_flat + eps * v) - fy0) / eps
#         nrm = jnp.linalg.norm(Jv)
#         v = jnp.where(nrm > 0, Jv / nrm, v)
#         return v

#     v = jax.lax.fori_loop(0, n_iters, body, v)
#     Jv = (f_flat(y_flat + eps * v) - fy0) / eps
#     return jnp.linalg.norm(Jv)


# def _analytic_params(s):
#     """
#     Analytic RKC2 params shared across stages (Chebyshev damping).
#     Returns (c0, fp1, fp2, omega0, omega1).
#     """
#     s = jnp.maximum(s, 1)
#     delta = jnp.array(0.2)
#     theta = jnp.arccosh(1.0 + delta / (s * s))
#     omega0 = jnp.cosh(s * theta) / jnp.cosh(jnp.maximum(s - 1, 1) * theta)
#     omega1 = jnp.where(s >= 3, jnp.cosh((s - 1) * theta) / jnp.cosh((s - 2) * theta), 1.0)
#     c0 = 1.0 / omega0
#     fp1 = jnp.array(0.36792)
#     fp2 = jnp.array(0.38433)
#     return c0, fp1, fp2, omega0, omega1


# def _analytic_mu_kappa(i, omega0, omega1):
#     """Per-stage (μ_i, κ_i) for analytic fallback."""
#     a_i = jnp.where(i == 1, 1.0 / omega0, 2.0 * omega0 / jnp.where(omega1 == 0, 1.0, omega1))
#     b_i = jnp.where(i <= 2, 0.0, (omega0 / jnp.where(omega1 == 0, 1.0, omega1)) ** 2)
#     # We only ever use i>=2 inside the loop; still fine to return both.
#     return a_i, b_i

# class ROCK2JAX(dfx.AbstractSolver):
#     # Diffrax interface
#     term_structure = dfx.ODETerm
#     interpolation_cls = dfx.LocalLinearInterpolation

#     def order(self, terms):
#         return 2

#     # User-settable knobs (Equinox-style fields; no manual __init__)
#     min_stages: int = 0
#     max_stages: int = 200
#     eigen_estimator: object = None  # optional: (terms, t, y, args) -> scalar

#     # Cached arrays (fresh copies via default_factory)
#     ms: jax.Array = eqx.field(default_factory=lambda: _MS)
#     fp1_tab: jax.Array = eqx.field(default_factory=lambda: _FP1)
#     fp2_tab: jax.Array = eqx.field(default_factory=lambda: _FP2)
#     recf: jax.Array | None = eqx.field(default_factory=lambda: _RECF)
#     offsets: jax.Array | None = eqx.field(default_factory=lambda: _OFFSETS)
#     sizes: jax.Array | None = eqx.field(default_factory=lambda: _SIZES)

#     # Required by some diffrax versions
#     def func(self, terms, t0, y0, args):
#         return terms.vf(t0, y0, args)

#     def init(self, terms, t0, t1, y0, args):
#         return None

#     def _choose_stage_index(self, dt, eigen_est):
#         # Raw stage suggestion
#         mdeg = jnp.floor(jnp.sqrt((1.5 + jnp.abs(dt) * eigen_est) / 0.811)) + 1.0
#         s = jnp.clip(mdeg, self.min_stages, self.max_stages)
#         s = jnp.maximum(s, 3.0).astype(jnp.int32)
#         # Snap to discrete grid ms
#         idx = jnp.searchsorted(self.ms, s, side="left")
#         idx = jnp.clip(idx, 0, self.ms.shape[0] - 1)
#         s_eff = self.ms[idx].astype(jnp.int32)
#         return s_eff, idx

#     def _get_c0_fp(self, s_eff, idx):
#         """
#         Return (c0, fp1, fp2) for chosen stage-count s_eff from tableau or analytic fallback.
#         """
#         if self.recf is None:
#             # print("Using analytic ROCK2 coefficients")
#             c0, fp1, fp2, _, _ = _analytic_params(s_eff)
#             return c0, fp1, fp2
#         start = self.offsets[idx]
#         c0 = self.recf[start]
#         fp1 = self.fp1_tab[idx]
#         fp2 = self.fp2_tab[idx]
#         return c0, fp1, fp2

#     def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
#         del made_jump
#         dt = t1 - t0

#         # Eigenvalue bound
#         eigen_est = (self.eigen_estimator(terms, t0, y0, args)
#                      if self.eigen_estimator is not None
#                      else _estimate_spectral_radius_fd(terms, t0, y0, args))

#         # Choose stage count s and its ms index
#         s_eff, idx = self._choose_stage_index(dt, eigen_est)

#         # First-stage abscissa + finishing weights
#         c0, fp1, fp2 = self._get_c0_fp(s_eff, idx)

#         f = terms.vf

#         # Stage 1 (Euler-like)
#         fy = f(t0, y0, args)
#         u_im2 = y0
#         u_im1 = jax.tree_util.tree_map(lambda yy, ff: yy + (dt * c0) * ff, y0, fy)
#         u = u_im1
#         t_im1 = t0 + dt * c0
#         t_im2 = t0 + dt * c0
#         t_im3 = t0

#         # Prepare constants for loop
#         if self.recf is not None:
#             start = self.offsets[idx]  # JAX scalar

#             def get_mu_kap(i):
#                 pos = start + 1 + 2 * (i - 2)
#                 mu_i = self.recf[pos]
#                 kap_i = self.recf[pos + 1]
#                 return mu_i, kap_i
#         else:
#             # Analytic: precompute omega0/omega1 once for this s_eff
#             _, _, _, omega0, omega1 = _analytic_params(s_eff)

#             def get_mu_kap(i):
#                 # i is int32 JAX scalar
#                 a_i = jnp.where(i == 1, 1.0 / omega0, 2.0 * omega0 / jnp.where(omega1 == 0, 1.0, omega1))
#                 b_i = jnp.where(i <= 2, 0.0, (omega0 / jnp.where(omega1 == 0, 1.0, omega1)) ** 2)
#                 return a_i, b_i

#         # Internal Chebyshev stages: i = 2 .. (s_eff - 2)
#         # def cond_fun(carry):
#         #     i, *_ = carry
#         #     return i <= (s_eff - 2)
#         def cond_fun(carry):
#             i, *_ = carry
#             return i <= s_eff


#         def body_fun(carry):
#             i, u_im2, u_im1, u, t_im1, t_im2, t_im3 = carry
#             mu_i, kap_i = get_mu_kap(i)
#             nu_i = -1.0 - kap_i

#             fu_im1 = f(t_im1, u_im1, args)  # one RHS per internal stage
#             t_next = dt * mu_i - nu_i * t_im2 - kap_i * t_im3
#             u_next = jax.tree_util.tree_map(
#                 lambda fi, u1, u2: (dt * mu_i) * fi - nu_i * u1 - kap_i * u2,
#                 fu_im1, u_im1, u_im2
#             )

#             # Slide window
#             return (i + 1, u_im1, u_next, u_next, t_next, t_im1, t_im2)

#         (i, u_im2, u_im1, u, t_im1, t_im2, t_im3) = jax.lax.while_loop(
#             cond_fun,
#             body_fun,
#             (jnp.array(2, dtype=jnp.int32), u_im2, u_im1, u, t_im1, t_im2, t_im3)
#         )

#         # Two-stage finishing
#         dt1 = dt * fp1
#         dt2 = dt * fp2

#         f_u = f(t_im1, u, args)
#         u_im1 = jax.tree_util.tree_map(lambda uu, fu: uu + dt1 * fu, u, f_u)
#         t_im1 = t_im1 + dt1
#         f_u2 = f(t_im1, u_im1, args)

#         tmp = jax.tree_util.tree_map(lambda fu2, fu: dt2 * (fu2 - fu), f_u2, f_u)
#         y1 = jax.tree_util.tree_map(lambda uim1, fu2, tm: uim1 + dt1 * fu2 + tm, u_im1, f_u2, tmp)
#         y_error = tmp

#         dense_info = {"y0": y0, "y1": y1}
#         result = dfx.RESULTS.successful
#         return y1, y_error, dense_info, solver_state, result
