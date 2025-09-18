"""
This module contains various Allen-Cahn equation classes.
"""

import dataclasses
from typing import Callable, Optional, Union, Literal, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx

Array = jax.Array

from ..domains import Domain
from .base_eq import BaseEquation
from ..utils.derivatives import _lap_2nd_2D
from ..utils.derivatives import (
    _gradx_c,
    _grady_c,
    _avgx_c2f,
    _avgy_c2f,
    _divx_f2c,
    _divy_f2c,
    _gradx_c2f,
    _grady_c2f,
)


@dataclasses.dataclass
class AllenCahn2DPeriodic(BaseEquation):
    """Allen-Cahn equation in 2D with periodic boundary conditions.

    The Allen-Cahn equation describes phase transitions and interface dynamics.
    The equation is:

    .. math::
        \\frac{\\partial u}{\\partial t} = -R(u) \\mu

    where u is the concentration, R(u) is the reaction term, μ is the chemical potential, and κ is a parameter (the gradient energy coefficient).
    The chemical potential is given by:

    .. math::
        \\mu = \\mu_h(u) - \\kappa \\nabla^2 u
    """

    domain: Domain  # The computational domain for the equation
    """Domain of the equation"""
    kappa: float
    """Gradient energy coefficient"""
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the chemical potential"""
    R: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the reaction term"""
    derivs: str = "fd"
    """Type of derivative computation"""

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn

        if self.derivs == "fourier":
            self.rhs = jax.jit(self.rhs_fourier)
        elif self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fourier(self, state, t):
        state_hat = self.fft(state)
        mu = self.ifft(
            self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        ).real
        return -self.R(state) * mu

    def rhs_fd(self, state, t):
        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        return -self.R(state) * mu


@dataclasses.dataclass
class AllenCahn2DSmoothedBoundary(BaseEquation):
    """Allen-Cahn equation with smoothed boundary method for arbitrary geometries.

    This class implements the Allen-Cahn equation using the smoothed boundary
    method, which allows for complex domain geometries through a smooth
    level-set function ψ.

    The equation is:

    .. math::
        \\frac{\\partial u}{\\partial t} = -R(u) \\mu

    where the chemical potential includes boundary effects:

    .. math::
        \\mu = \\mu_h(u) - \\frac{\\kappa}{\\psi} \\nabla \\cdot (\\psi \\nabla u)
        - \\sqrt{\\kappa} \\frac{|\\nabla \\psi|}{\\psi} \\sqrt{2f} \\cos(\\theta)
    """

    domain: Domain
    """Domain of the equation"""
    kappa: float
    """Gradient energy coefficient"""
    f: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the free energy density"""
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the chemical potential"""
    R: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the reaction term"""
    theta: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the contact angle"""
    derivs: str = "fd"
    """Type of derivative computation"""

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

    def __post_init__(self):
        self.psi = self.domain.geometry.smooth
        self.sqrt_kappa = jnp.sqrt(self.kappa)
        self.hx, self.hy = self.domain.dx
        self.norm_grad_psi = (
            jnp.sqrt(
                _gradx_c(self.psi, self.hx) ** 2 + _grady_c(self.psi, self.hy) ** 2
            )
            / self.psi
        )
        self.left_half = jnp.zeros_like(self.psi)
        self.left_half = self.left_half.at[:, :100].set(1.0)
        if self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fd(self, state, t):
        f = self.f(state)
        mu = self.mu(state)
        mask_avgx = _avgx_c2f(self.psi)
        mask_avgy = _avgy_c2f(self.psi)
        mu += (
            -(self.kappa / self.psi)
            * (
                _divx_f2c(mask_avgx * _gradx_c2f(state, self.hx), self.hx)
                + _divy_f2c(mask_avgy * _grady_c2f(state, self.hy), self.hy)
            )
            - self.sqrt_kappa
            * self.norm_grad_psi
            * jnp.sqrt(2.0 * f)
            * jnp.cos(self.theta(t))
            * self.left_half
        )
        return -self.R(state) * mu


@dataclasses.dataclass
class AllenCahn2DPeriodicButlerVolmer(BaseEquation):
    domain: Domain  # The computational domain for the equation
    """Domain of the equation"""
    kappa: float
    """Gradient energy coefficient"""
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the chemical potential"""
    j0: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the exchange current"""
    alpha: float
    """Symmetry factor"""
    derivs: str = "fd"
    """Type of derivative computation"""

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn

        if self.derivs == "fourier":
            self.rhs = jax.jit(self.rhs_fourier)
        elif self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fourier(self, state, t):
        state_hat = self.fft(state)
        mu = self.ifft(
            self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        ).real
        return -self.R(state) * mu

    def rhs_fd(self, state, t, v):
        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        eta = mu + v
        return self.j0(state) * (
            jnp.exp(-self.alpha * eta) - jnp.exp((1.0 - self.alpha) * eta)
        )


@dataclasses.dataclass
class AllenCahn2DPeriodicButlerVolmerConstantCurrent(BaseEquation):
    domain: Domain  # The computational domain for the equation
    """Domain of the equation"""
    kappa: float
    """Gradient energy coefficient"""
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the chemical potential"""
    j0: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the exchange current"""
    alpha: float
    """Symmetry factor"""
    Crate: float
    """Current"""
    derivs: str = "fd"
    """Type of derivative computation"""

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn

        if self.derivs == "fourier":
            self.rhs = jax.jit(self.rhs_fourier)
        elif self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fourier(self, state, t):
        state_hat = self.fft(state)
        mu = self.ifft(
            self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        ).real
        return -self.R(state) * mu

    def rhs_fd(self, state, t):
        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        int_plus = jnp.sum(self.j0(state) * jnp.exp(0.5 * mu)) * hx * hy
        int_minus = jnp.sum(self.j0(state) * jnp.exp(-0.5 * mu)) * hx * hy
        y = (-self.Crate + jnp.sqrt(self.Crate**2 + 4.0 * int_plus * int_minus)) / (
            2.0 * int_plus
        )
        # Compute v to satisfy constant current constraint
        v = 2.0 * jnp.log(y)
        eta = mu + v
        return self.j0(state) * (
            jnp.exp(-self.alpha * eta) - jnp.exp((1.0 - self.alpha) * eta)
        )

    def get_voltage(self, state):
        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        int_plus = jnp.sum(self.j0(state) * jnp.exp(0.5 * mu)) * hx * hy
        int_minus = jnp.sum(self.j0(state) * jnp.exp(-0.5 * mu)) * hx * hy
        y = (-self.Crate + jnp.sqrt(self.Crate**2 + 4.0 * int_plus * int_minus)) / (
            2.0 * int_plus
        )
        # Compute v to satisfy constant current constraint
        return 2.0 * jnp.log(y)

# assume helpers exist: _avgx_c2f, _avgy_c2f, _gradx_c2f, _grady_c2f, _divx_f2c, _divy_f2c, _gradx_c, _grady_c
@dataclasses.dataclass
class AllenCahn2DSBM_BV_CC(BaseEquation):
    # ----- PDE & geometry -----
    domain: "Domain"
    kappa: float
    f: Union[Callable, eqx.Module]
    mu: Union[Callable, eqx.Module]
    j0: Union[Callable, eqx.Module]
    k0: float
    H: float
    Cmax: float
    Crate: float
    derivs: str = "fd"

    # ----- Prefactor controls -----
    use_prefactor: bool = False
    mean_constrained: bool = False
    # If mean_constrained=False, provide k_eval(theta)->k(x,y).
    # If mean_constrained=True, provide k_eval_mc(theta)->k(x,y) with enforced mean (e.g., normalized).
    k_eval: Optional[Callable[[Array], Array]] = None
    k_eval_mc: Optional[Callable[[Array], Array]] = None
    theta: Optional[Array] = None

    # ----- caches (set in __post_init__) -----
    Nx: int = dataclasses.field(init=False, repr=False)
    Ny: int = dataclasses.field(init=False, repr=False)
    psi: Array = dataclasses.field(init=False, repr=False)
    Lx: float = dataclasses.field(init=False, repr=False)
    Ly: float = dataclasses.field(init=False, repr=False)
    L_ref: float = dataclasses.field(init=False, repr=False)
    hx: float = dataclasses.field(init=False, repr=False)
    hy: float = dataclasses.field(init=False, repr=False)
    dxdy: float = dataclasses.field(init=False, repr=False)
    kappa_over_psi: Array = dataclasses.field(init=False, repr=False)
    mask_avgx: Array = dataclasses.field(init=False, repr=False)
    mask_avgy: Array = dataclasses.field(init=False, repr=False)
    tau_rxn: float = dataclasses.field(init=False, repr=False)
    Itilde: float = dataclasses.field(init=False, repr=False)
    tf_tilde_est: float = dataclasses.field(init=False, repr=False)

    # --------- compiled callables (set in __post_init__) ---------
    mu_sb_fn: Callable = dataclasses.field(init=False, repr=False)
    rhs_fn: Callable = dataclasses.field(init=False, repr=False)
    rhs_with_k_fn: Callable = dataclasses.field(init=False, repr=False)
    get_voltage_fn: Callable = dataclasses.field(init=False, repr=False)
    get_voltage_with_k_fn: Callable = dataclasses.field(init=False, repr=False)
    get_current_fn: Callable = dataclasses.field(init=False, repr=False)
    get_current_with_k_fn: Callable = dataclasses.field(init=False, repr=False)

    # ---------------------- init & setup ----------------------
    def __post_init__(self):
        kB = 8.617333262145e-5; #Boltzmann Constant [eV/K]
        T=298.15; #Temperature [K]
        NA = 6.02214076e23; #Avogadro's Number [molecules/mol]
        # --- geometry / spacing ---
        self.Nx, self.Ny = self.domain.geometry.smooth.shape
        self.psi = self.domain.geometry.smooth
        self.Lx, self.Ly = self.domain.L
        self.L_ref = max(self.Lx, self.Ly)
        hx, hy = self.domain.dx
        self.hx, self.hy = hx / self.L_ref, hy / self.L_ref
        self.dxdy = self.hx * self.hy

        # --- smoothed-boundary scaffolding ---
        eps = 1e-30
        self.kappa= self.kappa / (self.Cmax*(self.L_ref ** 2)*NA*T*kB)  # nondim
        self.kappa_over_psi = self.kappa / jnp.maximum(self.psi, eps)
        self.mask_avgx = _avgx_c2f(self.psi)
        self.mask_avgy = _avgy_c2f(self.psi)

        # --- nondimensional current/time scales ---
        self.tau_rxn = (self.Cmax * self.H / self.k0)
        afrac = jnp.sum(self.domain.geometry.binary) / (self.Nx * self.Ny)
        self.Itilde = afrac * self.tau_rxn / (3600.0 / self.Crate)
        self.tf_tilde_est = 1.0 / (self.Crate * self.tau_rxn / 3600.0)

        if self.derivs != "fd":
            raise ValueError(f"Invalid derivative type: {self.derivs}")

        # ---- compile bound methods with eqx.filter_jit ----
        self.mu_sb_fn               = eqx.filter_jit(self._mu_sb)               # (state) -> mu_sb
        self.rhs_fn                 = eqx.filter_jit(self._rhs_fd_simple)       # (state, t) -> dy/dt (uses self.theta if enabled)
        self.rhs_with_k_fn          = eqx.filter_jit(self._rhs_fd_with_k)       # (state, t, k_field) -> dy/dt
        self.get_voltage_fn         = eqx.filter_jit(self._get_voltage_simple)  # (state) -> scalar
        self.get_voltage_with_k_fn  = eqx.filter_jit(self._get_voltage_with_k)  # (state, k_field) -> scalar
        self.get_current_fn         = eqx.filter_jit(self._get_current_simple)  # (state) -> scalar
        self.get_current_with_k_fn  = eqx.filter_jit(self._get_current_with_k)  # (state, k_field) -> scalar

    # ---------------------- ABC-required methods ----------------------
    # Provide concrete methods so the ABC is satisfied.
    def rhs(self, state: Array, t: float) -> Array:
        # delegate to compiled fn
        return self.rhs_fn(state, t)

    # Optional: keep method names (wrappers) for ergonomic access
    def get_voltage(self, state: Array) -> Array:
        return self.get_voltage_fn(state)

    def get_current(self, state: Array) -> Array:
        return self.get_current_fn(state)

    # If you want explicit-with-k variants as methods too:
    def rhs_with_k(self, state: Array, t: float, k_field: Array) -> Array:
        return self.rhs_with_k_fn(state, t, k_field)

    def get_voltage_with_k(self, state: Array, k_field: Array) -> Array:
        return self.get_voltage_with_k_fn(state, k_field)

    def get_current_with_k(self, state: Array, k_field: Array) -> Array:
        return self.get_current_with_k_fn(state, k_field)

    # ---------------------- internals ----------------------
    def _k_from_theta(self) -> Array:
        if not self.use_prefactor:
            return jnp.ones_like(self.psi)
        if self.mean_constrained:
            if self.k_eval_mc is None or self.theta is None:
                raise ValueError("mean_constrained=True but k_eval_mc/theta not set.")
            return self.k_eval_mc(self.theta)
        else:
            if self.k_eval is None or self.theta is None:
                raise ValueError("use_prefactor=True but k_eval/theta not set.")
            return self.k_eval(self.theta)

    def _mu_sb(self, state: Array) -> Array:
        mu = self.mu(state)
        mu -= self.kappa_over_psi * (
            _divx_f2c(self.mask_avgx * _gradx_c2f(state, self.hx), self.hx)
            + _divy_f2c(self.mask_avgy * _grady_c2f(state, self.hy), self.hy)
        )
        return mu

    def _int_terms(
        self, mu: Array, j0_local: Array, k_field: Array
    ) -> Tuple[Array, Array]:
        w = k_field * j0_local * self.psi
        int_plus  = jnp.sum(w * jnp.exp( 0.5 * mu)) * self.dxdy
        int_minus = jnp.sum(w * jnp.exp(-0.5 * mu)) * self.dxdy
        return int_plus, int_minus

    def _rhs_fd_simple(self, state: Array, t: float) -> Array:
        mu = self.mu_sb_fn(state)                    # compiled call
        j0_local = self.j0(state)
        k_field = self._k_from_theta() if self.use_prefactor else jnp.ones_like(self.psi)
        eps = 1e-30
        int_plus, int_minus = self._int_terms(mu, j0_local, k_field)
        int_plus = jnp.maximum(int_plus, eps)
        y = (-self.Itilde + jnp.sqrt(self.Itilde**2 + 4.0 * int_plus * int_minus)) / (2.0 * int_plus)
        v = 2.0 * jnp.log(jnp.maximum(y, eps))
        eta = mu + v
        return k_field * j0_local * (jnp.exp(-0.5 * eta) - jnp.exp(0.5 * eta))

    def _rhs_fd_with_k(self, state: Array, t: float, k_field: Array) -> Array:
        mu = self.mu_sb_fn(state)
        j0_local = self.j0(state)
        eps = 1e-30
        int_plus, int_minus = self._int_terms(mu, j0_local, k_field)
        int_plus = jnp.maximum(int_plus, eps)
        y = (-self.Itilde + jnp.sqrt(self.Itilde**2 + 4.0 * int_plus * int_minus)) / (2.0 * int_plus)
        v = 2.0 * jnp.log(jnp.maximum(y, eps))
        eta = mu + v
        return k_field * j0_local * (jnp.exp(-0.5 * eta) - jnp.exp(0.5 * eta))

    def _get_voltage_simple(self, state: Array) -> Array:
        mu = self.mu_sb_fn(state)
        j0_local = self.j0(state)
        k_field = self._k_from_theta() if self.use_prefactor else jnp.ones_like(self.psi)
        eps = 1e-30
        int_plus, int_minus = self._int_terms(mu, j0_local, k_field)
        int_plus = jnp.maximum(int_plus, eps)
        y = (-self.Itilde + jnp.sqrt(self.Itilde**2 + 4.0 * int_plus * int_minus)) / (2.0 * int_plus)
        return 2.0 * jnp.log(jnp.maximum(y, eps))

    def _get_voltage_with_k(self, state: Array, k_field: Array) -> Array:
        mu = self.mu_sb_fn(state)
        j0_local = self.j0(state)
        eps = 1e-30
        int_plus, int_minus = self._int_terms(mu, j0_local, k_field)
        int_plus = jnp.maximum(int_plus, eps)
        y = (-self.Itilde + jnp.sqrt(self.Itilde**2 + 4.0 * int_plus * int_minus)) / (2.0 * int_plus)
        return 2.0 * jnp.log(jnp.maximum(y, eps))

    def _get_current_simple(self, state: Array) -> Array:
        v = self._get_voltage_simple(state)
        mu = self.mu_sb_fn(state)
        eta = mu + v
        j0_local = self.j0(state)
        k_field = self._k_from_theta() if self.use_prefactor else jnp.ones_like(self.psi)
        rxn = k_field * j0_local * (jnp.exp(-0.5 * eta) - jnp.exp(0.5 * eta))
        return jnp.sum(rxn * self.psi) * self.dxdy

    def _get_current_with_k(self, state: Array, k_field: Array) -> Array:
        v = self._get_voltage_with_k(state, k_field)
        mu = self.mu_sb_fn(state)
        eta = mu + v
        j0_local = self.j0(state)
        rxn = k_field * j0_local * (jnp.exp(-0.5 * eta) - jnp.exp(0.5 * eta))
        return jnp.sum(rxn * self.psi) * self.dxdy

    # --- diagnostics ---
    def get_SOC(self, state: Array) -> Array:
        binmask = self.domain.geometry.binary
        return jnp.sum(state * binmask) / jnp.sum(binmask)

    def get_cvar(self, state: Array) -> Array:
        binmask = self.domain.geometry.binary
        cbar = self.get_SOC(state)
        return jnp.sum((state - cbar) ** 2 * binmask) / jnp.sum(binmask)

    # --- convenience ---
    def set_theta(self, theta: Array):
        self.theta = jnp.asarray(theta)

    def k_field_from(self, theta: Array) -> Array:
        if not self.use_prefactor:
            return jnp.ones_like(self.psi)
        return (self.k_eval_mc if self.mean_constrained else self.k_eval)(theta)

        
# @dataclasses.dataclass
# class AllenCahn2DSBM_BV_CC(BaseEquation):
#     domain: Domain
#     """Domain of the equation"""
#     kappa: float
#     """Gradient energy coefficient"""
#     f: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
#     """Function for the free energy density"""
#     mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
#     """Function for the chemical potential"""
#     j0: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
#     """Function for the exchange current"""
#     k0: float
#     """Rate constant (mol/(m^2 s))"""
#     H: float
#     """Particle thickness (m)"""
#     Cmax: float
#     """Maximum concentration (mol/m^3)"""
#     Crate: float
#     """C-rate (1/hr)"""
#     derivs: str = "fd"
#     """Type of derivative computation"""

#     def rhs(self, state, t):
#         raise NotImplementedError("rhs method not implemented")

#     def __post_init__(self):
#         self.Nx, self.Ny = self.domain.geometry.smooth.shape
#         self.psi = self.domain.geometry.smooth
#         self.sqrt_kappa = jnp.sqrt(self.kappa)
#         self.Lx, self.Ly = self.domain.L
#         self.L_ref = max(self.Lx, self.Ly)
#         self.hx, self.hy = self.domain.dx
#         self.hx, self.hy = self.hx / self.L_ref, self.hy / self.L_ref 
#         self.norm_grad_psi = (
#             jnp.sqrt(
#                 _gradx_c(self.psi, self.hx) ** 2 + _grady_c(self.psi, self.hy) ** 2
#             )
#             / self.psi
#         )
#         self.left_half = jnp.zeros_like(self.psi)
#         self.left_half = self.left_half.at[:, :100].set(1.0)
#         self.tau_rxn = (self.Cmax*self.H/self.k0)
#         self.afrac = jnp.sum(self.domain.geometry.binary)/(self.Nx*self.Ny)
#         self.Itilde = self.afrac * (self.tau_rxn)/(3600/self.Crate)
#         self.tf_tilde_est = 1/(self.Crate*self.tau_rxn/3600)
#         if self.derivs == "fd":
#             self.rhs = jax.jit(self.rhs_fd)
#         else:
#             raise ValueError(f"Invalid derivative type: {self.derivs}")

#     def rhs_fd(self, state, t):
#         # f = self.f(state)
#         mu = self.mu(state)
#         mask_avgx = _avgx_c2f(self.psi)
#         mask_avgy = _avgy_c2f(self.psi)
#         mu += (
#             -(self.kappa / self.psi)
#             * (
#                 _divx_f2c(mask_avgx * _gradx_c2f(state, self.hx), self.hx)
#                 + _divy_f2c(mask_avgy * _grady_c2f(state, self.hy), self.hy)
#             )
#             # - self.sqrt_kappa
#             # * self.norm_grad_psi
#             # * jnp.sqrt(2.0 * f)
#             # * jnp.cos(self.theta(t))
#             # * self.left_half
#         )
#         int_plus = (
#             jnp.sum(self.j0(state) * jnp.exp(0.5 * mu) * self.psi) * self.hx * self.hy
#         )
#         int_minus = (
#             jnp.sum(self.j0(state) * jnp.exp(-0.5 * mu) * self.psi) * self.hx * self.hy
#         )
#         y = (-self.Itilde + jnp.sqrt(self.Itilde**2 + 4.0 * int_plus * int_minus)) / (
#             2.0 * int_plus
#         )
#         # Compute v to satisfy constant current constraint
#         v = 2.0 * jnp.log(y)
#         eta = mu + v
#         return self.j0(state) * (
#             jnp.exp(-0.5 * eta) - jnp.exp(0.5 * eta)
#         )

#     def get_voltage(self, state):
#         # f = self.f(state)
#         mu = self.mu(state)
#         mask_avgx = _avgx_c2f(self.psi)
#         mask_avgy = _avgy_c2f(self.psi)
#         mu += (
#             -(self.kappa / self.psi)
#             * (
#                 _divx_f2c(mask_avgx * _gradx_c2f(state, self.hx), self.hx)
#                 + _divy_f2c(mask_avgy * _grady_c2f(state, self.hy), self.hy)
#             )
#             # - self.sqrt_kappa
#             # * self.norm_grad_psi
#             # * jnp.sqrt(2.0 * f)
#             # * jnp.cos(self.theta(t))
#             # * self.left_half
#         )
#         int_plus = (
#             jnp.sum(self.j0(state) * jnp.exp(0.5 * mu) * self.psi) * self.hx * self.hy
#         )
#         int_minus = (
#             jnp.sum(self.j0(state) * jnp.exp(-0.5 * mu) * self.psi) * self.hx * self.hy
#         )
#         y = (-self.Itilde + jnp.sqrt(self.Itilde**2 + 4.0 * int_plus * int_minus)) / (
#             2.0 * int_plus
#         )
#         # Compute v to satisfy constant current constraint
#         return 2.0 * jnp.log(y)
    
#     def get_current(self,state):
#         # f = self.f(state)
#         mu = self.mu(state)
#         mask_avgx = _avgx_c2f(self.psi)
#         mask_avgy = _avgy_c2f(self.psi)
#         mu += (
#             -(self.kappa / self.psi)
#             * (
#                 _divx_f2c(mask_avgx * _gradx_c2f(state, self.hx), self.hx)
#                 + _divy_f2c(mask_avgy * _grady_c2f(state, self.hy), self.hy)
#             )
#             # - self.sqrt_kappa
#             # * self.norm_grad_psi
#             # * jnp.sqrt(2.0 * f)
#             # * jnp.cos(self.theta(t))
#             # * self.left_half
#         )
#         v = self.get_voltage(state)
#         eta = mu + v
#         rxn = self.j0(state) * (jnp.exp(-0.5 * eta) - jnp.exp(0.5 * eta))
#         return jnp.sum(rxn*self.psi)*self.hx*self.hy

#     def get_SOC(self,state):
#         return jnp.sum(state * self.domain.geometry.binary)/jnp.sum(self.domain.geometry.binary)

#     def get_cvar(self,state):
#         return jnp.sum((state - self.get_SOC(state))**2 * self.domain.geometry.binary)/jnp.sum(self.domain.geometry.binary)
