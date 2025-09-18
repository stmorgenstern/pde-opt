import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
import diffrax as dfx
import diffrax as diffrax
from PIL import Image
import numpy as np
from pde_opt.numerics.domains import Domain
from pde_opt.numerics.shapes import Shape,SpatialHeterogeneity
# from pde_opt.numerics.equations.allen_cahn import AllenCahn2DSmoothedBoundaryButlerVolmerConstantCurrent
from pde_opt.numerics.equations.allen_cahn import AllenCahn2DSBM_BV_CC
from pde_opt.pde_model import PDEModel
from pde_opt.numerics.solvers import ROCK2JAX
import optimistix as optx   # pip install optimistix
import equinox as eqx

def gen_c0(shape, key, noise_fac: float = 0.05, nucsite: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Generate initial condition array given a Shape.

    Args:
        shape: Shape instance (must have .binary).
        key: JAX PRNGKey
        noise_fac: scale of uniform random perturbation in [-1, 1]
        nucsite: array of nucleation values (same shape as shape.binary),
                 applied only inside the mask. Default: zeros.

    Returns:
        c0: initial concentration array, same shape as shape.binary
    """
    psi_b = shape.binary.astype(jnp.float32)
    H, W = psi_b.shape

    # Uniform noise ∈ [-1, 1], masked by psi_b
    noise = noise_fac * random.uniform(key, (H, W), minval=-1.0, maxval=1.0)

    base = 0.01 * jnp.ones((H, W), dtype=jnp.float32)

    if nucsite is None:
        nucsite = jnp.zeros((H, W), dtype=jnp.float32)
    else:
        nucsite = nucsite.astype(jnp.float32)

    c0 = base + (noise * psi_b) + (psi_b * nucsite)
    return c0

jax.config.update('jax_enable_x64', True)

binary_mask = np.zeros((200, 200))
center = (100, 100)
radius = 75
y, x = np.ogrid[:200, :200]
dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
binary_mask[dist_from_center <= radius] = 1

shape = Shape(
    binary=jnp.array(binary_mask),
    dx=(1.0, 1.0),
    smooth_epsilon=3.0,
    smooth_curvature=0.008,
    smooth_dt=0.01,
    smooth_tf=100.0,
)

Nx, Ny = binary_mask.shape
pixel_length_rat = 50e-9
Lx=Nx*pixel_length_rat
Ly = Ny*pixel_length_rat
# Lx = 0.01 * Nx
# Ly = 0.01 * Ny

domain = Domain(
    (Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless", shape
)

kB = 8.617333262145e-5; #Boltzmann Constant [eV/K]
T=298.15; #Temperature [K]
omega = 0.11484582867125136; #eV
omega_tilde = omega / (kB * T)
f=lambda c: c * jnp.log(c) + (1.0 - c) * jnp.log(1.0 - c) + omega_tilde * c * (1.0 - c) + 0.059
mu=lambda c: jnp.log(c / (1.0 - c)) + omega_tilde * (1.0 - 2.0 * c)
j0 = lambda c: jnp.sqrt(c)*(1.0 - c)

# ----------------------------
# 0) Build mean-constrained evaluator and PDE instance
# ----------------------------
Nb = 100
sh = SpatialHeterogeneity.from_shape(shape=domain.geometry, n_basis=Nb, prior_fn=jnp.exp)
sh.build_mean_constrained(mask_type="binary")
k_eval_mc = sh.make_mean_constrained_evaluator(apply_mask=False)

theta0 = jnp.zeros((sh.mean_constrained_U.shape[1],))

eq = AllenCahn2DSBM_BV_CC(
    domain=domain, kappa=kappa, f=f, mu=mu, j0=j0,
    k0=k0, H=H, Cmax=c_max, Crate=C_rate,
    use_prefactor=True, mean_constrained=True,
    k_eval=None, k_eval_mc=k_eval_mc, theta=theta0,
)

# ----------------------------
# 1) Initial condition (your nucleation map)
# ----------------------------
shape = domain.geometry
psi_edge_inner = shape.get_mask_edge(connectivity=8, side="inner")
nucsite = jnp.where(psi_edge_inner, 0.2, 0.0)
key = random.PRNGKey(0)
u0_edge = gen_c0(shape, key, noise_fac=0.0, nucsite=nucsite)

# ----------------------------
# 2) Warmup solve with YOUR event to bound tf
# ----------------------------
t_start = 0.0
t_final_guess = eq.tf_tilde_est
dt0 =(eq.tf_tilde_est - t_start) / 100.0
def make_soc_event(eqmodel, *, threshold: float = 0.999):
    def cond_fn(t, y, args, **kwargs):
        return eqmodel.get_SOC(y) > threshold
    return dfx.Event(cond_fn)

event = make_soc_event(eq, threshold=0.999)
solver = ROCK2JAX()

warm_term = dfx.ODETerm(lambda t, y, args: eq.rhs(y, t))
warm_sol = dfx.diffeqsolve(
    warm_term, solver,
    t0=t_start, t1=t_final_guess, dt0=dt0, y0=u0_edge,
    stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6)
    , max_steps=50_000, event=event,
)
t_final_bound = warm_sol.ts[-1]

# -------- UC pieces (psi-weighted) --------
def _uc_from_ys(ys, psi):
    """
    ys:  (T, Nx, Ny)
    psi: (Nx, Ny) weighting field (domain.geometry.smooth or .binary)
    """
    w = jnp.asarray(psi)
    A = jnp.sum(w)  # dxdy cancels, so we can omit it

    # mean concentration per time
    cbar_t = jnp.sum(ys * w[None, ...], axis=(1, 2)) / A  # (T,)

    # variance per time
    diffs = ys - cbar_t[:, None, None]
    v_t = jnp.sum((diffs**2) * w[None, ...], axis=(1, 2)) / A  # (T,)

    # UC = 1 - sum_i sqrt(v_i * cbar_i (1-cbar_i)) / sum_i cbar_i (1-cbar_i)
    eps = 1e-12
    cterm = jnp.clip(cbar_t * (1.0 - cbar_t), 0.0, 1.0)  # safety
    numer = jnp.sum(jnp.sqrt(jnp.maximum(v_t, 0.0) * cterm))
    denom = jnp.sum(cterm) + eps
    UC = 1.0 - numer / denom
    return UC


# Fixed time grid for optimization runs (no event)
Nsave = 128
ts_fixed = jnp.linspace(t_start, t_final_bound, Nsave)

# ----------------------------
# 3) Forward solve on fixed horizon (no event), pass k_field via args
# ----------------------------
rhs_with_k = eq.rhs_with_k_fn           # jitted function: (y, t, k_field) -> dy/dt
fixed_ode   = dfx.ODETerm(lambda t, y, k_field: rhs_with_k(y, t, k_field))
controller  = dfx.PIDController(rtol=1e-4, atol=1e-6)

def forward_fixed(y0, ts, k_field):
    dt0 = (ts[-1] - ts[0]) / 100
    return dfx.diffeqsolve(
        fixed_ode, solver,
        t0=ts[0], t1=ts[-1], dt0=dt0, y0=y0,
        args=k_field,
        stepsize_controller=controller,
        saveat=dfx.SaveAt(ts=ts),
        max_steps=50_000, event=None,
    )
forward_fixed_jit = eqx.filter_jit(forward_fixed)

# -------- Optimistix objective: minimize (1 - UC) --------
# args will be (y0, ts_fixed, psi)
def uc_objective(theta, args):
    y0, ts, psi = args
    k_field = k_eval_mc(theta)                 # mean-constrained k(x,y)
    sol = forward_fixed(y0, ts, k_field)       # fixed-horizon solve (no event)
    UC = _uc_from_ys(sol.ys, psi)
    return 1.0 - UC                            # minimize 1-UC  <=> maximize UC



# Choose psi for quadrature: smooth (recommended) or binary
psi = domain.geometry.smooth  # or: domain.geometry.binary

# ---- Run BFGS on theta ----
solver_bfgs = optx.BFGS(
    rtol=1e-4, atol=1e-6,
    verbose=frozenset({"step","accepted","loss","step_size"}),
)
res_uc = optx.minimise(
    uc_objective,
    solver_bfgs,
    theta0,
    args=(u0_edge, ts_fixed, psi),   # arrays-only args (no eq/domain objects)
    max_steps=200,
    throw=False,
)

theta_star_uc = res_uc.value
print("BFGS status:", res_uc.state)

# ---------- Report UC (init vs optimized) ----------
# compute baseline and optimized UC on the SAME fixed horizon
sol_init = forward_fixed(u0_edge, ts_fixed, k_eval_mc(theta0))
sol_star = forward_fixed(u0_edge, ts_fixed, k_eval_mc(theta_star_uc))

UC_init = float(_uc_from_ys(sol_init.ys, psi))
UC_star = float(_uc_from_ys(sol_star.ys, psi))
print(f"UC initial:   {UC_init:.6f}")
print(f"UC optimized: {UC_star:.6f}")