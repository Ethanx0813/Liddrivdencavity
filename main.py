"""
Lid-Driven Cavity Flow Solver
==============================
2D incompressible Navier-Stokes equations solved with finite differences.

Governing equations:
  du/dt + u·du/dx + v·du/dy = -dp/dx + (1/Re)·∇²u
  dv/dt + u·dv/dx + v·dv/dy = -dp/dy + (1/Re)·∇²v
  ∇·u = 0  (incompressibility)

Method:
  - Explicit time-stepping (forward Euler) for advection + diffusion
  - Pressure Poisson equation (iterative Jacobi) for incompressibility
  - Uniform Cartesian grid, N×N interior points
  - No-slip BC on all walls; lid (top wall) moves at u = U_lid

Usage:
  python lid_driven_cavity.py
  -> saves results as PNG plots (velocity, vorticity, pressure, streamlines)

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ── Configuration ──────────────────────────────────────────────────────────────

class Config:
    N        = 64        # grid points per side (interior)
    Re       = 400       # Reynolds number (try 100, 400, 1000, 5000)
    U_lid    = 1.0       # lid velocity (top wall, x-direction)
    dt       = 0.001     # time step
    n_steps  = 10000     # total time steps to run
    nit      = 50        # pressure Poisson iterations per step
    save_every = 500     # print progress every N steps
    tol      = 1e-6      # convergence tolerance (steady-state check)
    check_conv = 100     # check convergence every N steps


# ── Solver ─────────────────────────────────────────────────────────────────────

class LidDrivenCavity:
    """
    Solves the 2D lid-driven cavity problem on a unit square [0,1]×[0,1].

    Grid layout (i = row from bottom, j = col from left):
      i=N+1 : top wall (lid), u = U_lid, v = 0
      i=0   : bottom wall,    u = 0,     v = 0
      j=0   : left wall,      u = 0,     v = 0
      j=N+1 : right wall,     u = 0,     v = 0
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        N = cfg.N
        self.dx = 1.0 / N
        self.dy = 1.0 / N

        # velocity & pressure arrays (with ghost cells: size N+2)
        self.u = np.zeros((N + 2, N + 2))
        self.v = np.zeros((N + 2, N + 2))
        self.p = np.zeros((N + 2, N + 2))

        # build coordinate arrays for interior points
        self.x = np.linspace(0, 1, N + 2)
        self.y = np.linspace(0, 1, N + 2)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.step_count = 0
        self.history    = []          # (step, max_u, max_v) for convergence plot

    # ── boundary conditions ──────────────────────────────────────────────────

    def apply_bc(self):
        cfg = self.cfg
        # no-slip: bottom, left, right walls
        self.u[0, :]  = 0.0;  self.v[0, :]  = 0.0   # bottom
        self.u[:, 0]  = 0.0;  self.v[:, 0]  = 0.0   # left
        self.u[:, -1] = 0.0;  self.v[:, -1] = 0.0   # right
        # lid: top wall moves at U_lid
        self.u[-1, :] = cfg.U_lid
        self.v[-1, :] = 0.0
        # pressure: Neumann (zero normal gradient) on all walls
        self.p[0, :]  = self.p[1, :]    # bottom
        self.p[-1, :] = self.p[-2, :]   # top
        self.p[:, 0]  = self.p[:, 1]    # left
        self.p[:, -1] = self.p[:, -2]   # right

    # ── pressure RHS (divergence of intermediate velocity) ───────────────────

    def build_rhs(self):
        dt = self.cfg.dt
        dx, dy = self.dx, self.dy
        u, v = self.u, self.v

        # central differences for du/dx and dv/dy on interior
        dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        dvdy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)

        b = np.zeros_like(self.p)
        b[1:-1, 1:-1] = (dudx + dvdy) / dt
        return b

    # ── pressure Poisson solver (Jacobi iterations) ──────────────────────────

    def pressure_poisson(self, b):
        dx, dy = self.dx, self.dy
        p = self.p.copy()

        for _ in range(self.cfg.nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2
              + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2
              - b[1:-1, 1:-1] * dx**2 * dy**2
            ) / (2 * (dx**2 + dy**2))

            # Neumann BCs on pressure
            p[0, :]  = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0]  = p[:, 1]
            p[:, -1] = p[:, -2]

        self.p = p

    # ── momentum equations (explicit Euler) ──────────────────────────────────

    def momentum_step(self):
        cfg = self.cfg
        dt   = cfg.dt
        dx, dy = self.dx, self.dy
        nu   = 1.0 / cfg.Re

        u, v, p = self.u, self.v, self.p
        un, vn  = u.copy(), v.copy()

        # ── u-momentum ──
        adv_u = (
            un[1:-1, 1:-1] * (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)
          + vn[1:-1, 1:-1] * (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)
        )
        diff_u = nu * (
            (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) / dx**2
          + (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]) / dy**2
        )
        pgrad_x = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)

        u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (-adv_u - pgrad_x + diff_u)

        # ── v-momentum ──
        adv_v = (
            un[1:-1, 1:-1] * (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)
          + vn[1:-1, 1:-1] * (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)
        )
        diff_v = nu * (
            (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dx**2
          + (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dy**2
        )
        pgrad_y = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

        v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * (-adv_v - pgrad_y + diff_v)

    # ── single time step ─────────────────────────────────────────────────────

    def advance(self):
        b = self.build_rhs()
        self.pressure_poisson(b)
        self.momentum_step()
        self.apply_bc()
        self.step_count += 1

    # ── run until convergence or max steps ───────────────────────────────────

    def run(self):
        cfg = self.cfg
        print(f"Lid-Driven Cavity Solver")
        print(f"  Re={cfg.Re}, N={cfg.N}, dt={cfg.dt}, max_steps={cfg.n_steps}")
        print(f"  Grid: {cfg.N}×{cfg.N} interior points")
        print("-" * 50)

        u_prev = self.u.copy()
        v_prev = self.v.copy()

        for step in range(1, cfg.n_steps + 1):
            self.advance()

            if step % cfg.save_every == 0:
                max_u = np.abs(self.u).max()
                max_v = np.abs(self.v).max()
                print(f"  step {step:6d}  |u|_max={max_u:.4f}  |v|_max={max_v:.4f}")
                self.history.append((step, max_u, max_v))

            if step % cfg.check_conv == 0:
                du = np.linalg.norm(self.u - u_prev)
                dv = np.linalg.norm(self.v - v_prev)
                if du < cfg.tol and dv < cfg.tol:
                    print(f"\n  Converged at step {step}  (|Δu|={du:.2e}, |Δv|={dv:.2e})")
                    break
                u_prev = self.u.copy()
                v_prev = self.v.copy()

        print("-" * 50)
        print(f"  Done. Total steps: {self.step_count}")

    # ── derived fields ────────────────────────────────────────────────────────

    def vorticity(self):
        """ω = dv/dx - du/dy"""
        dx, dy = self.dx, self.dy
        dvdx = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * dx)
        dudy = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * dy)
        om = np.zeros_like(self.u)
        om[1:-1, 1:-1] = dvdx - dudy
        return om

    def stream_function(self):
        """Compute ψ by integrating v = dψ/dx row by row (simple integration)."""
        dx = self.dx
        psi = np.zeros_like(self.u)
        for i in range(1, self.cfg.N + 1):
            psi[i, 1:] = psi[i, :-1] + self.v[i, :-1] * dx
        return psi

    def velocity_magnitude(self):
        return np.sqrt(self.u**2 + self.v**2)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(solver: LidDrivenCavity, filename: str = "cavity_results.png"):
    cfg = solver.cfg
    x, y = solver.x, solver.y
    X, Y = solver.X, solver.Y

    u = solver.u
    v = solver.v
    p = solver.p
    mag   = solver.velocity_magnitude()
    vort  = solver.vorticity()
    psi   = solver.stream_function()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Lid-Driven Cavity Flow  (Re = {cfg.Re}, N = {cfg.N}, step = {solver.step_count})",
        fontsize=14, fontweight='bold', y=0.98
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # ── 1. Velocity magnitude ──
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(X, Y, mag, levels=50, cmap='plasma')
    ax1.set_title("Velocity magnitude  |u|")
    fig.colorbar(c1, ax=ax1, fraction=0.046)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_aspect('equal')

    # ── 2. Streamlines ──
    ax2 = fig.add_subplot(gs[0, 1])
    speed = mag + 1e-10
    ax2.streamplot(
        y[1:-1], x[1:-1],
        u[1:-1, 1:-1], v[1:-1, 1:-1],
        color=mag[1:-1, 1:-1], cmap='plasma',
        linewidth=0.8, density=1.5,
        arrowsize=0.8
    )
    ax2.set_title("Streamlines")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_aspect('equal')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    # ── 3. Pressure ──
    ax3 = fig.add_subplot(gs[0, 2])
    c3 = ax3.contourf(X, Y, p, levels=50, cmap='RdBu_r')
    ax3.contour(X, Y, p, levels=15, colors='k', linewidths=0.3, alpha=0.4)
    ax3.set_title("Pressure  p")
    fig.colorbar(c3, ax=ax3, fraction=0.046)
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.set_aspect('equal')

    # ── 4. Vorticity ──
    ax4 = fig.add_subplot(gs[1, 0])
    vlim = np.percentile(np.abs(vort), 98)
    c4 = ax4.contourf(X, Y, vort, levels=50,
                      cmap='bwr', vmin=-vlim, vmax=vlim)
    ax4.set_title("Vorticity  ω = ∂v/∂x − ∂u/∂y")
    fig.colorbar(c4, ax=ax4, fraction=0.046)
    ax4.set_xlabel("x"); ax4.set_ylabel("y")
    ax4.set_aspect('equal')

    # ── 5. u-velocity along vertical centerline (x=0.5) ──
    ax5 = fig.add_subplot(gs[1, 1])
    mid_j = cfg.N // 2
    ax5.plot(u[:, mid_j], y, 'b-', linewidth=1.8, label='u(x=0.5, y)')
    ax5.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax5.set_title("u-velocity at x = 0.5")
    ax5.set_xlabel("u"); ax5.set_ylabel("y")
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    # ── 6. v-velocity along horizontal centerline (y=0.5) ──
    ax6 = fig.add_subplot(gs[1, 2])
    mid_i = cfg.N // 2
    ax6.plot(x, v[mid_i, :], 'r-', linewidth=1.8, label='v(x, y=0.5)')
    ax6.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax6.set_title("v-velocity at y = 0.5")
    ax6.set_xlabel("x"); ax6.set_ylabel("v")
    ax6.set_xlim(0, 1)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {filename}")


def plot_convergence(solver: LidDrivenCavity, filename: str = "cavity_convergence.png"):
    if not solver.history:
        return
    steps, max_u, max_v = zip(*solver.history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, max_u, 'b-o', markersize=3, label='max |u|')
    ax.plot(steps, max_v, 'r-o', markersize=3, label='max |v|')
    ax.set_xlabel("Step"); ax.set_ylabel("Max velocity component")
    ax.set_title(f"Convergence history  (Re={solver.cfg.Re})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Convergence plot saved → {filename}")


# ── Ghia et al. benchmark data ─────────────────────────────────────────────────
# Ghia, Ghia & Shin (1982), Table 1 — u-velocity at x=0.5 for Re=100,400,1000

GHIA_U_RE100 = {
    # y    : u
    1.0000:  1.00000,
    0.9766:  0.84123,
    0.9688:  0.78871,
    0.9609:  0.73722,
    0.9531:  0.68717,
    0.8516:  0.23151,
    0.7344:  0.00332,
    0.6172: -0.13641,
    0.5000: -0.20581,
    0.4531: -0.21090,
    0.2813: -0.15662,
    0.1719: -0.10150,
    0.1016: -0.06434,
    0.0703: -0.04775,
    0.0625: -0.04192,
    0.0547: -0.03717,
    0.0000:  0.00000,
}

GHIA_U_RE400 = {
    1.0000:  1.00000,
    0.9766:  0.75837,
    0.9688:  0.68439,
    0.9609:  0.61756,
    0.9531:  0.55892,
    0.8516:  0.29093,
    0.7344:  0.16256,
    0.6172:  0.02135,
    0.5000: -0.11477,
    0.4531: -0.17119,
    0.2813: -0.32726,
    0.1719: -0.24299,
    0.1016: -0.14612,
    0.0703: -0.10338,
    0.0625: -0.09266,
    0.0547: -0.08186,
    0.0000:  0.00000,
}


def plot_ghia_comparison(solver: LidDrivenCavity, filename: str = "cavity_ghia.png"):
    """Compare centerline u-velocity against Ghia et al. (1982) benchmark."""
    cfg = solver.cfg
    if cfg.Re not in (100, 400):
        print(f"  (Ghia comparison only available for Re=100 or Re=400, skipping)")
        return

    ghia = GHIA_U_RE100 if cfg.Re == 100 else GHIA_U_RE400
    gy = list(ghia.keys())
    gu = list(ghia.values())

    mid_j = cfg.N // 2
    sim_u = solver.u[:, mid_j]
    sim_y = solver.y

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(sim_u, sim_y, 'b-', linewidth=2, label='Present solver')
    ax.plot(gu, gy, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel("u-velocity"); ax.set_ylabel("y")
    ax.set_title(f"Centerline u-velocity at x=0.5  (Re={cfg.Re})")
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Ghia comparison saved → {filename}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Configure here ──────────────────────────────────────────────
    cfg = Config()
    cfg.Re       = 400      # Try 100, 400, 1000 (>1000 needs smaller dt)
    cfg.N        = 64       # Grid resolution (64 is fast; 128 is more accurate)
    cfg.dt       = 0.001    # Stable if dt < dx/(U_lid) and dt < dx²*Re/4
    cfg.n_steps  = 15000
    cfg.nit      = 50
    cfg.tol      = 1e-7
    # ────────────────────────────────────────────────────────────────

    # Stability hint
    dx = 1.0 / cfg.N
    cfl   = cfg.U_lid * cfg.dt / dx
    diff  = cfg.dt / (dx**2 * cfg.Re)
    print(f"Stability estimates:  CFL={cfl:.3f}  diffusion number={diff:.3f}")
    if cfl > 1.0:
        print("  WARNING: CFL > 1, simulation may be unstable. Reduce dt.")
    if diff > 0.25:
        print("  WARNING: diffusion number > 0.25, reduce dt or increase Re.")

    solver = LidDrivenCavity(cfg)
    solver.run()

    plot_results(solver,     "cavity_results.png")
    plot_convergence(solver, "cavity_convergence.png")
    plot_ghia_comparison(solver, "cavity_ghia.png")

    print("\nAll done.")