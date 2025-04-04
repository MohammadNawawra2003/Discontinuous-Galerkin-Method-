# dg_legendre_p1_1d.py # Renamed for clarity
# Original Legendre DG code, with added plot for P0 and P1 basis functions.
# Main execution block set to p=1 for comparison with P1 Lagrange.

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre, roots_legendre # Added roots_legendre for L2 error calc later
from scipy.signal import square
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os # Added for directory creation

# --- Provided table (likely CFL limits or related coefficients) ---
table = [
    [1.0000, 1.0000, 1.2564, 1.3926, 1.6085], # p=0
    [0, 0.3333, 0.4096, 0.4642, 0.5348],    # p=1 <= Row used for dt estimate when p=1
    [0, 0, 0.2098, 0.2352, 0.2716],       # p=2
    [0, 0, 0.1301, 0.1454, 0.1679],       # p=3
    [0, 0, 0.0897, 0.1000, 0.1155],       # p=4
    [0, 0, 0.0661, 0.0736, 0.0851],       # p=5
    [0, 0, 0.0510, 0.0568, 0.0656],       # p=6
    [0, 0, 0.0407, 0.0453, 0.0523],       # p=7
    [0, 0, 0.0334, 0.0371, 0.0428],       # p=8
    [0, 0, 0.0279, 0.0310, 0.0358],       # p=9
    [0, 0, 0.0237, 0.0264, 0.0304]        # p=10
]


# --- Time Stepping Functions ---
def fwd_euler(u, Q, dt, m):
    """ Solves du/dt = Q u using Forward Euler. u[0] is initial condition. """
    for i in range(m):
        u[i + 1] = u[i] + dt * Q.dot(u[i])
    return u

def rk22(u, Q, dt, m):
    """ Solves du/dt = Q u using RK22 (Midpoint/Heun). u[0] is initial condition. """
    for i in range(m):
        u_mid = u[i] + dt / 2. * Q.dot(u[i])
        u[i + 1] = u[i] + dt * Q.dot(u_mid)
    return u

def rk44(u, Q, dt, m):
    """ Solves du/dt = Q u using classic RK44. u[0] is initial condition. """
    # Use tqdm for RK44 progress bar within the main call now
    # for i in range(m): # Original loop moved below
    #     K1 = Q.dot(u[i])
    #     K2 = Q.dot(u[i] + K1 * dt / 2.)
    #     K3 = Q.dot(u[i] + K2 * dt / 2.)
    #     K4 = Q.dot(u[i] + K3 * dt)
    #     u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
    # return u
    # Reworked slightly to integrate tqdm better if called from main loop
    if Q is None: # Allow calling just for structure if Q isn't ready
        for i in range(m): u[i+1]=u[i] # Placeholder
        return u
    for i in tqdm(range(m), desc=f"RK44 Steps", unit="step"):
        K1 = Q.dot(u[i])
        K2 = Q.dot(u[i] + K1 * dt / 2.)
        K3 = Q.dot(u[i] + K2 * dt / 2.)
        K4 = Q.dot(u[i] + K3 * dt)
        u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
    return u


# --- DG Matrix Building Function ---
def build_matrix(n, p, a):
    """ Builds the DG stiffness matrix L for periodic BCs (Legendre basis). """
    # D matrix (derivative term)
    diags_of_D = [2 * np.ones(p - 2 * i) for i in range((p + 1) // 2)]
    offsets_of_D = -np.arange(1, p + 1, 2)
    if p == 0: D = sp.bsr_matrix((1, 1))
    else: D = sp.diags([np.zeros(p + 1)] + diags_of_D, np.r_[0, offsets_of_D], format='bsr')

    # A matrix (Alternating +/- 1 from P_i(+-1)P_j(+-1) terms)
    A = np.ones((p + 1, p + 1)); A[1::2, ::2] = -1.; A[::2, 1::2] = -1.
    A = sp.bsr_matrix(A)

    # B matrix (Alternating +/- 1 in columns from P_i(+-1) terms)
    B = np.ones((p + 1, p + 1)); B[:, 1::2] = -1.
    B = sp.bsr_matrix(B)

    # I matrix (Represents P_i(1)P_j(1) or P_i(-1)P_j(-1) contributions)
    I = np.ones((p + 1, p + 1)) # Check original derivation if this should be Identity for certain terms
    I = sp.bsr_matrix(I)

    # Flux terms depend on upwind parameter 'a' (alpha in other codes)
    mat_lft = -(1. + a) / 2. * B.T # Left neighbor contribution (flux term)
    mat_rgt = +(1. - a) / 2. * B  # Right neighbor contribution (flux term)
    # Diagonal block: Volume integral (D.T) + Boundary contributions
    mat_ctr = D.T + (1. + a) / 2. * A - (1. - a) / 2. * I

    # Assemble the global block matrix L
    blocks = []; row_indices = []; col_indices = []; data = []
    block_rows = []
    for i in range(n):
        this_row_blocks = []
        for j in range(n):
            if (j == i - 1) or (i == 0 and j == n - 1): this_row_blocks.append(mat_lft)
            elif j == i: this_row_blocks.append(mat_ctr)
            elif (j == i + 1) or (i == n - 1 and j == 0): this_row_blocks.append(mat_rgt)
            else: this_row_blocks.append(None)
        block_rows.append(this_row_blocks)

    L_stiff = sp.bmat(block_rows, format='bsr', dtype=np.float64)
    return L_stiff


# --- Initial Condition Projection (Legendre) ---
def compute_coefficients(f, L, n, p):
    """ Computes initial DG Legendre coefficients by L2 projection. """
    # Using 5 points should be enough for projection up to p=3 or 4
    num_quad = max(5, p + 2) # Ensure enough points
    s_quad, w_quad = roots_legendre(num_quad) # Get Gauss-Legendre points/weights
    dx = L / n
    psi_basis = [legendre(i) for i in range(p + 1)]

    u_coeffs = np.zeros((n, p + 1))
    for k in range(n):
        x_left = k * dx
        x_quad_k = x_left + (s_quad + 1.0) * dx / 2.0 # Map quad points
        f_vals = f(x_quad_k, L) # Evaluate initial function

        for i in range(p + 1):
            psi_vals = psi_basis[i](s_quad)
            # Formula: u_j = ( (2j+1) / 2 ) * sum_q w_q * f(x(s_q)) * P_j(s_q)
            integral_weighted = np.dot(w_quad, f_vals * psi_vals)
            u_coeffs[k, i] = (2 * i + 1) / 2.0 * integral_weighted

    return u_coeffs.reshape(n * (p + 1)) # Flatten


# --- Main DG Solver Function (Method of Lines - Legendre) ---
def advection1d(L, n, dt, m, p, c, f, a, rktype, anim=False, save=False, tend=0.):
    """ Solves the 1D advection equation using Legendre DG with Method of Lines. """
    # Build M_inv (diagonal)
    inv_mass_matrix_diag_term = np.arange(1, 2 * p + 2, 2)
    inv_mass_matrix_diag = np.tile(inv_mass_matrix_diag_term, n)
    inv_mass_matrix = sp.diags([inv_mass_matrix_diag], [0], format='bsr', dtype=np.float64)

    # Build L (stiffness/flux matrix)
    stiff_matrix = build_matrix(n, p, a)

    # Build Q = -c/dx * M_inv * L
    dx = L / n
    Q = -c * (1.0 / dx) * inv_mass_matrix.dot(stiff_matrix)

    # --- Time Integration ---
    N_dof = n * (p + 1)
    u_history = np.zeros((m + 1, N_dof), dtype=np.float64)
    u_history[0] = compute_coefficients(f, L=L, n=n, p=p) # Initial condition

    print(f"Starting time integration ({rktype})...")
    if rktype == 'ForwardEuler': u_history = fwd_euler(u_history, Q, dt, m)
    elif rktype == 'RK22': u_history = rk22(u_history, Q, dt, m)
    elif rktype == 'RK44': u_history = rk44(u_history, Q, dt, m) # tqdm is inside now
    else: raise ValueError(f"Unknown rktype: {rktype}")
    print("Time integration finished.")

    # Reshape result: (p+1, n_elements, n_timesteps+1)
    u_final_reshaped = u_history.T.reshape((n, p + 1, m + 1))
    u_final_reshaped = np.swapaxes(u_final_reshaped, 0, 1)

    # Optional animation
    animation_object = None
    if anim:
        # Define font sizes for plotting if not globally defined
        global ftSz1, ftSz2, ftSz3
        try: ftSz1
        except NameError:
           print("Setting default font sizes for animation.")
           ftSz1, ftSz2, ftSz3 = 16, 14, 12
           plt.rcParams["text.usetex"] = False
           plt.rcParams['font.family'] = 'serif'
        animation_object = plot_function(u_final_reshaped, L=L, n=n, dt=dt, m=m, p=p, c=c, f=f, save=save, tend=tend)

    return u_final_reshaped, animation_object


# --- Animation Function (Legendre) ---
def plot_function(u_coeffs_history, L, n, dt, m, p, c, f, save=False, tend=0.):
    """ Creates animation of the Legendre DG solution vs exact solution. """
    # u_coeffs_history has shape (p+1, n, m+1)
    n_plot_eval = 100 # Points per element for smooth plotting
    dx = L / n
    x_plot_full = np.linspace(0., L, n * n_plot_eval + 1) # Global x coordinates for plotting

    # Reconstruct solution u(x,t) from coefficients for all times
    print("Reconstructing solution for animation...")
    v_plot = np.zeros((m + 1, len(x_plot_full))) # Store reconstructed solution
    xi_plot_ref = np.linspace(-1, 1, n_plot_eval + 1) # Reference coordinates for plotting within element
    psi_basis_plot = np.array([legendre(i)(xi_plot_ref) for i in range(p + 1)]) # Shape (p+1, n_plot_eval+1)

    for time_idx in tqdm(range(m + 1), desc="Reconstructing Frames"):
        ptr = 0
        for elem_idx in range(n):
            coeffs_at_time_elem = u_coeffs_history[:, elem_idx, time_idx] # Shape (p+1,)
            # Evaluate u_h(xi) = sum_i coeffs[i] * P_i(xi)
            u_h_elem_plot = np.dot(coeffs_at_time_elem, psi_basis_plot) # Shape (n_plot_eval+1,)
            v_plot[time_idx, ptr:ptr + n_plot_eval + 1] = u_h_elem_plot
            ptr += n_plot_eval # Move pointer, overlap boundary points

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    ax.grid(True, linestyle=':')

    # Set font sizes
    global ftSz1, ftSz2, ftSz3
    try: ftSz1
    except NameError: ftSz1, ftSz2, ftSz3 = 16, 14, 12 # Defaults

    time_template = r'$t = \mathtt{{{:.4f}}} \;[s]$'
    time_text = ax.text(0.75, 0.90, '', fontsize=ftSz1, transform=ax.transAxes)

    # Create line for the DG solution
    dg_line, = ax.plot([], [], color='b', lw=1.5, label=f'DG Solution (Legendre P{p}, n={n}, RK44)')
    # Create line for exact solution
    exact_func = lambda x, t: f(np.mod(x - c * t, L), L)
    exact, = ax.plot([], [], color='r', alpha=0.7, lw=3, zorder=0, label='Exact')
    initial_exact_y = exact_func(x_plot_full, 0)
    ymin = min(initial_exact_y) - 0.3; ymax = max(initial_exact_y) + 0.3
    ax.set_ylim(ymin, ymax); ax.set_xlim(0, L)
    ax.set_xlabel(r"$x$", fontsize=ftSz2); ax.set_ylabel(r"$u(x,t)$", fontsize=ftSz2)
    ax.legend(fontsize=ftSz3)

    def init():
        dg_line.set_data([], [])
        exact.set_data(x_plot_full, initial_exact_y)
        time_text.set_text(time_template.format(0))
        return tuple([dg_line, exact, time_text])

    def animate(t_idx):
        current_time = t_idx * dt
        dg_line.set_data(x_plot_full, v_plot[t_idx, :])
        exact.set_ydata(exact_func(x_plot_full, current_time))
        time_text.set_text(time_template.format(current_time))
        return tuple([dg_line, exact, time_text])

    fps = 30; num_frames = m + 1; interval = max(1, 1000 // fps)
    print("Creating animation...")
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=False,
                         init_func=init, repeat=False)

    if save:
        anim_filename = f"./figures/dg_advection_p{p}_n{n}_legendre_RK44.mp4"
        if not os.path.exists('./figures'): os.makedirs('./figures')
        print(f"Saving animation to {anim_filename}...")
        writerMP4 = FFMpegWriter(fps=fps)
        try: anim.save(anim_filename, writer=writerMP4); print("Animation saved.")
        except Exception as e: print(f"Error saving animation: {e}")
    else:
        plt.show()
    return anim


# --- Helper function to evaluate Legendre DG solution ---
def evaluate_dg_solution(x_eval, coeffs_element_wise, L, n, p):
    """ Evaluates the Legendre DG solution given by coefficients on each element. """
    dx = L / n
    u_h_eval = np.zeros_like(x_eval, dtype=float)
    psi_basis = [legendre(i) for i in range(p + 1)]

    for i, x_val in enumerate(x_eval):
        if x_val >= L: element_idx, xi_val = n - 1, 1.0
        elif x_val <= 0: element_idx, xi_val = 0, -1.0
        else:
            element_idx = int(np.floor(x_val / dx))
            element_idx = min(element_idx, n - 1)
            x_left = element_idx * dx
            xi_val = 2.0 * (x_val - x_left) / dx - 1.0
            xi_val = np.clip(xi_val, -1.0, 1.0)

        psi_at_xi = np.array([psi(xi_val) for psi in psi_basis])
        # coeffs_element_wise has shape (p+1, n)
        u_h_eval[i] = np.dot(psi_at_xi, coeffs_element_wise[:, element_idx])
    return u_h_eval

# --- Matplotlib Global Settings ---
ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'

# --- Exact Solution Function ---
def u_exact(x, t, L, c, initial_func):
    """ Calculates the exact solution u(x,t) = u0(x-ct) with periodic wrapping. """
    x_origin = np.mod(x - c * t, L)
    return initial_func(x_origin, L)


# ==============================================================
# --- Main Execution Block (Legendre Version) ---
# ==============================================================
if __name__ == "__main__":

    # <<< --- START: Plot the Basis Functions (P0 & P1) --- >>>
    print("Plotting P0 and P1 Legendre Basis Functions...")
    xi_plot_basis = np.linspace(-1, 1, 100)
    P0_func = legendre(0)
    P1_func = legendre(1)
    P0_vals = P0_func(xi_plot_basis) # Should be all 1s
    P1_vals = P1_func(xi_plot_basis) # Should be line y=x

    plt.figure(figsize=(8, 5))
    plt.plot(xi_plot_basis, P0_vals, 'b-', lw=2, label='$P_0(\\xi) = 1$')
    plt.plot(xi_plot_basis, P1_vals, 'g-', lw=2, label='$P_1(\\xi) = \\xi$')

    plt.title("Legendre Basis Functions (p=1) on Reference Element [-1, 1]")
    plt.xlabel("Reference Coordinate $\\xi$")
    plt.ylabel("Basis Function Value")
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.grid(True, linestyle=':')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.show() # Display this plot before starting the simulation
    # <<< --- END: Plot the Basis Functions --- >>>


    # --- Configuration (SETTING p=1 FOR COMPARISON) ---
    L_ = 1.0         # Domain Length [0, L]
    n_ = 20          # Number of spatial elements
    p_ = 1           # Polynomial degree (SET TO 1 FOR P1 COMPARISON)
    c_ = 1.0         # Advection speed (a=1 in paper Sec 8.1)

    # --- Time Stepping Parameters ---
    # Using the CFL constraint from the provided table and formula
    try:
        # Use column index 3 for RK44 stability estimate? Check table source.
        # Let's assume column index 1 (value 0.3333 for p=1) is more appropriate
        # for basic stability limit, adjust safety factor accordingly.
        CFL_col_idx = 1 # Try column 1 for p=1
        CFL_limit = table[p_][CFL_col_idx]
    except IndexError:
        print(f"Error: p={p_} or column index {CFL_col_idx} is out of bounds for the provided table.")
        exit()

    safety_factor = 0.5     # Safety factor for stability
    dt_ = safety_factor * CFL_limit / abs(c_) * (L_ / n_) # Time step calculation

    # --- Final time T ---
    T_final = L_ / abs(c_) # T = 1.0
    m_ = max(1, int(np.ceil(T_final / dt_))) # Number of time steps needed
    dt_adjusted = T_final / m_

    print(f"\n--- Legendre P{p_} Simulation Configuration ---") # Add newline
    print(f"  Domain L = {L_}, Elements n = {n_}, Polynomial Degree p = {p_}")
    print(f"  Advection Speed c = {c_}")
    print(f"Time Discretization:")
    print(f"  CFL Limit (p={p_}, col_idx={CFL_col_idx}) = {CFL_limit:.4f}")
    print(f"  Safety Factor = {safety_factor}")
    print(f"  Initial dt estimate = {dt_:.6f}")
    print(f"  Target T_final = {T_final:.3f}")
    print(f"  Number of steps m = {m_}")
    print(f"  Adjusted dt = {dt_adjusted:.6f}")

    # --- Initial Condition ---
    f_initial = lambda x, L: square(2 * np.pi * x / L, 1./3.)

    # --- DG Parameters ---
    rk_method = 'RK44'
    upwind_param = 1.0 # Use 1.0 for full upwind with c_>0

    # --- Run Simulation ---
    u_coeffs_time_history, anim_obj_legendre = advection1d(L_, n_, dt_adjusted, m_, p_, c_,
                                       f=f_initial, a=upwind_param,
                                       rktype=rk_method, anim=True,
                                       save=False, tend=T_final)

    # --- Post-processing ---
    final_coeffs_per_element = u_coeffs_time_history[:, :, m_] # Shape (p+1, n)

    print("\nPost-processing: Plotting final solution (Legendre P1)...")
    n_plot_points_per_element = 50
    x_plot = np.linspace(0, L_, n_ * n_plot_points_per_element + 1)
    u_h_final = evaluate_dg_solution(x_plot, final_coeffs_per_element, L_, n_, p_)
    u_ex_final = u_exact(x_plot, T_final, L_, c_, f_initial)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_ex_final, 'r-', linewidth=3, alpha=0.7, label=f'Exact Solution at T={T_final:.2f}')
    plt.plot(x_plot, u_h_final, 'b-', linewidth=1.5, label=f'DG Solution (Legendre P{p_}, n={n_}, RK44)')
    for k_elem in range(n_ + 1):
         plt.axvline(k_elem * L_ / n_, color='gray', linestyle=':', linewidth=0.5)
    plt.xlabel("x"); plt.ylabel("u(x, T)")
    plt.title(f"DG Legendre P{p_} Solution vs Exact Solution at T={T_final:.2f} (Final)")
    plt.legend(); plt.grid(True, linestyle=':')
    plt.ylim(-1.5, 1.5)
    plt.show()

    print("\nPost-processing: Calculating L2 error (Legendre P1)...")
    num_quad_points = p_ + 1 # For p=1, need 2 points (exact for degree 3)
    if num_quad_points < 2: num_quad_points = 2 # Ensure at least 2 points for accuracy
    xi_quad, w_quad = roots_legendre(num_quad_points)

    l2_error_sq_sum = 0.0
    dx = L_ / n_
    jacobian = dx / 2.0

    psi_basis = [legendre(i) for i in range(p_ + 1)]
    psi_at_quad = np.array([[psi(xi) for psi in psi_basis] for xi in xi_quad])

    for k in range(n_):
        x_left = k * dx
        x_quad_k = x_left + (xi_quad + 1) * jacobian
        coeffs_k = final_coeffs_per_element[:, k]
        u_h_at_quad_k = np.dot(psi_at_quad, coeffs_k)
        u_ex_at_quad_k = u_exact(x_quad_k, T_final, L_, c_, f_initial)
        error_sq_at_quad = (u_h_at_quad_k - u_ex_at_quad_k)**2
        l2_error_sq_sum += np.sum(w_quad * error_sq_at_quad) * jacobian

    l2_error = np.sqrt(l2_error_sq_sum)
    print(f"L2 Error (Legendre P{p_}) ||u_h - u_exact|| at T={T_final:.2f} = {l2_error:.6e}")

# ==============================================================
# End of Script
# ==============================================================
