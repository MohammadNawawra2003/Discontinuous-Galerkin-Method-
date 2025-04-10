# dg_lagrange_p1_1d.py
# Implements 1D DG for Advection using P1 Lagrange basis functions,
# following specific feedback and notes provided. Uses Method of Lines with RK44.
# Includes fix for NameError (adds 'table') and uses a fixed small dt.
# Added plot for basis functions. Animation call is included.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre # Keep for L2 error calculation
from scipy.signal import square          # For initial condition
from matplotlib.animation import FuncAnimation, FFMpegWriter # For animation
from tqdm import tqdm                    # For progress bars
import os                                # For creating directories

# --- Provided table (likely CFL limits or related coefficients for Legendre) ---
table = [
    [1.0000, 1.0000, 1.2564, 1.3926, 1.6085], # p=0
    [0, 0.3333, 0.4096, 0.4642, 0.5348],    # p=1 <= Row used for estimate
    [0, 0, 0.2098, 0.2352, 0.2716],       # p=2
    [0, 0, 0.1301, 0.1454, 0.1679],       # p=3
    # Rest of table included as provided in user's code
    [0, 0, 0.0897, 0.1000, 0.1155],       # p=4
    [0, 0, 0.0661, 0.0736, 0.0851],       # p=5
    [0, 0, 0.0510, 0.0568, 0.0656],       # p=6
    [0, 0, 0.0407, 0.0453, 0.0523],       # p=7
    [0, 0, 0.0334, 0.0371, 0.0428],       # p=8
    [0, 0, 0.0279, 0.0310, 0.0358],       # p=9
    [0, 0, 0.0237, 0.0264, 0.0304]        # p=10
]


# --- P1 Lagrange Basis Functions (on reference element xi in [-1, 1]) ---
def L1(xi):
    """ Lagrange P1 basis function shape function phi_1 (value 1 at xi=-1, 0 at xi=+1). """
    return 0.5 * (1.0 - xi)

def L2(xi):
    """ Lagrange P1 basis function shape function phi_2 (value 0 at xi=-1, 1 at xi=+1). """
    return 0.5 * (1.0 + xi)

# Derivatives w.r.t xi on reference element
dL1_dxi = -0.5 # Constant derivative for P1 basis 1
dL2_dxi = 0.5  # Constant derivative for P1 basis 2


# --- Calculate DG Spatial Operator RHS R(U) = -SU + FU for P1 Lagrange ---
def spatial_operator_lagrangeP1(u_flat, n, L, c, alpha):
    """
    Computes the right-hand-side R(U) = S^{stiffness} - F^{flux} for P1 Lagrange DG.
    Matches the derivation M U' = S U - F^{flux} from whiteboard notes, so R = SU - F^{flux}.
    """
    N_dof = n * 2
    u = u_flat.reshape((n, 2)) 
    R = np.zeros_like(u)

    u_prev_elem_right_node = np.roll(u[:, 1], 1)

    K_tilde = np.array([[-0.5, -0.5], 
                      [ 0.5,  0.5]], dtype=np.float64)

    for k in range(n):
        u_k = u[k, :]
        u_km1_right = u_prev_elem_right_node[k]

        term_Stiffness = c * (K_tilde @ u_k)

        if alpha == 1.0 and c > 0:
             u_hat_left = u_km1_right    
             u_hat_right = u_k[1]        
        else:
             u_hat_left = u_km1_right
             u_hat_right = u_k[1]

        flux_val_left = c * u_hat_left
        flux_val_right = c * u_hat_right

        term_Flux = np.array([flux_val_left, -flux_val_right], dtype=np.float64)

        R[k, :] = term_Flux + term_Stiffness

    return R.reshape(N_dof)


# --- Time Stepping Function (RK44 for Lagrange) ---
def rk44_lagrange(u_history, spatial_op_func, M_inv_global, dt, m, n, L, c, alpha):
    """ Solves dU/dt = M_inv * R(U) using classic RK44 for Lagrange DG. """
    for i in tqdm(range(m), desc=f"RK44 Lagrange", unit="step"):
        R1 = spatial_op_func(u_history[i], n, L, c, alpha)
        K1 = M_inv_global @ R1

        R2 = spatial_op_func(u_history[i] + K1 * dt / 2., n, L, c, alpha)
        K2 = M_inv_global @ R2

        R3 = spatial_op_func(u_history[i] + K2 * dt / 2., n, L, c, alpha)
        K3 = M_inv_global @ R3

        R4 = spatial_op_func(u_history[i] + K3 * dt, n, L, c, alpha)
        K4 = M_inv_global @ R4

        u_history[i + 1] = u_history[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
    return u_history


# --- Initial Condition (Sine Wave) ---
def compute_coefficients_lagrangeP1(f, L, n):
    """ Computes initial DG coefficients (nodal values) for P1 Lagrange by interpolation. """
    N_dof = n * 2
    u_coeffs = np.zeros((n, 2), dtype=np.float64) 
    dx = L / n
    for k in range(n):
        x_left = k * dx
        x_right = (k + 1) * dx
        u_coeffs[k, 0] = f(x_left, L)  
        u_coeffs[k, 1] = f(x_right, L) 
    return u_coeffs.reshape(N_dof)


# --- Build Global Inverse Mass Matrix (Block Diagonal for P1 Lagrange) ---
def build_Minv_global_lagrangeP1(n, L):
    """
    Builds the global inverse mass matrix M_inv for P1 Lagrange.
    M is block diagonal, so M_inv is also block diagonal.
    Local Mass Matrix M_local = integral(Li * Lj * dx)
                              = (dx/6) * [[2, 1], [1, 2]]
    Local Inverse M_inv_local = (2/dx) * [[2, -1], [-1, 2]]
    """
    dx = L / n
    M_inv_local = (2.0 / dx) * np.array([[ 2.0, -1.0],
                                        [-1.0,  2.0]], dtype=np.float64)

    M_inv_global = np.zeros((n * 2, n * 2), dtype=np.float64)
    for k in range(n):
        idx_start = 2 * k
        idx_end = 2 * k + 2
        M_inv_global[idx_start:idx_end, idx_start:idx_end] = M_inv_local
    return M_inv_global


# --- Evaluation function for P1 Lagrange Solution ---
def evaluate_dg_solution_lagrangeP1(x_eval, nodal_vals_element_wise, L, n):
    """ Evaluates the P1 Lagrange DG solution at arbitrary points x. """
    dx = L / n
    u_h_eval = np.zeros_like(x_eval, dtype=float)
    for i, x_val in enumerate(x_eval):
        if x_val >= L: element_idx, xi_val = n - 1, 1.0
        elif x_val <= 0: element_idx, xi_val = 0, -1.0
        else:
            element_idx = int(np.floor(x_val / dx))
            element_idx = min(element_idx, n - 1)
            x_left = element_idx * dx
            xi_val = 2.0 * (x_val - x_left) / dx - 1.0
            xi_val = np.clip(xi_val, -1.0, 1.0)
        u_left_node = nodal_vals_element_wise[0, element_idx]
        u_right_node = nodal_vals_element_wise[1, element_idx]
        u_h_eval[i] = u_left_node * L1(xi_val) + u_right_node * L2(xi_val)
    return u_h_eval


# --- Animation Function (Adapted for Lagrange P1) ---
def plot_function_lagrange(u_nodal_history, L, n, dt, m, c, f, save=False, tend=0.):
    """ Creates animation of the P1 Lagrange DG solution vs exact solution. """
    p_equiv = 1
    n_plot_eval = 100 # INCREASED points per element for smooth plotting
    dx = L / n
    x_plot_full = np.linspace(0., L, n * n_plot_eval + 1)

    print("Reconstructing solution for animation...")
    v_plot = np.zeros((m + 1, len(x_plot_full)))
    for time_idx in tqdm(range(m + 1), desc="Reconstructing Frames"):
         nodal_vals_at_time = u_nodal_history[:, :, time_idx]
         v_plot[time_idx, :] = evaluate_dg_solution_lagrangeP1(x_plot_full, nodal_vals_at_time, L, n)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    ax.grid(True, linestyle=':')

    global ftSz1, ftSz2, ftSz3 # Use global font sizes
    try: ftSz1
    except NameError: ftSz1, ftSz2, ftSz3 = 16, 14, 12 # Defaults

    time_template = r'$t = \mathtt{{{:.4f}}} \;[s]$'
    time_text = ax.text(0.75, 0.90, '', fontsize=ftSz1, transform=ax.transAxes)

    dg_line, = ax.plot([], [], color='g', lw=1.5, label=f'DG Solution (Lagrange P1, n={n}, RK44)')
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
        anim_filename = f"./figures/dg_advection_p1lagrange_n{n}_RK44.mp4"
        if not os.path.exists('./figures'): os.makedirs('./figures')
        print(f"Saving animation to {anim_filename}...")
        writerMP4 = FFMpegWriter(fps=fps)
        try:
            anim.save(anim_filename, writer=writerMP4)
            print("Animation saved.")
        except Exception as e: print(f"Error saving animation: {e}")
    else:
        plt.show() # Show the animation window
    return anim


# --- Main DG Solver Function (Lagrange P1 Wrapper - Calls Plotting) ---
# --- Main DG Solver Function (Lagrange P1 Wrapper - Calls Plotting) ---
def advection1d_lagrangeP1_with_anim(L, n, dt, m, c, f, a, rktype='RK44', anim=True, save=False, tend=0.):
    """ Wrapper to call Lagrange solver and then the animation function. """
    p_equiv = 1 # P1 Lagrange
    N_dof = n * (p_equiv + 1) # n * 2 DoFs

    M_inv_global = build_Minv_global_lagrangeP1(n, L)

    u_history = np.zeros((m + 1, N_dof), dtype=np.float64)

    u_history[0] = compute_coefficients_lagrangeP1(f, L=L, n=n)

    if rktype == 'RK44':
        u_history = rk44_lagrange(u_history, spatial_operator_lagrangeP1, M_inv_global,
                                 dt, m, n, L, c, a)
    else:
        print(f"Error: Only RK44 implemented for Lagrange version currently.")
        raise NotImplementedError

    u_final_reshaped = u_history.T.reshape((n, p_equiv + 1, m + 1))
    u_final_reshaped = np.swapaxes(u_final_reshaped, 0, 1)

    animation_object = None
    if anim:
        animation_object = plot_function_lagrange(u_final_reshaped, L=L, n=n, dt=dt, m=m, c=c, f=f, save=save, tend=tend)

    return u_final_reshaped, animation_object


# --- Matplotlib Global Settings ---
ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'

# --- Exact Solution Function --- 
def u_exact(x, t, L, c, initial_func):
    """ Calculates the exact solution u(x,t) = u0(x-ct) with periodic wrapping. """
    x_origin = np.mod(x - c * t, L)
    return initial_func(x_origin, L)

# --- Sine Wave Initial Condition Function ---
def f_initial(x, L):
    """ Sine wave function for the initial condition. """
    k = 2 * np.pi  # Wave number
    return np.sin(k * x)

# ==============================================================
# --- Main Execution Block (Lagrange P1 Version) ---
# ==============================================================
if __name__ == "__main__":

    # <<< --- START: Plot the Basis Functions --- >>>
    print("Plotting P1 Lagrange Basis Functions...")
    xi_plot_basis = np.linspace(-1, 1, 100) # Use a different variable name
    L1_vals = L1(xi_plot_basis)
    L2_vals = L2(xi_plot_basis)

    plt.figure(figsize=(8, 5))
    plt.plot(xi_plot_basis, L1_vals, 'b-', lw=2, label='$L_1(\\xi) = \\phi_{k,1} = (1-\\xi)/2$') # Thicker lines
    plt.plot(xi_plot_basis, L2_vals, 'g-', lw=2, label='$L_2(\\xi) = \\phi_{k,2} = (1+\\xi)/2$') # Thicker lines
    # Mark node points and their values clearly
    plt.plot([-1], [1], 'bo', markersize=8, label='Node 1 ($L_1=1, L_2=0$)')
    plt.plot([1], [0], 'bo', markersize=8)
    plt.plot([1], [1], 'go', markersize=8, label='Node 2 ($L_1=0, L_2=1$)')
    plt.plot([-1], [0], 'go', markersize=8)

    plt.title("P1 Lagrange Shape Functions on Reference Element [-1, 1]")
    plt.xlabel("Reference Coordinate $\\xi$")
    plt.ylabel("Shape Function Value")
    plt.xticks([-1, 0, 1]) # Ensure nodes are marked on x-axis
    plt.yticks([0, 0.5, 1]) # Mark key y-values
    plt.grid(True, linestyle=':')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(fontsize=12) # Adjust font size if needed
    plt.show() # Display this plot before starting the simulation
    # <<< --- END: Plot the Basis Functions --- >>>


    # --- Configuration (Matches Paper Setup for Comparison) ---
    L_ = 1.0         # Domain Length [0, L]
    n_ = 20          # Number of spatial elements
    p_equiv_ = 1     # Using P1 Lagrange basis elements
    c_ = 1.0         # Advection speed (a=1 in paper Sec 8.1)

    # --- Time Stepping ---
    T_final = L_ / abs(c_) # Final time = 1.0
    # --- Use Fixed Small dt for Stability ---
    dt_fixed = 0.005 # START WITH THIS. Adjust if needed.
    m_ = max(1, int(np.ceil(T_final / dt_fixed)))
    dt_adjusted = T_final / m_

    print(f"\n--- Lagrange P1 Simulation Configuration ---") # Add newline
    print(f"  Domain L = {L_}, Elements n = {n_}, Polynomial Degree p = {p_equiv_}")
    print(f"  Advection Speed c = {c_}")
    print(f"Time Discretization:")
    print(f"  Target T_final = {T_final:.3f}")
    print(f"  Number of steps m = {m_}")
    print(f"  USING FIXED dt (adjusted) = {dt_adjusted:.6f}")

    # --- Initial Condition (Square Wave) ---
    #f_initial = lambda x, L: square(2 * np.pi * x / L, 1./3.)

    # --- DG Parameters ---
    rk_method = 'RK44'       # Time integration method
    upwind_param = 1.0       # Use 1.0 for full upwind with c_>0

    # --- Run Simulation & Generate Animation ---
    # Set anim=True to show animation, save=False to display
    u_nodal_history, anim_obj = advection1d_lagrangeP1_with_anim(
        L_, n_, dt_adjusted, m_, c_,
        f=f_initial, a=upwind_param,
        rktype=rk_method,
        anim=True,
        save=True, # CHANGE TO TRUE TO SAVE .mp4
        tend=T_final
    )

    # --- Post-processing: Final Plot and L2 Error ---
    final_nodal_vals_per_element = u_nodal_history[:, :, m_]

    print("\nPost-processing: Plotting final solution (Lagrange P1)...")
    n_plot_points_per_element = 50
    x_plot = np.linspace(0, L_, n_ * n_plot_points_per_element + 1)
    u_h_final_lagrange = evaluate_dg_solution_lagrangeP1(x_plot, final_nodal_vals_per_element, L_, n_)
    u_ex_final = u_exact(x_plot, T_final, L_, c_, f_initial)

    plt.figure(figsize=(10, 6)) # Create a new figure for the final plot
    plt.plot(x_plot, u_ex_final, 'r-', linewidth=3, alpha=0.7, label=f'Exact Solution at T={T_final:.2f}')
    plt.plot(x_plot, u_h_final_lagrange, 'g-', linewidth=1.5, label=f'DG Solution (Lagrange P1, n={n_}, RK44)')
    for k_elem in range(n_ + 1):
         plt.axvline(k_elem * L_ / n_, color='gray', linestyle=':', linewidth=0.5)
    plt.xlabel("x"); plt.ylabel("u(x, T)")
    plt.title(f"DG Lagrange P1 Solution vs Exact Solution at T={T_final:.2f} (Final)")
    plt.legend(); plt.grid(True, linestyle=':')
    plt.ylim(-1.5, 1.5)
    plt.show() # Show the final static plot

    print("\nPost-processing: Calculating L2 error (Lagrange P1)...")
    num_quad_points = 3
    xi_quad, w_quad = roots_legendre(num_quad_points)
    l2_error_sq_sum = 0.0
    dx = L_ / n_
    jacobian = dx / 2.0

    for k in range(n_):
        x_left = k * dx
        x_quad_k = x_left + (xi_quad + 1) * jacobian
        u_left_node = final_nodal_vals_per_element[0, k]
        u_right_node = final_nodal_vals_per_element[1, k]
        u_h_at_quad_k = np.array([u_left_node * L1(xi) + u_right_node * L2(xi) for xi in xi_quad])
        u_ex_at_quad_k = u_exact(x_quad_k, T_final, L_, c_, f_initial)
        error_sq_at_quad = (u_h_at_quad_k - u_ex_at_quad_k)**2
        l2_error_sq_sum += np.sum(w_quad * error_sq_at_quad) * jacobian

    l2_error_lagrange = np.sqrt(l2_error_sq_sum)
    print(f"L2 Error (Lagrange P1) ||u_h - u_exact|| at T={T_final:.2f} = {l2_error_lagrange:.6e}")

# ==============================================================
# End of Script
# ==============================================================
