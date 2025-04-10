# dg_lagrange_p1_1d.py
# Implements 1D DG for Advection using P1 Lagrange basis functions,
# following specific feedback and notes provided. Uses Method of Lines with RK44.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre # For L2 error calculation
from scipy.signal import square          # For initial condition
from matplotlib.animation import FuncAnimation, FFMpegWriter # For animation
from tqdm import tqdm                    # For progress bars
import os                                # For creating directories


table = [
    [1.0000, 1.0000, 1.2564, 1.3926, 1.6085], # p=0
    [0, 0.3333, 0.4096, 0.4642, 0.5348],    # p=1 <= Row used for estimate
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
# --- P1 Lagrange Basis Functions (on reference element xi in [-1, 1]) ---
# These correspond to phi_1 and phi_2 in the handwritten notes
def L1(xi):
    """ Lagrange P1 basis function shape function phi_1 (value 1 at xi=-1, 0 at xi=+1). """
    # Matches formula for phi_1(x)|xi_1 in notes (with xi_1=-1, xi_2=1 -> x_1=x_left, x_2=x_right)
    # (x_2 - x) / (x_2 - x_1) maps to (1 - xi) / 2
    return 0.5 * (1.0 - xi)

def L2(xi):
    """ Lagrange P1 basis function shape function phi_2 (value 0 at xi=-1, 1 at xi=+1). """
    # Matches formula for phi_2(x)|xi_2 in notes (with xi_2=-1, xi_3=1 -> x_2=x_left, x_3=x_right)
    # (x - x_2) / (x_3 - x_2) maps to (xi - (-1)) / (1 - (-1)) = (xi + 1) / 2
    return 0.5 * (1.0 + xi)

# Derivatives w.r.t xi on reference element
dL1_dxi = -0.5 # Constant derivative for P1 basis 1
dL2_dxi = 0.5  # Constant derivative for P1 basis 2

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
    # Using dense numpy array as n=20 is small enough.
    # For larger n, scipy.sparse.block_diag would be better.
    return M_inv_global


# --- Calculate DG Spatial Operator RHS R(U) = -SU + FU for P1 Lagrange ---
def spatial_operator_lagrangeP1(u_flat, n, L, c, alpha):
    """
    Computes the right-hand-side R(U) = S^{stiffness} - F^{flux} for P1 Lagrange DG.
    Matches the derivation M U' = S U - F^{flux} from whiteboard notes, so R = SU - F^{flux}.
    Or, following integration by parts: M U' = - (Flux Term) + (Stiffness Term).
    Let's stick to M U' = R = -F^{flux} + S^{stiffness}.

    Args:
        u_flat: Current state vector (flattened nodal values, N_dof = n*2)
        n: Number of elements
        L: Domain length
        c: Advection speed
        alpha: Upwind parameter (1.0 for full upwind if c>0)
    Returns:
        R_flat: Flattened right-hand-side vector R(U)
    """
    N_dof = n * 2
    u = u_flat.reshape((n, 2)) # Reshape to (n_elements, 2_nodes_per_element)
                               # u[k, 0] = left node value, u[k, 1] = right node value
    R = np.zeros_like(u)

    # Get nodal value from previous element (right node) using periodic BC
    # This is u_{k-1}(xi=+1) needed for the flux u_hat_left
    u_prev_elem_right_node = np.roll(u[:, 1], 1)

    # Pre-calculate Stiffness contribution matrix K_tilde = integral( Lj * dLi/dxi dxi )
    # This relates to the S term in the whiteboard notes M U' = S U - F U
    # S_{stiffness, i} = Sum_j [ integral( c * Lj * dLi/dx dx ) ] * Uj
    #                  = Sum_j [ c * integral( Lj * dLi/dxi dxi ) ] * Uj
    K_tilde = np.array([[-0.5, -0.5],  # integral(L1*dL1/dxi), integral(L2*dL1/dxi)
                      [ 0.5,  0.5]], # integral(L1*dL2/dxi), integral(L2*dL2/dxi)
                     dtype=np.float64)

    # Loop over elements to compute local RHS R_local = -F_local + S_local
    for k in range(n):
        u_k = u[k, :] # Nodal values [u_left, u_right] for element k = [U_k1, U_k2]
        u_km1_right = u_prev_elem_right_node[k] # Right nodal value of element k-1 = U_{k-1, 2}

        # --- Calculate Stiffness Term S_local = c * K_tilde @ U_k ---
        # This is the volume integral term after integration by parts
        # S_{stiffness, i} = Sum_j [ c * K_tilde_ij ] * Uj
        term_Stiffness = c * (K_tilde @ u_k)

        # --- Calculate Flux Term Contribution -F_local ---
        # Contribution = [ c*u_hat_left, -c*u_hat_right ] based on derivation M U' = R
        # Where R_i = - ( [c u_hat L_i]_right - [c u_hat L_i]_left ) + Stiffness_i

        # Upwind flux (alpha=1.0, c>0): u_hat = u_left_state
        # This matches the whiteboard diagram where flux info comes from left if c>0
        if alpha == 1.0 and c > 0:
             u_hat_left = u_km1_right    # u_hat at left boundary uses u from element k-1
             u_hat_right = u_k[1]        # u_hat at right boundary uses u from element k
        # Add other flux options if needed, ensure c < 0 case is handled if required
        # elif ...
        else:
             # Defaulting to upwind for c>0 if alpha is not 1.0
             if not (alpha == 1.0 and c > 0):
                 print(f"Warning: Alpha={alpha}, c={c}. Using default upwind flux logic for c>0.")
             u_hat_left = u_km1_right
             u_hat_right = u_k[1]

        flux_val_left = c * u_hat_left
        flux_val_right = c * u_hat_right

        # -F_i = - ( L_i(+1) * flux_val_right - L_i(-1) * flux_val_left )
        # -F_1 = - ( L1(+1)*flux_val_right - L1(-1)*flux_val_left ) = - ( 0*f_r - 1*f_l ) =  flux_val_left
        # -F_2 = - ( L2(+1)*flux_val_right - L2(-1)*flux_val_left ) = - ( 1*f_r - 0*f_l ) = -flux_val_right
        term_Flux = np.array([flux_val_left, -flux_val_right], dtype=np.float64)

        # --- Calculate R_local = -F_local + S_local ---
        R[k, :] = term_Flux + term_Stiffness

    return R.reshape(N_dof) # Return flattened vector

# --- Time Stepping Function (RK44 for Lagrange) ---
def rk44_lagrange(u_history, spatial_op_func, M_inv_global, dt, m, n, L, c, alpha):
    """ Solves dU/dt = M_inv * R(U) using classic RK44 for Lagrange DG. """
    print(f"Starting time integration (RK44 Lagrange)...")
    for i in tqdm(range(m), desc=f"RK44 Lagrange", unit="step"):
        R1 = spatial_op_func(u_history[i], n, L, c, alpha)
        K1 = M_inv_global @ R1 # K1 = dU/dt stage 1

        R2 = spatial_op_func(u_history[i] + K1 * dt / 2., n, L, c, alpha)
        K2 = M_inv_global @ R2 # K2 = dU/dt stage 2

        R3 = spatial_op_func(u_history[i] + K2 * dt / 2., n, L, c, alpha)
        K3 = M_inv_global @ R3 # K3 = dU/dt stage 3

        R4 = spatial_op_func(u_history[i] + K3 * dt, n, L, c, alpha)
        K4 = M_inv_global @ R4 # K4 = dU/dt stage 4

        u_history[i + 1] = u_history[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
    print("Time integration finished.")
    return u_history

# --- Initial Condition (Lagrange P1 Nodal Interpolation) ---
def compute_coefficients_lagrangeP1(f, L, n):
    """ Computes initial DG coefficients (nodal values) for P1 Lagrange by interpolation. """
    N_dof = n * 2
    u_coeffs = np.zeros((n, 2), dtype=np.float64) # Nodal values (n_elements, 2_nodes)
    dx = L / n
    for k in range(n): # Loop over elements
        x_left = k * dx
        x_right = (k + 1) * dx
        u_coeffs[k, 0] = f(x_left, L)  # Value at left node
        u_coeffs[k, 1] = f(x_right, L) # Value at right node
    return u_coeffs.reshape(N_dof) # Return flattened vector

# --- Evaluation function for P1 Lagrange Solution ---
def evaluate_dg_solution_lagrangeP1(x_eval, nodal_vals_element_wise, L, n):
    """ Evaluates the P1 Lagrange DG solution at arbitrary points x. """
    dx = L / n
    u_h_eval = np.zeros_like(x_eval, dtype=float)
    # nodal_vals_element_wise has shape (2, n) -> [0,:] = left node vals, [1,:] = right node vals

    for i, x_val in enumerate(x_eval):
        # Determine element index and local coordinate xi
        if x_val >= L: element_idx, xi_val = n - 1, 1.0
        elif x_val <= 0: element_idx, xi_val = 0, -1.0
        else:
            element_idx = int(np.floor(x_val / dx))
            element_idx = min(element_idx, n - 1) # Ensure index valid
            x_left = element_idx * dx
            xi_val = 2.0 * (x_val - x_left) / dx - 1.0
            xi_val = np.clip(xi_val, -1.0, 1.0) # Clip for safety

        # Get nodal values for the element
        u_left_node = nodal_vals_element_wise[0, element_idx]
        u_right_node = nodal_vals_element_wise[1, element_idx]

        # Evaluate using basis functions L1, L2: u_h(xi) = U1*L1(xi) + U2*L2(xi)
        u_h_eval[i] = u_left_node * L1(xi_val) + u_right_node * L2(xi_val)

    return u_h_eval

# --- Animation Function (Adapted for Lagrange P1) ---
def plot_function_lagrange(u_nodal_history, L, n, dt, m, c, f, save=False, tend=0.):
    """ Creates animation of the P1 Lagrange DG solution vs exact solution. """
    p_equiv = 1 # Explicitly state P1
    # u_nodal_history has shape (2, n, m+1)
    n_plot_eval = 100 # Points per element for smooth plotting
    dx = L / n
    x_plot_full = np.linspace(0., L, n * n_plot_eval + 1) # Global x coordinates

    # Reconstruct solution u(x,t) from nodal values for all times for plotting
    print("Reconstructing solution for animation...")
    v_plot = np.zeros((m + 1, len(x_plot_full))) # Store reconstructed solution at plot points
    for time_idx in tqdm(range(m + 1), desc="Reconstructing Frames"):
         nodal_vals_at_time = u_nodal_history[:, :, time_idx] # Shape (2, n)
         v_plot[time_idx, :] = evaluate_dg_solution_lagrangeP1(x_plot_full, nodal_vals_at_time, L, n)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    ax.grid(True, linestyle=':') # Grid with dots

    # Set font sizes (ensure they are defined)
    global ftSz1, ftSz2, ftSz3
    try:
       ftSz1, ftSz2, ftSz3
    except NameError:
       ftSz1, ftSz2, ftSz3 = 16, 14, 12 # Default sizes
       plt.rcParams["text.usetex"] = False # Default to False
       plt.rcParams['font.family'] = 'serif'

    time_template = r'$t = \mathtt{{{:.4f}}} \;[s]$'
    time_text = ax.text(0.75, 0.90, '', fontsize=ftSz1, transform=ax.transAxes)

    # Create line for the DG solution (single line for P1 interpolation)
    dg_line, = ax.plot([], [], color='g', lw=1.5, label=f'DG Solution (Lagrange P1, n={n}, RK44)')
    # Create line for exact solution
    exact_func = lambda x, t: f(np.mod(x - c * t, L), L) # Periodically wrapped exact solution
    exact, = ax.plot([], [], color='r', alpha=0.7, lw=3, zorder=0, label='Exact')
    # Set plot limits based on initial exact solution +/- padding
    initial_exact_y = exact_func(x_plot_full, 0)
    ymin = min(initial_exact_y) - 0.3 # Adjusted padding
    ymax = max(initial_exact_y) + 0.3
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, L)

    ax.set_xlabel(r"$x$", fontsize=ftSz2)
    ax.set_ylabel(r"$u(x,t)$", fontsize=ftSz2)
    ax.legend(fontsize=ftSz3)

    def init():
        """ Initializes the animation plot. """
        dg_line.set_data([], [])
        exact.set_data(x_plot_full, initial_exact_y)
        time_text.set_text(time_template.format(0))
        return tuple([dg_line, exact, time_text])

    def animate(t_idx):
        """ Updates the plot for frame t_idx. """
        current_time = t_idx * dt
        dg_line.set_data(x_plot_full, v_plot[t_idx, :])
        exact.set_ydata(exact_func(x_plot_full, current_time))
        time_text.set_text(time_template.format(current_time))
        return tuple([dg_line, exact, time_text])

    # Animation setup
    fps = 30
    num_frames = m + 1
    interval = max(1, 1000 // fps)

    print("Creating animation...")
    # Use blit=False for wider compatibility, although blit=True might be faster if it works
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=False,
                         init_func=init, repeat=False)

    if save:
        anim_filename = f"./figures/dg_advection_p1lagrange_n{n}_RK44.mp4"
        # Ensure figure directory exists
        if not os.path.exists('./figures'):
             os.makedirs('./figures')
        print(f"Saving animation to {anim_filename}...")
        writerMP4 = FFMpegWriter(fps=fps)
        try:
            anim.save(anim_filename, writer=writerMP4)
            print("Animation saved.")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Ensure FFmpeg is installed and in your system's PATH.")
            print("Consider setting save=False to display animation instead.")
    else:
        plt.show() # Show the animation window

    return anim # Return animation object if needed


# --- Main DG Solver Function (Lagrange P1 Wrapper - Calls Plotting) ---
def advection1d_lagrangeP1_with_anim(L, n, dt, m, c, f, a, rktype='RK44', anim=True, save=False, tend=0.):
    """ Wrapper to call Lagrange solver and then the animation function. """

    p_equiv = 1 # P1 Lagrange
    N_dof = n * (p_equiv + 1) # n * 2 DoFs

    # Build global inverse mass matrix
    M_inv_global = build_Minv_global_lagrangeP1(n, L)

    # Initialize solution history array
    u_history = np.zeros((m + 1, N_dof), dtype=np.float64)

    # Set initial condition using interpolation
    u_history[0] = compute_coefficients_lagrangeP1(f, L=L, n=n)

    # Perform time stepping
    if rktype == 'RK44':
        u_history = rk44_lagrange(u_history, spatial_operator_lagrangeP1, M_inv_global,
                                 dt, m, n, L, c, a)
    else:
        print(f"Error: Only RK44 implemented for Lagrange version currently.")
        raise NotImplementedError

    # Reshape the result for easier handling: (n_nodes_per_elem, n_elements, n_timesteps+1)
    u_final_reshaped = u_history.T.reshape((n, p_equiv + 1, m + 1))
    u_final_reshaped = np.swapaxes(u_final_reshaped, 0, 1) # Swap axes -> (2, n, m+1)

    # Call animation/plotting if requested
    animation_object = None
    if anim:
        animation_object = plot_function_lagrange(u_final_reshaped, L=L, n=n, dt=dt, m=m, c=c, f=f, save=save, tend=tend)

    return u_final_reshaped, animation_object # Return the reshaped history and animation


# --- Matplotlib Global Settings ---
ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = False # Set to False if LaTeX issues arise
plt.rcParams['font.family'] = 'serif'

# --- Exact Solution Function ---
def u_exact(x, t, L, c, initial_func):
    """ Calculates the exact solution u(x,t) = u0(x-ct) with periodic wrapping. """
    x_origin = np.mod(x - c * t, L) # Modulo L ensures periodicity
    return initial_func(x_origin, L)

# ==============================================================
# --- Main Execution Block (Lagrange P1 Version) ---
# ==============================================================
if __name__ == "__main__":

    # --- Configuration (Matches Paper Setup for Comparison) ---
    L_ = 1.0         # Domain Length [0, L]
    n_ = 20          # Number of spatial elements
    p_equiv_ = 1     # Using P1 Lagrange basis elements
    c_ = 1.0         # Advection speed (a=1 in paper Sec 8.1)

    # --- Time Stepping ---
    T_final = L_ / abs(c_) # Final time = 1.0

    # --- Use Fixed Small dt for Stability ---
    # Based on previous attempts, start with a known stable value if found,
    # otherwise, keep it small.
    dt_fixed = 0.005 # START WITH THIS. Adjust if unstable/too slow.
    m_ = max(1, int(np.ceil(T_final / dt_fixed)))
    dt_adjusted = T_final / m_

    print(f"--- Lagrange P1 Configuration ---")
    print(f"  Domain L = {L_}, Elements n = {n_}, Polynomial Degree p = {p_equiv_}")
    print(f"  Advection Speed c = {c_}")
    print(f"Time Discretization:")
    print(f"  Target T_final = {T_final:.3f}")
    print(f"  Number of steps m = {m_}")
    print(f"  USING FIXED dt (adjusted) = {dt_adjusted:.6f}")

    # --- Initial Condition (Square Wave) ---
    f_initial = lambda x, L: square(2 * np.pi * x / L, 1./3.)

    # --- DG Parameters ---
    rk_method = 'RK44'       # Time integration method
    upwind_param = 1.0       # Use 1.0 for full upwind with c_>0

    # --- Run Simulation & Generate Animation ---
    # Set anim=True to show animation, save=False to display (or True to save)
    u_nodal_history, anim_obj = advection1d_lagrangeP1_with_anim(
        L_, n_, dt_adjusted, m_, c_,
        f=f_initial, a=upwind_param,
        rktype=rk_method,
        anim=True,
        save=False, # CHANGE TO TRUE TO SAVE .mp4 (requires ffmpeg)
        tend=T_final
    )

    # --- Post-processing: Final Plot and L2 Error ---
    # This code runs after the animation window is closed or after saving is complete.
    final_nodal_vals_per_element = u_nodal_history[:, :, m_] # Shape (2, n)

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
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title(f"DG Lagrange P1 Solution vs Exact Solution at T={T_final:.2f} (Final)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.ylim(-1.5, 1.5)
    plt.show() # Show the final static plot

    print("\nPost-processing: Calculating L2 error (Lagrange P1)...")
    num_quad_points = 3 # Suitable for integrating (P1)^2 terms
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
