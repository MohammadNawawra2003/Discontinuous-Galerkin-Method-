# dg_lagrange_p1_1d_improved_v2.py
# Implements 1D DG for Advection using P1 Lagrange basis functions.
# Incorporates teacher feedback and handwritten notes:
# - Consolidated initial conditions.
# - Optimized M_inv * R calculation using element-wise local inverse.
# - Added convergence study (L2 error vs h).
# - Using more conservative CFL in convergence study for stability.
# Uses Method of Lines with RK44.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre # For L2 error calculation
from scipy.signal import square          # For initial condition
from matplotlib.animation import FuncAnimation, FFMpegWriter # For animation
from tqdm import tqdm                    # For progress bars
import os                                # For creating directories
import math                              # For log in convergence rate calculation

# --- P1 Lagrange Basis Functions (on reference element xi in [-1, 1]) ---
# phi_1 (node at xi=-1)
def L1(xi):
    """ Lagrange P1 basis function (value 1 at xi=-1, 0 at xi=+1). """
    return 0.5 * (1.0 - xi)

# phi_2 (node at xi=+1)
def L2(xi):
    """ Lagrange P1 basis function (value 0 at xi=-1, 1 at xi=+1). """
    return 0.5 * (1.0 + xi)

# Derivatives w.r.t xi on reference element
dL1_dxi = -0.5
dL2_dxi = 0.5

# --- Build Local Inverse Mass Matrix (for P1 Lagrange) ---
# CORRESPONDS TO M_k^{-1} in handwritten notes
def get_Minv_local_lagrangeP1(dx):
    """
    Returns the local 2x2 inverse mass matrix for P1 Lagrange.
    Local Mass Matrix M_local = integral(Li * Lj * dx) = (dx/6) * [[2, 1], [1, 2]]
    Local Inverse M_inv_local = inv(M_local) = (2/dx) * [[2, -1], [-1, 2]]
    Args:
        dx: Element width
    Returns:
        M_inv_local: 2x2 numpy array (M_k^{-1})
    """
    if dx <= 1e-15: # Avoid division by zero or very small numbers
        raise ValueError(f"Element width dx={dx} is too small or non-positive")
    # Formula: M_inv_local = (2.0 / dx) * np.array([[ 2.0, -1.0], [-1.0,  2.0]])
    M_inv_local = (2.0 / dx) * np.array([[ 2.0, -1.0],
                                         [-1.0,  2.0]], dtype=np.float64)
    return M_inv_local

# --- Calculate DG Spatial Operator RHS R(U) = -SU + FU for P1 Lagrange ---
# CORRESPONDS TO 'b' vector in handwritten notes (RHS before M_inv multiplication)
def spatial_operator_lagrangeP1(u_flat, n, L, c, alpha):
    """
    Computes the right-hand-side vector 'b' = R(U) = -F^{flux} + S^{stiffness}
    for the P1 Lagrange DG semi-discretization M * dU/dt = R(U).

    Args:
        u_flat: Current state vector (flattened nodal values, N_dof = n*2)
        n: Number of elements
        L: Domain length
        c: Advection speed
        alpha: Upwind parameter (1.0 for full upwind if c>0)
    Returns:
        R_flat: Flattened right-hand-side vector R(U) ('b' in notes)
    """
    N_dof = n * 2
    if u_flat.shape != (N_dof,):
         raise ValueError(f"Input u_flat has wrong shape {u_flat.shape}, expected ({N_dof},)")
    u = u_flat.reshape((n, 2)) # Reshape to (n_elements, 2_nodes_per_element)
    R = np.zeros_like(u) # This will hold the element-wise blocks of 'b'

    # Get nodal value from previous element (right node) using periodic BC
    u_prev_elem_right_node = np.roll(u[:, 1], 1) # u[k-1, 1]

    # Stiffness contribution matrix K_tilde = integral( Lj * dLi/dxi dxi )
    K_tilde = np.array([[-0.5, -0.5],
                        [ 0.5,  0.5]], dtype=np.float64)

    # Loop over elements to compute local RHS R_local
    for k in range(n):
        u_k = u[k, :] # Nodal values [u_left, u_right] for element k
        u_km1_right = u_prev_elem_right_node[k] # Right nodal value of element k-1

        # --- Stiffness Term S_local = c * K_tilde @ U_k ---
        term_Stiffness = c * (K_tilde @ u_k)

        # --- Flux Term Contribution -F_local ---
        # Numerical flux u_hat at element boundaries
        if alpha == 1.0 and c >= 0: # Upwind for c>=0
             u_hat_left = u_km1_right
             u_hat_right = u_k[1]
        elif alpha == 1.0 and c < 0: # Upwind for c<0
             u_hat_left = u_k[0]
             u_kp1_left = u[ (k+1)%n , 0] # Left node of element k+1 (periodic)
             u_hat_right = u_kp1_left
        else: # Defaulting to central flux (average) if alpha != 1.0
             u_kp1_left = u[ (k+1)%n, 0]
             u_hat_left = 0.5 * (u_km1_right + u_k[0])
             u_hat_right = 0.5 * (u_k[1] + u_kp1_left)

        flux_val_left = c * u_hat_left
        flux_val_right = c * u_hat_right

        # Assemble flux contributions to RHS vector 'b' (R)
        # R_i = BoundaryFluxTerm_i + VolumeIntegralTerm_i
        # Based on M U' = [c*Li*u_hat]_bound - integral( c * Sum(Uk*Lk) * dLi/dx )
        # BoundaryFluxTerm = [flux_val_left, -flux_val_right]
        # VolumeIntegralTerm = term_Stiffness = c * K_tilde @ u_k
        term_Flux_Boundary = np.array([flux_val_left, -flux_val_right], dtype=np.float64)

        R[k, :] = term_Flux_Boundary + term_Stiffness

    return R.reshape(N_dof) # Return flattened 'b' vector


# --- Time Stepping Function (RK44 for Lagrange - Element-wise M_inv) ---
# IMPLEMENTS u^{n+1} = u^n + dt * (sum of K stages), where K = M^{-1} * R(u_stage)
# The M^{-1} * R part is done element-wise as per notes: update[k] = M_k^{-1} * b[k]
def rk44_lagrange_local_Minv(u_history, spatial_op_func, M_inv_local, dt, m, n, L, c, alpha):
    """
    Solves dU/dt = M_inv * R(U) using classic RK44 for Lagrange DG.
    Applies the local inverse mass matrix M_k^{-1} element-wise.
    """
    print(f"Starting time integration (RK44 Lagrange, Local M_inv)...")
    N_dof = n * 2
    K1_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R1
    K2_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R2
    K3_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R3
    K4_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R4

    for i in tqdm(range(m), desc=f"RK44 Lagrange (n={n})", unit="step"):
        u_current = u_history[i]

        try:
            # Stage 1
            R1_flat = spatial_op_func(u_current, n, L, c, alpha) # Calculate R(u^n) = b1
            for k in range(n): # Apply M_inv_local element-wise: K1[k] = M_k^{-1} * b1[k]
                idx = slice(2 * k, 2 * k + 2)
                K1_flat[idx] = M_inv_local @ R1_flat[idx]

            # Stage 2
            u_stage2 = u_current + K1_flat * dt / 2.
            R2_flat = spatial_op_func(u_stage2, n, L, c, alpha) # Calculate R(u_stage2) = b2
            for k in range(n): # Apply M_inv_local element-wise: K2[k] = M_k^{-1} * b2[k]
                idx = slice(2 * k, 2 * k + 2)
                K2_flat[idx] = M_inv_local @ R2_flat[idx]

            # Stage 3
            u_stage3 = u_current + K2_flat * dt / 2.
            R3_flat = spatial_op_func(u_stage3, n, L, c, alpha) # Calculate R(u_stage3) = b3
            for k in range(n): # Apply M_inv_local element-wise: K3[k] = M_k^{-1} * b3[k]
                idx = slice(2 * k, 2 * k + 2)
                K3_flat[idx] = M_inv_local @ R3_flat[idx]

            # Stage 4
            u_stage4 = u_current + K3_flat * dt
            R4_flat = spatial_op_func(u_stage4, n, L, c, alpha) # Calculate R(u_stage4) = b4
            for k in range(n): # Apply M_inv_local element-wise: K4[k] = M_k^{-1} * b4[k]
                idx = slice(2 * k, 2 * k + 2)
                K4_flat[idx] = M_inv_local @ R4_flat[idx]

            # Check for NaNs/Infs in stages (indicates instability)
            if not np.all(np.isfinite(K1_flat)) or \
               not np.all(np.isfinite(K2_flat)) or \
               not np.all(np.isfinite(K3_flat)) or \
               not np.all(np.isfinite(K4_flat)):
                print(f"\nWarning: Instability detected at time step {i+1} (n={n}). Aborting RK step.")
                # Fill remaining history with NaNs to signify failure
                u_history[i + 1:, :] = np.nan
                return u_history # Stop integration

            # Final Update: u^{i+1} = u^i + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
            u_history[i + 1] = u_current + dt * (K1_flat + 2 * K2_flat + 2 * K3_flat + K4_flat) / 6.

        except Exception as e:
             print(f"\nError during RK step {i+1} (n={n}): {e}")
             u_history[i + 1:, :] = np.nan # Mark as failed
             return u_history

    print("Time integration finished.")
    return u_history


# --- Initial Condition Functions ---
def ic_sine_wave(x, L):
    """ Smooth sine wave initial condition (Good for convergence study). """
    return np.sin(2 * np.pi * x / L)

def ic_square_wave(x, L):
    """ Discontinuous square wave initial condition. """
    return square(2 * np.pi * x / L, duty=0.5) # Standard 50% duty cycle square wave

# --- Compute Initial Nodal Coefficients (Lagrange P1 Interpolation) ---
def compute_coefficients_lagrangeP1(f_initial_func, L, n):
    """ Computes initial DG coefficients (nodal values) for P1 Lagrange by interpolation. """
    N_dof = n * 2
    u_coeffs = np.zeros((n, 2), dtype=np.float64) # Nodal values (n_elements, 2_nodes)
    dx = L / n
    x_nodes = np.linspace(0, L, n + 1) # Global node locations
    for k in range(n): # Loop over elements
        u_coeffs[k, 0] = f_initial_func(x_nodes[k], L)    # Value at left node
        u_coeffs[k, 1] = f_initial_func(x_nodes[k+1], L) # Value at right node
    return u_coeffs.reshape(N_dof) # Return flattened vector

# --- Evaluation function for P1 Lagrange Solution ---
def evaluate_dg_solution_lagrangeP1(x_eval, nodal_vals_flat, L, n):
    """ Evaluates the P1 Lagrange DG solution at arbitrary points x. """
    N_dof = n*2
    if nodal_vals_flat.shape != (N_dof,):
         # Handle case where simulation might have failed and returned NaNs
         if np.all(np.isnan(nodal_vals_flat)):
             print("Warning: Evaluating DG solution with NaN input. Returning NaNs.")
             return np.full_like(x_eval, np.nan, dtype=float)
         raise ValueError(f"Expected flat nodal_vals shape ({N_dof},), got {nodal_vals_flat.shape}")

    # Check if input contains NaNs even if shape is correct
    if not np.all(np.isfinite(nodal_vals_flat)):
        print("Warning: Evaluating DG solution with NaN/Inf values in nodal_vals_flat. Result may be inaccurate.")

    nodal_vals_element_wise = nodal_vals_flat.reshape((n, 2)) # (n_elem, 2 nodes)
    dx = L / n
    u_h_eval = np.zeros_like(x_eval, dtype=float)

    for i, x_val in enumerate(x_eval):
        # Determine element index k and local coordinate xi in [-1, 1]
        if x_val >= L: # Handle edge case x=L
             element_idx = n - 1
             xi_val = 1.0
        elif x_val <= 0: # Handle edge case x=0
             element_idx = 0
             xi_val = -1.0
        else:
            element_idx = int(np.floor(x_val / dx))
            element_idx = min(element_idx, n - 1) # Ensure index is valid (<= n-1)
            x_left = element_idx * dx
            # Check for division by zero if dx is extremely small
            if dx > 1e-15:
                 xi_val = 2.0 * (x_val - x_left) / dx - 1.0
            else:
                 xi_val = 0.0 # Assign middle point if dx is too small
            xi_val = np.clip(xi_val, -1.0, 1.0) # Clip for safety

        # Get nodal values for the element
        u_left_node = nodal_vals_element_wise[element_idx, 0]
        u_right_node = nodal_vals_element_wise[element_idx, 1]

        # Evaluate using basis functions L1, L2: u_h(xi) = U_left*L1(xi) + U_right*L2(xi)
        u_h_eval[i] = u_left_node * L1(xi_val) + u_right_node * L2(xi_val)

    return u_h_eval

# --- L2 Error Calculation ---
def calculate_l2_error_lagrangeP1(u_nodal_final_flat, f_initial_func, L, n, c, T_final):
    """ Calculates the L2 error between the DG solution and the exact solution at T_final. """
    print(f"Calculating L2 error (Lagrange P1, n={n})...")

    # Check if the final solution is valid
    if u_nodal_final_flat is None or not np.all(np.isfinite(u_nodal_final_flat)):
        print(f"Warning: Cannot calculate L2 error for n={n} due to invalid final solution (NaN/Inf). Returning NaN.")
        return np.nan

    num_quad_points = 5 # Use sufficient quadrature points for accuracy
    try:
        xi_quad, w_quad = roots_legendre(num_quad_points)
    except Exception as e:
        print(f"Error getting Legendre roots: {e}")
        return np.nan

    l2_error_sq_sum = 0.0
    dx = L / n
    if dx <= 1e-15:
        print(f"Warning: dx={dx} too small for L2 error calc (n={n}). Returning NaN.")
        return np.nan
    jacobian = dx / 2.0

    nodal_vals_element_wise = u_nodal_final_flat.reshape((n, 2))

    # Exact solution function at T_final
    u_exact_final_func = lambda x: u_exact(x, T_final, L, c, f_initial_func)

    for k in range(n):
        x_left = k * dx
        # Map reference quad points xi_quad in [-1, 1] to physical points x_quad_k in element k
        x_quad_k = x_left + (xi_quad + 1.0) * jacobian

        # Evaluate DG solution u_h at reference quadrature points within element k
        u_left_node = nodal_vals_element_wise[k, 0]
        u_right_node = nodal_vals_element_wise[k, 1]
        u_h_at_quad_ref = u_left_node * L1(xi_quad) + u_right_node * L2(xi_quad)

        # Evaluate exact solution u_ex at physical quadrature points
        u_ex_at_quad_k = u_exact_final_func(x_quad_k)

        # Calculate squared error at quadrature points
        error_sq_at_quad = (u_h_at_quad_ref - u_ex_at_quad_k)**2

        # Add contribution from element k to the total L2 error squared integral
        l2_error_sq_sum += np.sum(w_quad * error_sq_at_quad) * jacobian

    # Final check before sqrt
    if l2_error_sq_sum < 0 or not np.isfinite(l2_error_sq_sum):
        print(f"Warning: Invalid L2 error sum ({l2_error_sq_sum}) before sqrt for n={n}. Returning NaN.")
        return np.nan

    l2_error = np.sqrt(l2_error_sq_sum)
    print(f"L2 Error ||u_h - u_exact|| at T={T_final:.2f} (n={n}) = {l2_error:.6e}")
    return l2_error

# --- Animation Function (Adapted for Lagrange P1) ---
def plot_function_lagrange(u_nodal_history_flat, L, n, dt, m, c, f_initial_func, save=False, tend=0.):
    """ Creates animation of the P1 Lagrange DG solution vs exact solution. """
    p_equiv = 1 # P1
    N_dof = n * (p_equiv + 1)
    n_plot_eval_per_elem = 10
    x_plot_full = np.linspace(0., L, n * n_plot_eval_per_elem + 1)

    # Check if history contains NaNs
    if np.any(np.isnan(u_nodal_history_flat)):
        print("\nWarning: Simulation history contains NaNs. Animation may be incomplete or incorrect.")
        # Find first NaN time step
        first_nan_step = np.where(np.isnan(u_nodal_history_flat))[0]
        if len(first_nan_step) > 0:
            m_plot = first_nan_step[0] # Plot only up to the failure point
            print(f"Plotting animation frames up to step {m_plot}.")
        else:
             m_plot = m # Should not happen if check passed, but fallback
    else:
        m_plot = m # Plot all steps

    # Reconstruct solution u(x,t) up to m_plot
    print("Reconstructing solution for animation...")
    v_plot = np.zeros((m_plot + 1, len(x_plot_full)))
    for time_idx in tqdm(range(m_plot + 1), desc="Reconstructing Frames"):
         nodal_vals_flat_at_time = u_nodal_history_flat[time_idx, :]
         v_plot[time_idx, :] = evaluate_dg_solution_lagrangeP1(x_plot_full, nodal_vals_flat_at_time, L, n)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    ax.grid(True, linestyle=':')

    global ftSz1, ftSz2, ftSz3
    try: ftSz1, ftSz2, ftSz3
    except NameError: ftSz1, ftSz2, ftSz3 = 16, 14, 12
    plt.rcParams["text.usetex"] = False
    plt.rcParams['font.family'] = 'serif'

    time_template = r'$t = \mathtt{{{:.4f}}} \;[s]$'
    time_text = ax.text(0.75, 0.90, '', fontsize=ftSz1, transform=ax.transAxes)

    dg_line, = ax.plot([], [], color='g', lw=1.5, label=f'DG Solution (Lagrange P1, n={n}, RK44)')
    exact_func_t = lambda x, t: u_exact(x, t, L, c, f_initial_func)
    exact, = ax.plot([], [], color='r', alpha=0.7, lw=3, zorder=0, label='Exact')
    initial_exact_y = exact_func_t(x_plot_full, 0)
    ymin = min(initial_exact_y) - 0.3
    ymax = max(initial_exact_y) + 0.3
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, L)
    ax.set_xlabel(r"$x$", fontsize=ftSz2)
    ax.set_ylabel(r"$u(x,t)$", fontsize=ftSz2)
    ax.legend(fontsize=ftSz3)

    def init():
        dg_line.set_data([], [])
        exact.set_data(x_plot_full, initial_exact_y)
        time_text.set_text(time_template.format(0))
        return tuple([dg_line, exact, time_text])

    def animate(t_idx):
        # Only access valid frames
        if t_idx < v_plot.shape[0]:
             current_time = t_idx * dt
             dg_line.set_data(x_plot_full, v_plot[t_idx, :])
             exact.set_ydata(exact_func_t(x_plot_full, current_time))
             time_text.set_text(time_template.format(current_time))
        return tuple([dg_line, exact, time_text])

    fps = 30
    num_frames_to_show = m_plot + 1 # Use number of valid frames
    interval = max(1, int(1000.0 / fps))

    print("Creating animation...")
    anim = FuncAnimation(fig, animate, frames=num_frames_to_show, interval=interval, blit=False,
                         init_func=init, repeat=False)

    if save:
        output_dir = './figures'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        ic_name_str = f_initial_func.__name__.replace('ic_', '')
        anim_filename = os.path.join(output_dir, f"dg_advection_p1lagrange_{ic_name_str}_n{n}_RK44.mp4")
        print(f"Saving animation to {anim_filename}...")
        writerMP4 = FFMpegWriter(fps=fps)
        try:
            anim.save(anim_filename, writer=writerMP4)
            print("Animation saved.")
        except Exception as e:
            print(f"\n--- Animation Save Error ---")
            print(f"Error: {e}")
            print("Ensure FFmpeg is installed and accessible in your system's PATH.")
            print("----------------------------\n")
            plt.show()
    else:
        plt.show()

    return anim


# --- Core Simulation Logic (No Plotting/Animation) ---
def run_simulation_core(L, n, dt, m, c, f_initial_func, a, rktype='RK44'):
    """ Runs the DG simulation and returns the full solution history. """
    p_equiv = 1 # P1 Lagrange
    N_dof = n * (p_equiv + 1) # n * 2 DoFs

    # Calculate local inverse mass matrix (M_k^{-1})
    dx = L / n
    if dx <= 1e-15:
        print(f"Error: dx={dx} too small in run_simulation_core for n={n}.")
        return None # Indicate failure
    M_inv_local = get_Minv_local_lagrangeP1(dx)

    # Initialize solution history array
    u_history = np.zeros((m + 1, N_dof), dtype=np.float64)

    # Set initial condition
    u_history[0] = compute_coefficients_lagrangeP1(f_initial_func, L=L, n=n)

    # Perform time stepping
    if rktype == 'RK44':
        u_history = rk44_lagrange_local_Minv(
            u_history, spatial_operator_lagrangeP1, M_inv_local,
            dt, m, n, L, c, a)
    else:
        print(f"Error: Only {rktype} implemented.")
        return None # Indicate failure

    # Return the full history (may contain NaNs if RK failed)
    return u_history


# --- Main DG Solver Function (Lagrange P1 Wrapper - Calls Plotting/Core) ---
def advection1d_lagrangeP1_solver(L, n, dt, m, c, f_initial_func, a, rktype='RK44',
                                 anim=True, save=False, tend=0., plot_final=True):
    """ Wrapper to run simulation and optionally plot/animate results. """

    # Run the core simulation
    u_history_flat = run_simulation_core(L, n, dt, m, c, f_initial_func, a, rktype)

    # Check if simulation core returned valid history
    if u_history_flat is None or np.any(np.isnan(u_history_flat)):
        print(f"\n--- Simulation failed for n={n} ---")
        return None, np.nan, None # Return failure indication

    u_final_flat = u_history_flat[m] # Final state vector

    # --- Post-processing ---
    animation_object = None
    if anim:
        animation_object = plot_function_lagrange(
            u_history_flat, L=L, n=n, dt=dt, m=m, c=c,
            f_initial_func=f_initial_func, save=save, tend=tend)

    if plot_final and not anim: # Plot final comparison if animation was off
        print("\nPlotting final solution comparison...")
        n_plot_points_per_element = 50
        x_plot = np.linspace(0, L, n * n_plot_points_per_element + 1)
        u_h_final_lagrange = evaluate_dg_solution_lagrangeP1(x_plot, u_final_flat, L, n)
        u_ex_final = u_exact(x_plot, tend, L, c, f_initial_func)

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, u_ex_final, 'r-', linewidth=3, alpha=0.7, label=f'Exact Solution at T={tend:.2f}')
        plt.plot(x_plot, u_h_final_lagrange, 'g-', linewidth=1.5, label=f'DG Solution (Lagrange P1, n={n}, RK44)')
        for k_elem in range(n + 1): plt.axvline(k_elem * L / n, color='gray', linestyle=':', linewidth=0.5)
        plt.xlabel("x", fontsize=ftSz2)
        plt.ylabel("u(x, T)", fontsize=ftSz2)
        plt.title(f"DG Lagrange P1 Solution vs Exact Solution at T={tend:.2f} (Final)", fontsize=ftSz1)
        plt.legend(fontsize=ftSz3)
        plt.grid(True, linestyle=':')
        ymin = min(u_ex_final.min(), u_h_final_lagrange.min()) - 0.2
        ymax = max(u_ex_final.max(), u_h_final_lagrange.max()) + 0.2
        plt.ylim(ymin, ymax)
        plt.show()

    # Calculate L2 error at final time
    l2_error = calculate_l2_error_lagrangeP1(u_final_flat, f_initial_func, L, n, c, tend)

    return u_final_flat, l2_error, animation_object


# --- Convergence Study Function ---
def run_convergence_study(L, c, T_final, f_initial_func, upwind_param, rk_method, n_values, cfl_target=None, dt_fixed=None):
    """
    Performs a convergence study for the DG method.
    """
    print("\n--- Starting Convergence Study ---")
    if not callable(f_initial_func):
        raise TypeError("f_initial_func must be a callable function.")

    if cfl_target is None and dt_fixed is None:
        raise ValueError("Must provide either cfl_target or dt_fixed.")
    if cfl_target is not None and dt_fixed is not None:
        print("Warning: Both cfl_target and dt_fixed provided. Using cfl_target.")
        dt_fixed = None

    l2_errors = []
    h_values = []

    for n_conv in n_values:
        print(f"\nRunning Convergence Study for n = {n_conv}")
        dx_conv = L / n_conv
        if dx_conv <= 1e-15:
             print(f"Warning: dx={dx_conv} too small for n={n_conv}. Skipping.")
             l2_errors.append(np.nan)
             h_values.append(dx_conv if dx_conv > 0 else np.nan)
             continue
        h_values.append(dx_conv)

        # Determine dt and m for this n
        dt_conv = 0.0
        if dt_fixed is not None:
            if dt_fixed <= 0: raise ValueError("dt_fixed must be positive.")
            dt_conv = dt_fixed
        else: # Use CFL
            if cfl_target is None or cfl_target <= 0: raise ValueError("cfl_target must be positive.")
            if abs(c) > 1e-12:
                 dt_conv = cfl_target * dx_conv / abs(c)
            else: # Handle c=0
                 dt_conv = T_final / 100.0 # Arbitrary dt (e.g., 100 steps total)
            if dt_conv <= 1e-15:
                 print(f"Warning: Calculated dt={dt_conv} too small for n={n_conv}. Skipping.")
                 l2_errors.append(np.nan)
                 continue

        m_conv = max(1, int(np.ceil(T_final / dt_conv)))
        dt_adjusted_conv = T_final / m_conv
        actual_cfl = abs(c) * dt_adjusted_conv / dx_conv if abs(dx_conv) > 1e-12 and abs(c) > 1e-12 else 0

        print(f"  dx = {dx_conv:.4e}")
        print(f"  m = {m_conv}, dt = {dt_adjusted_conv:.4e}, Actual CFL = {actual_cfl:.3f}")

        # Run simulation core
        u_history_conv = run_simulation_core(
            L, n_conv, dt_adjusted_conv, m_conv, c,
            f_initial_func, upwind_param, rk_method
        )

        # Check if simulation failed
        if u_history_conv is None or np.any(np.isnan(u_history_conv)):
             print(f"Simulation failed for n={n_conv}, cannot calculate L2 error.")
             l2_errors.append(np.nan)
        else:
            u_final_conv = u_history_conv[m_conv]
            # Calculate L2 error
            l2_err_n = calculate_l2_error_lagrangeP1(u_final_conv, f_initial_func, L, n_conv, c, T_final)
            l2_errors.append(l2_err_n) # Appends NaN if error calc failed

    # --- Plotting Convergence Results ---
    h_values = np.array(h_values)
    l2_errors = np.array(l2_errors)

    # Filter out non-finite errors/h_values for rate calculation and plotting
    valid_mask = np.isfinite(h_values) & np.isfinite(l2_errors) & (l2_errors > 1e-15) # Need positive error for log
    h_valid = h_values[valid_mask]
    l2_errors_valid = l2_errors[valid_mask]

    rates = []
    if len(h_valid) > 1:
        # Estimate convergence rate: Order = log(E1/E2) / log(h1/h2)
        log_errors = np.log(l2_errors_valid)
        log_h = np.log(h_valid)
        # Ensure h is decreasing for rate calculation consistency if needed,
        # although formula works either way if pairs (h,error) are consistent.
        # Let's sort by h descending to be sure
        sort_indices = np.argsort(h_valid)[::-1]
        h_sorted = h_valid[sort_indices]
        log_errors_sorted = log_errors[sort_indices]
        log_h_sorted = log_h[sort_indices]

        rates = (log_errors_sorted[:-1] - log_errors_sorted[1:]) / (log_h_sorted[:-1] - log_h_sorted[1:])

    print("\n--- Convergence Study Results ---")
    print("  n    |    h       |   L2 Error   | Approx. Rate")
    print("-------|------------|--------------|--------------")
    # Map h_valid back to n_values for printing
    n_values_valid = []
    for h_v in h_valid:
        # Find corresponding n (handle potential float inaccuracies)
        idx = np.where(np.abs(h_values - h_v) < 1e-12)[0]
        if len(idx)>0:
            n_values_valid.append(n_values[idx[0]])
        else:
             n_values_valid.append(np.nan) # Should not happen if mask is correct

    # Sort n_values_valid and l2_errors_valid based on h_sorted for printing
    n_print_order = [n_values_valid[i] for i in sort_indices]
    h_print_order = h_sorted
    l2_print_order = np.exp(log_errors_sorted)

    if len(n_print_order) > 0:
        print(f"{n_print_order[0]:>6d} | {h_print_order[0]:.6f} | {l2_print_order[0]:.6e} |     -    ")
        for i in range(len(rates)):
             # Ensure rate corresponds to the *decrease* from row i to row i+1
             print(f"{n_print_order[i+1]:>6d} | {h_print_order[i+1]:.6f} | {l2_print_order[i+1]:.6e} |   {rates[i]:.3f}  ")
    else:
        print("No valid points found for convergence analysis.")
        # Print original data for debugging
        print("Original n:", n_values)
        print("Original h:", h_values)
        print("Original L2:", l2_errors)
    print("---------------------------------")
    if len(rates) > 0:
        print(f"Average Observed Rate (where calculable): {np.mean(rates):.3f}")
    print("(Expected rate for P1 elements is ~2.0 for smooth solutions)")

    # Plotting uses the original valid (but potentially unsorted) h and error
    plt.figure(figsize=(8, 6))
    plt.loglog(h_valid, l2_errors_valid, 'bo-', markerfacecolor='none', label='L2 Error')

    if len(h_valid) > 0:
        p_expected = 1
        order_expected = p_expected + 1
        C_ref = l2_errors_valid[0] / (h_valid[0]**order_expected) # Scale ref line
        h_plot_ref = np.sort(h_valid) # Ensure reference line goes left-to-right
        plt.loglog(h_plot_ref, C_ref * h_plot_ref**order_expected,
                   'r--', label=f'$\\mathcal{{O}}(h^{order_expected})$ Reference')

    plt.xlabel("Element Size $h = L/n$", fontsize=ftSz2)
    plt.ylabel("$L_2$ Error at $T_{final}$", fontsize=ftSz2)
    plt.title(f"DG P1 Lagrange Convergence (IC: {f_initial_func.__name__})", fontsize=ftSz1)
    plt.gca().invert_xaxis() # Plot h decreasing L->R
    plt.grid(True, which='both', linestyle=':')
    plt.legend(fontsize=ftSz3)
    plt.show()


# --- Matplotlib Global Settings ---
ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'

# --- Exact Solution Function ---
def u_exact(x, t, L, c, initial_func):
    """ Calculates the exact solution u(x,t) = u0(x-ct) with periodic wrapping. """
    x = np.asarray(x)
    x_origin = np.mod(x - c * t, L)
    return initial_func(x_origin, L)

# ==============================================================
# --- Main Execution Block ---
# ==============================================================
if __name__ == "__main__":

    # --- Select Mode ---
    run_normal_simulation = True
    run_conv_study = True

    # --- Configuration (Shared) ---
    L_ = 1.0
    c_ = 1.0
    T_final = 1.0
    rk_method = 'RK44'
    upwind_param = 1.0

    # --- Configuration for Normal Simulation ---
    if run_normal_simulation:
        n_sim = 40
        initial_condition_name = 'sine' # 'sine' or 'square'
        cfl_target_sim = 0.2 # Can be less conservative than convergence study

        dx_sim = L_ / n_sim
        dt_cfl_sim = 0.0
        if abs(c_) > 1e-12:
            dt_cfl_sim = cfl_target_sim * dx_sim / abs(c_)
        else:
            dt_cfl_sim = T_final / 100.0
        if dt_cfl_sim <= 1e-15: raise ValueError(f"Calculated dt={dt_cfl_sim} too small.")

        m_sim = max(1, int(np.ceil(T_final / dt_cfl_sim)))
        dt_adjusted_sim = T_final / m_sim
        # FIX: Use c_ instead of c in the check
        actual_cfl_sim = abs(c_) * dt_adjusted_sim / dx_sim if abs(dx_sim) > 1e-12 and abs(c_) > 1e-12 else 0

        print(f"\n--- Running Normal Simulation ---")
        print(f"  n = {n_sim}, T_final = {T_final:.3f}, c = {c_}")
        print(f"  IC: {initial_condition_name}, Upwind alpha: {upwind_param}")
        print(f"  Target CFL = {cfl_target_sim}, Actual CFL = {actual_cfl_sim:.3f}")
        print(f"  m = {m_sim}, dt = {dt_adjusted_sim:.6f}")

        if initial_condition_name == 'sine': f_initial_sim = ic_sine_wave
        elif initial_condition_name == 'square': f_initial_sim = ic_square_wave
        else: raise ValueError(f"Unknown IC name: {initial_condition_name}")

        u_final_sim, l2_err_sim, anim_obj = advection1d_lagrangeP1_solver(
            L_, n_sim, dt_adjusted_sim, m_sim, c_,
            f_initial_func=f_initial_sim, a=upwind_param, rktype=rk_method,
            anim=True, save=False, tend=T_final, plot_final=True
        )
        if u_final_sim is not None:
            print(f"\nSimulation Complete for n={n_sim}. Final L2 Error = {l2_err_sim:.6e}")
        else:
            print(f"\nSimulation FAILED for n={n_sim}.")


    # --- Configuration for Convergence Study ---
    if run_conv_study:
        f_initial_conv = ic_sine_wave # MUST use smooth IC
        n_values_conv = [5, 10, 20, 40, 80] # As per notes
        # Use a MORE CONSERVATIVE CFL for study to minimize stability/temporal error issues
        cfl_target_conv = 0.05 # Reduced from 0.1

        run_convergence_study(
            L_, c_, T_final, f_initial_conv, upwind_param, rk_method,
            n_values=n_values_conv,
            cfl_target=cfl_target_conv, # Use CFL-based dt
            dt_fixed=None
        )

# ==============================================================
# End of Script
# ==============================================================
