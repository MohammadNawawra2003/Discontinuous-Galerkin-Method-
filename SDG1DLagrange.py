# dg_lagrange_p1_1d_improved_v2.py
# Implements 1D DG for Advection using P1 Lagrange basis functions.
# - Consolidated initial conditions.
# - Optimized M_inv * R calculation using element-wise local inverse.
# - Added convergence study (L2 error vs h) with specific table format.
# - Reports asymptotic rate from finest grids.
# - Corrected convergence plot order.
# - Added minor robustness checks for plotting/animation.
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
import shutil                            # For checking ffmpeg path

# --- Matplotlib Global Settings ---
# Define font sizes globally first
ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'

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

# --- Calculate DG Spatial Operator RHS R(U) = F^{flux} - S^{stiffness} for P1 Lagrange ---
# CORRESPONDS TO 'b' vector in handwritten notes (RHS before M_inv multiplication)
# NOTE: Re-derived, RHS should be M dU/dt = BoundaryFlux - VolumeIntegral
# Our R = BoundaryFlux + VolumeIntegral(Sign Adjusted) -> M dU/dt = R
def spatial_operator_lagrangeP1(u_flat, n, L, c, alpha):
    """
    Computes the right-hand-side vector 'b' = R(U) such that M * dU/dt = R(U).
    R_i = [c*phi_i*u_hat]_bound - c*Integral( u * d(phi_i)/dx dx)
    For P1 Lagrange, this corresponds to R = term_Flux_Boundary + term_Stiffness.

    Args:
        u_flat: Current state vector (flattened nodal values, N_dof = n*2)
        n: Number of elements
        L: Domain length
        c: Advection speed
        alpha: Numerical flux parameter (1.0 for full upwind)
               Note: The implementation below primarily uses alpha=1.0 logic.
                     Central flux (alpha != 1.0) is implemented but less tested.
    Returns:
        R_flat: Flattened right-hand-side vector R(U) ('b' in notes)
    """
    N_dof = n * 2
    if not isinstance(u_flat, np.ndarray) or u_flat.shape != (N_dof,):
         raise ValueError(f"Input u_flat has wrong shape/type {u_flat.shape}, expected ({N_dof},)")
    u = u_flat.reshape((n, 2)) # Reshape to (n_elements, 2_nodes_per_element)
    R = np.zeros_like(u, dtype=np.float64) # This will hold the element-wise blocks of 'b'

    # Get nodal value from previous element (right node) using periodic BC
    u_prev_elem_right_node = np.roll(u[:, 1], 1) # u[k-1, 1]

    # Stiffness-related contribution matrix K_tilde = integral( Lj * dLi/dxi dxi )
    K_tilde = np.array([[-0.5, -0.5],
                        [ 0.5,  0.5]], dtype=np.float64)

    # Loop over elements to compute local RHS R_local
    for k in range(n):
        u_k = u[k, :] # Nodal values [u_left, u_right] for element k
        u_km1_right = u_prev_elem_right_node[k] # Right nodal value of element k-1

        # --- Stiffness Term S_local = c * Integral( d(phi_i)/dx * u dx ) ---
        term_Stiffness = c * (K_tilde @ u_k)

        # --- Flux Term Contribution F_local = [c * phi_i * u_hat]_boundary ---
        # Numerical flux u_hat at element boundaries
        if alpha == 1.0 and c >= 0: # Upwind for c>=0
             u_hat_left = u_km1_right      # u(x_k^-)
             u_hat_right = u_k[1]          # u(x_{k+1}^-) which is u_{k, right}
        elif alpha == 1.0 and c < 0: # Upwind for c<0
             u_hat_left = u_k[0]           # u(x_k^+) which is u_{k, left}
             u_kp1_left = u[ (k+1)%n , 0]  # Left node of element k+1 (periodic) = u(x_{k+1}^+)
             u_hat_right = u_kp1_left
        else: # Defaulting to central flux (average) if alpha != 1.0
             u_kp1_left = u[ (k+1)%n, 0] # Left node of element k+1 (periodic)
             u_hat_left = 0.5 * (u_km1_right + u_k[0])
             u_hat_right = 0.5 * (u_k[1] + u_kp1_left)
             # Warning removed to avoid noise

        flux_val_left = c * u_hat_left
        flux_val_right = c * u_hat_right

        # Assemble flux contributions to RHS vector 'b' (R)
        term_Flux_Boundary = np.array([flux_val_left, -flux_val_right], dtype=np.float64)

        # Assemble R_k = Boundary Flux Term + Volume Integral Term
        R[k, :] = term_Flux_Boundary + term_Stiffness

    return R.reshape(N_dof) # Return flattened 'b' vector


# --- Time Stepping Function (RK44 for Lagrange - Element-wise M_inv) ---
# IMPLEMENTS u^{n+1} = u^n + dt * (sum of K stages), where K = M^{-1} * R(u_stage)
# The M^{-1} * R part is done element-wise as per notes: update[k] = M_k^{-1} * b[k]
def rk44_lagrange_local_Minv(u_history, spatial_op_func, M_inv_local, dt, m, n, L, c, alpha):
    """
    Solves dU/dt = M_inv * R(U) using classic RK44 for Lagrange DG.
    Applies the local inverse mass matrix M_k^{-1} element-wise.
    Args:
        u_history: Numpy array to store solution history (shape (m+1, N_dof))
        spatial_op_func: Function handle for the spatial operator R(U)
        M_inv_local: The 2x2 local inverse mass matrix
        dt: Time step size
        m: Number of time steps
        n: Number of elements
        L: Domain length
        c: Advection speed
        alpha: Upwind parameter for spatial operator
    Returns:
        u_history: The updated solution history array. May contain NaNs if unstable.
    """
    # Minor optimization: Precompute dt/2 and dt/6
    dt_half = dt / 2.0
    dt_sixth = dt / 6.0

    print(f"Starting time integration (RK44 Lagrange, Local M_inv)...")
    N_dof = n * 2
    K1_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R1
    K2_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R2
    K3_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R3
    K4_flat = np.zeros(N_dof, dtype=np.float64) # Stores M_inv * R4
    # Temporary storage for element-wise RHS and K
    R_local = np.zeros(2, dtype=np.float64)
    K_local = np.zeros(2, dtype=np.float64)

    for i in tqdm(range(m), desc=f"RK44 Lagrange (n={n})", unit="step"):
        u_current = u_history[i] # No copy needed if we don't modify u_current directly

        # Check if previous state is valid before starting RK step
        if not np.all(np.isfinite(u_current)):
             print(f"\nWarning: Invalid state detected at start of time step {i+1} (n={n}). Aborting integration.")
             u_history[i + 1:, :] = np.nan
             return u_history

        try:
            # Stage 1
            R1_flat = spatial_op_func(u_current, n, L, c, alpha) # Calculate R(u^n) = b1
            for k in range(n): # Apply M_inv_local element-wise: K1[k] = M_k^{-1} * b1[k]
                idx = slice(2 * k, 2 * k + 2)
                R_local[:] = R1_flat[idx]
                K_local[:] = M_inv_local @ R_local
                K1_flat[idx] = K_local
            if not np.all(np.isfinite(K1_flat)): raise RuntimeError("NaN/Inf in K1")

            # Stage 2
            u_stage2 = u_current + K1_flat * dt_half # Use dt/2
            R2_flat = spatial_op_func(u_stage2, n, L, c, alpha) # Calculate R(u_stage2) = b2
            for k in range(n): # Apply M_inv_local element-wise: K2[k] = M_k^{-1} * b2[k]
                idx = slice(2 * k, 2 * k + 2)
                R_local[:] = R2_flat[idx]
                K_local[:] = M_inv_local @ R_local
                K2_flat[idx] = K_local
            if not np.all(np.isfinite(K2_flat)): raise RuntimeError("NaN/Inf in K2")

            # Stage 3
            u_stage3 = u_current + K2_flat * dt_half # Use dt/2
            R3_flat = spatial_op_func(u_stage3, n, L, c, alpha) # Calculate R(u_stage3) = b3
            for k in range(n): # Apply M_inv_local element-wise: K3[k] = M_k^{-1} * b3[k]
                idx = slice(2 * k, 2 * k + 2)
                R_local[:] = R3_flat[idx]
                K_local[:] = M_inv_local @ R_local
                K3_flat[idx] = K_local
            if not np.all(np.isfinite(K3_flat)): raise RuntimeError("NaN/Inf in K3")

            # Stage 4
            u_stage4 = u_current + K3_flat * dt # Use full dt
            R4_flat = spatial_op_func(u_stage4, n, L, c, alpha) # Calculate R(u_stage4) = b4
            for k in range(n): # Apply M_inv_local element-wise: K4[k] = M_k^{-1} * b4[k]
                idx = slice(2 * k, 2 * k + 2)
                R_local[:] = R4_flat[idx]
                K_local[:] = M_inv_local @ R_local
                K4_flat[idx] = K_local
            if not np.all(np.isfinite(K4_flat)): raise RuntimeError("NaN/Inf in K4")

            # Final Update: u^{i+1} = u^i + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
            u_next = u_current + dt_sixth * (K1_flat + 2 * K2_flat + 2 * K3_flat + K4_flat) # Use dt/6

             # Final check before assigning to history
            if not np.all(np.isfinite(u_next)):
                raise RuntimeError("NaN/Inf in final u_next")

            u_history[i + 1] = u_next


        except RuntimeError as e: # Catch NaNs/Infs from intermediate stages
             print(f"\nWarning: Instability detected at time step {i+1} (n={n}) - {e}. Aborting RK step.")
             u_history[i + 1:, :] = np.nan
             return u_history # Stop integration
        except Exception as e:
             print(f"\nError during RK step {i+1} (n={n}): {e}")
             # Print traceback for unexpected errors
             import traceback
             traceback.print_exc()
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
    # Ensure input is array for vectorized operations
    x = np.asarray(x)
    # Use modulo to handle periodicity correctly if x is outside [0, L)
    x_mod = np.mod(x, L)
    # Standard 50% duty cycle square wave
    return square(2 * np.pi * x_mod / L, duty=0.5)


# --- Compute Initial Nodal Coefficients (Lagrange P1 Interpolation) ---
def compute_coefficients_lagrangeP1(f_initial_func, L, n):
    """
    Computes initial DG coefficients (nodal values) for P1 Lagrange
    by interpolating the initial condition function at the element nodes.
    Args:
        f_initial_func: Function handle for the initial condition u(x, 0).
        L: Domain length.
        n: Number of elements.
    Returns:
        Flat numpy array (N_dof,) containing the initial nodal values.
    """
    N_dof = n * 2
    u_coeffs = np.zeros((n, 2), dtype=np.float64) # Nodal values (n_elements, 2_nodes)
    dx = L / n
    x_nodes = np.linspace(0, L, n + 1) # Global node locations (includes x=0 and x=L)
    for k in range(n): # Loop over elements
        # Node indices for element k are k and k+1
        x_left_node = x_nodes[k]
        x_right_node = x_nodes[k+1]
        u_coeffs[k, 0] = f_initial_func(x_left_node, L)    # Value at left node
        u_coeffs[k, 1] = f_initial_func(x_right_node, L) # Value at right node
    return u_coeffs.reshape(N_dof) # Return flattened vector

# --- Evaluation function for P1 Lagrange Solution ---
def evaluate_dg_solution_lagrangeP1(x_eval, nodal_vals_flat, L, n):
    """
    Evaluates the P1 Lagrange DG solution u_h(x) at arbitrary points x.
    Uses vectorized operations for efficiency.
    Args:
        x_eval: Numpy array of x-coordinates where solution is needed.
        nodal_vals_flat: Flat numpy array (N_dof,) of nodal values at a specific time.
        L: Domain length.
        n: Number of elements.
    Returns:
        Numpy array of evaluated solution values u_h(x_eval).
    """
    N_dof = n*2
    if nodal_vals_flat is None:
        print("Warning: Evaluating DG solution with None input for nodal_vals_flat. Returning NaNs.")
        # Ensure x_eval is array-like before calling full_like
        return np.full_like(np.asarray(x_eval), np.nan, dtype=float)

    # Check type and shape explicitly
    if not isinstance(nodal_vals_flat, np.ndarray) or nodal_vals_flat.shape != (N_dof,):
         # Handle case where simulation might have failed and returned NaNs
         if isinstance(nodal_vals_flat, np.ndarray) and np.all(np.isnan(nodal_vals_flat)):
             print("Warning: Evaluating DG solution with NaN input array. Returning NaNs.")
             return np.full_like(np.asarray(x_eval), np.nan, dtype=float)
         raise ValueError(f"Expected flat nodal_vals shape ({N_dof},), got {nodal_vals_flat.shape}")

    nodal_vals_element_wise = nodal_vals_flat.reshape((n, 2)) # (n_elem, 2 nodes)
    dx = L / n
    x_eval = np.asarray(x_eval) # Ensure x_eval is a numpy array

    if dx <= 1e-15: # Handle case of invalid mesh
        print(f"Warning: Element width dx={dx} too small in evaluation. Returning NaNs.")
        return np.full_like(x_eval, np.nan, dtype=float)

    # Vectorized approach for finding element index and local coordinate
    x_val_clamped = np.clip(x_eval, 0.0, L) # Clamp all points to domain
    # Calculate element indices, handle right boundary explicitly using clip after floor
    # Subtract small epsilon BEFORE floor to handle points exactly on right boundary
    element_indices = np.floor((x_val_clamped - 1e-14) / dx).astype(int)
    element_indices = np.clip(element_indices, 0, n - 1) # Ensure indices are within [0, n-1]

    # Calculate physical left boundary for each point's element
    x_left_nodes = element_indices * dx
    # Calculate local coordinate xi for all points
    # Add check for dx to prevent division by zero here as well
    if dx > 1e-14:
        xi_vals = 2.0 * (x_val_clamped - x_left_nodes) / dx - 1.0
    else: # Should not happen if check above worked, but defensive
        xi_vals = np.zeros_like(x_val_clamped)
    xi_vals = np.clip(xi_vals, -1.0, 1.0) # Clip xi to [-1, 1]

    # Get nodal values for corresponding elements using advanced indexing
    # Add check for NaN/Inf *before* indexing if needed, although reshape checks finite above
    if not np.all(np.isfinite(nodal_vals_element_wise)):
         print("Warning: NaN/Inf detected in reshaped nodal_vals_element_wise during evaluation.")
         # This might lead to NaNs below, which is handled

    u_left_nodes = nodal_vals_element_wise[element_indices, 0]
    u_right_nodes = nodal_vals_element_wise[element_indices, 1]

    # Evaluate using basis functions (vectorized)
    L1_xi = L1(xi_vals)
    L2_xi = L2(xi_vals)
    u_h_eval = u_left_nodes * L1_xi + u_right_nodes * L2_xi

    # Handle potential NaNs from nodal values *after* evaluation
    nan_mask = np.isnan(u_left_nodes) | np.isnan(u_right_nodes)
    if np.any(nan_mask):
        # Only print if not already warned about input NaNs
        if np.all(np.isfinite(nodal_vals_flat)):
            print("Warning: NaNs detected in specific nodal values during evaluation.")
        u_h_eval[nan_mask] = np.nan

    return u_h_eval


# --- L2 Error Calculation ---
def calculate_l2_error_lagrangeP1(u_nodal_final_flat, f_initial_func, L, n, c, T_final):
    """
    Calculates the L2 error ||u_h - u_exact|| at T_final using Gaussian quadrature.
    Args:
        u_nodal_final_flat: Flat array (N_dof,) of DG nodal values at T_final.
        f_initial_func: Function handle for the initial condition u(x, 0).
        L: Domain length.
        n: Number of elements.
        c: Advection speed.
        T_final: Final time.
    Returns:
        The calculated L2 error (float), or np.nan if calculation fails.
    """
    print(f"Calculating L2 error (Lagrange P1, n={n})...")

    # Check if the final solution is valid
    if u_nodal_final_flat is None or not np.all(np.isfinite(u_nodal_final_flat)):
        print(f"Warning: Cannot calculate L2 error for n={n} due to invalid final solution (NaN/Inf). Returning NaN.")
        return np.nan

    num_quad_points = 5 # Should be sufficient
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
    jacobian = dx / 2.0 # Jacobian of transformation from [-1, 1] to element k

    nodal_vals_element_wise = u_nodal_final_flat.reshape((n, 2))

    # Exact solution function at T_final
    u_exact_final_func = lambda x: u_exact(x, T_final, L, c, f_initial_func)

    # Pre-evaluate basis functions at quadrature points
    L1_at_quad = L1(xi_quad)
    L2_at_quad = L2(xi_quad)

    for k in range(n):
        x_left = k * dx
        # Map reference quad points xi_quad in [-1, 1] to physical points x_quad_k in element k
        x_quad_k = x_left + (xi_quad + 1.0) * jacobian

        # Evaluate DG solution u_h at reference quadrature points within element k
        u_left_node = nodal_vals_element_wise[k, 0]
        u_right_node = nodal_vals_element_wise[k, 1]
        # Vectorized evaluation within the element
        u_h_at_quad_ref = u_left_node * L1_at_quad + u_right_node * L2_at_quad

        # Evaluate exact solution u_ex at physical quadrature points
        u_ex_at_quad_k = u_exact_final_func(x_quad_k)

        # Check for NaNs in either solution at quad points before squaring
        valid_points = np.isfinite(u_h_at_quad_ref) & np.isfinite(u_ex_at_quad_k)
        if not np.all(valid_points):
            print(f"Warning: NaN/Inf encountered during L2 error integrand calculation in element {k}.")
            # Option 1: Skip element (might bias results)
            # continue
            # Option 2: Integrate only over valid points (potentially complex)
            # Option 3: Return NaN for the whole error (safest)
            return np.nan

        # Calculate squared error at quadrature points
        error_sq_at_quad = (u_h_at_quad_ref - u_ex_at_quad_k)**2

        # Add contribution from element k to the total L2 error squared integral
        # Use dot product for weighted sum: sum(w_i * error_sq_i)
        l2_error_sq_sum += np.dot(w_quad, error_sq_at_quad) * jacobian

    # Final check before sqrt
    if l2_error_sq_sum < -1e-14 or not np.isfinite(l2_error_sq_sum): # Allow for small negative values due to FP errors
        print(f"Warning: Invalid L2 error sum ({l2_error_sq_sum}) before sqrt for n={n}. Returning NaN.")
        return np.nan
    elif l2_error_sq_sum < 0:
         l2_error_sq_sum = 0 # Clamp small negative values to zero

    l2_error = np.sqrt(l2_error_sq_sum)
    print(f"L2 Error ||u_h - u_exact|| at T={T_final:.2f} (n={n}) = {l2_error:.6e}")
    return l2_error

# --- Animation Function (Adapted for Lagrange P1) ---
def plot_function_lagrange(u_nodal_history_flat, L, n, dt, m, c, f_initial_func, save=False, tend=0.):
    """
    Creates animation of the P1 Lagrange DG solution vs exact solution.
    Args:
        u_nodal_history_flat: Array (m+1, N_dof) of nodal solution history.
        L, n, dt, m, c: Simulation parameters.
        f_initial_func: Initial condition function handle.
        save: Boolean, whether to save the animation.
        tend: Final time (used for labels).
    Returns:
        Matplotlib animation object (or None if saving fails/no valid frames).
    """
    # Add explicit check for input type
    if not isinstance(u_nodal_history_flat, np.ndarray):
         print(f"Error: Input u_nodal_history_flat is not a NumPy array (type: {type(u_nodal_history_flat)}). Cannot animate.")
         return None

    p_equiv = 1 # P1
    N_dof = n * (p_equiv + 1)
    n_plot_eval_per_elem = 10 # Points per element for smooth plotting
    x_plot_full = np.linspace(0., L, n * n_plot_eval_per_elem + 1)

    # Check if history contains NaNs
    first_nan_step_index = m + 1 # Assume all steps are valid initially
    # Check shape before checking for NaNs
    if u_nodal_history_flat.shape != (m + 1, N_dof):
        print(f"Warning: u_nodal_history_flat has unexpected shape {u_nodal_history_flat.shape}. Expected ({m+1}, {N_dof}).")
        # Attempt to proceed, but might fail later

    nan_indices = np.where(np.isnan(u_nodal_history_flat))
    if len(nan_indices[0]) > 0:
        first_nan_step_index = np.min(nan_indices[0])
        print(f"\nWarning: Simulation history contains NaNs starting at step {first_nan_step_index}.")
        print(f"Animation will only show frames up to step {first_nan_step_index - 1}.")
        if first_nan_step_index == 0:
             print("Error: Initial condition contains NaNs. Cannot animate.")
             return None


    m_plot = first_nan_step_index -1 # Last valid time index to plot
    num_frames_to_show = m_plot + 1 # Number of valid frames

    if num_frames_to_show <= 0:
        print("No valid frames to animate.")
        return None

    # Reconstruct solution u(x,t) up to m_plot
    print("Reconstructing solution for animation...")
    v_plot = np.zeros((num_frames_to_show, len(x_plot_full)))
    reconstruction_failed = False
    for time_idx in tqdm(range(num_frames_to_show), desc="Reconstructing Frames"):
         nodal_vals_flat_at_time = u_nodal_history_flat[time_idx, :]
         try:
             # Use vectorized evaluation
             v_plot[time_idx, :] = evaluate_dg_solution_lagrangeP1(x_plot_full, nodal_vals_flat_at_time, L, n)
         except Exception as e_eval:
             print(f"\nError during solution evaluation for animation frame {time_idx}: {e_eval}")
             # Set frame to NaN and flag failure
             v_plot[time_idx, :] = np.nan
             reconstruction_failed = True

         # Check for NaNs introduced by evaluation (e.g., from NaN nodal values)
         if not np.all(np.isfinite(v_plot[time_idx, :])):
              # Warning now comes from evaluate_dg_solution_lagrangeP1 or catch above
              reconstruction_failed = True # Mark failure but continue

    if reconstruction_failed:
        print("Warning: Animation might show glitches due to NaNs or errors in reconstructed solution.")


    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    ax.grid(True, linestyle=':')

    # Use globally defined font sizes
    global ftSz1, ftSz2, ftSz3

    time_template = r'$t = \mathtt{{{:.4f}}} \;[s]$'
    time_text = ax.text(0.75, 0.90, '', fontsize=ftSz1, transform=ax.transAxes)

    dg_line, = ax.plot([], [], color='g', lw=1.5, label=f'DG Solution (Lagrange P1, n={n}, RK44)')
    exact_func_t = lambda x, t: u_exact(x, t, L, c, f_initial_func)
    exact, = ax.plot([], [], color='r', alpha=0.7, lw=3, zorder=0, label='Exact')

    # Robust calculation of initial exact solution
    try:
        initial_exact_y = exact_func_t(x_plot_full, 0)
        if not np.all(np.isfinite(initial_exact_y)):
            print("Warning: Initial exact solution contains NaNs/Infs.")
    except Exception as e:
        print(f"Error calculating initial exact solution: {e}")
        initial_exact_y = np.zeros_like(x_plot_full) # Fallback


    # Determine plot limits based on valid initial/reconstructed data
    valid_v_plot = v_plot[:num_frames_to_show] # Only consider valid frames
    valid_v_plot_flat = valid_v_plot[np.isfinite(valid_v_plot)]
    valid_initial_exact_y = initial_exact_y[np.isfinite(initial_exact_y)]

    if len(valid_initial_exact_y) == 0 and len(valid_v_plot_flat) == 0:
        print("Warning: Cannot determine plot limits, all data is NaN/Inf.")
        ymin, ymax = -1.2, 1.2 # Fallback limits
    else:
        ymin = np.min(valid_initial_exact_y) if len(valid_initial_exact_y)>0 else np.inf
        ymax = np.max(valid_initial_exact_y) if len(valid_initial_exact_y)>0 else -np.inf
        if len(valid_v_plot_flat) > 0:
             # Use nanmin/nanmax in case reconstruction failed partially
             ymin = min(ymin, np.nanmin(valid_v_plot_flat))
             ymax = max(ymax, np.nanmax(valid_v_plot_flat))

    yrange = ymax - ymin
    if abs(yrange) < 1e-6 or not np.isfinite(yrange):
        yrange = 1.0 # Avoid zero/NaN range
        ymin = -0.5 # Reset if range was invalid
        ymax = 0.5
    ax.set_ylim(ymin - 0.15 * yrange, ymax + 0.15 * yrange)
    ax.set_xlim(0, L)
    ax.set_xlabel(r"$x$", fontsize=ftSz2)
    ax.set_ylabel(r"$u(x,t)$", fontsize=ftSz2)
    ax.legend(fontsize=ftSz3)
    title_str = f"DG P1 Lagrange vs Exact (IC: {f_initial_func.__name__.replace('ic_', '')})"
    ax.set_title(title_str, fontsize=ftSz1)


    def init():
        dg_line.set_data([], [])
        exact.set_data(x_plot_full, initial_exact_y)
        time_text.set_text(time_template.format(0))
        return tuple([dg_line, exact, time_text])

    def animate(t_idx):
        # t_idx corresponds to the frame number, which is the time step index
        if t_idx < num_frames_to_show:
             current_time = t_idx * dt
             # Plotting reconstructed data, which might contain NaNs if flagged
             dg_line.set_data(x_plot_full, v_plot[t_idx, :])
             try: # Recalculate exact solution safely
                  exact_y_data = exact_func_t(x_plot_full, current_time)
                  exact.set_ydata(exact_y_data)
             except Exception as e_exact:
                  print(f"Error updating exact solution at t={current_time}: {e_exact}")
                  exact.set_ydata(np.full_like(x_plot_full, np.nan)) # Show error as NaN

             time_text.set_text(time_template.format(current_time))
        # No else needed, FuncAnimation should stop after num_frames_to_show
        return tuple([dg_line, exact, time_text])

    fps = 30
    interval = max(1, int(1000.0 / fps)) # milliseconds per frame

    print("Creating animation...")
    # Add try-except around FuncAnimation creation itself
    try:
        anim = FuncAnimation(fig, animate, frames=num_frames_to_show, interval=interval, blit=False,
                             init_func=init, repeat=False)
    except Exception as e_anim_create:
         print(f"\n--- Error Creating Animation Object ---")
         print(f"Error: {e_anim_create}")
         print("-------------------------------------\n")
         plt.close(fig) # Close the figure if animation fails
         return None

    if save:
        output_dir = './figures'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        ic_name_str = f_initial_func.__name__.replace('ic_', '')
        anim_filename = os.path.join(output_dir, f"dg_advection_p1lagrange_{ic_name_str}_n{n}_RK44.mp4")
        print(f"Saving animation to {anim_filename}...")
        try:
            # Check if FFmpegWriter is available more robustly
            writer_path = None
            try:
                 writer_path = plt.rcParams['animation.ffmpeg_path']
                 if writer_path == 'ffmpeg':
                     if shutil.which('ffmpeg') is None: writer_path = None
            except KeyError:
                 if shutil.which('ffmpeg') is not None: writer_path = 'ffmpeg'
                 else: writer_path = None

            if writer_path is None:
                print("\n--- FFmpeg Not Found ---")
                print("Cannot save animation. FFmpeg writer not configured.")
                print("Install FFmpeg and ensure it's in PATH or set rcParam.")
                print("------------------------\n")
                raise RuntimeError("FFmpeg not found, cannot save animation.")

            writerMP4 = FFMpegWriter(fps=fps)
            anim.save(anim_filename, writer=writerMP4)
            print("Animation saved successfully.")
            plt.close(fig) # Close plot after successful save
        except Exception as e:
            print(f"\n--- Animation Save/Show Error ---")
            print(f"Error: {e}")
            print("Showing interactive plot instead (if possible).")
            print("----------------------------\n")
            try:
                plt.show() # Show interactively if saving fails
                plt.close(fig) # Close after showing
            except Exception as e_show:
                print(f"Error showing plot interactively: {e_show}")
            return None # Indicate saving failed
    else:
        # Add try-except around plt.show() as well
        try:
            plt.show() # Show interactively if not saving
            plt.close(fig) # Close after showing
        except Exception as e_show:
             print(f"\n--- Interactive Plot Error ---")
             print(f"Error: {e_show}")
             print("------------------------------\n")
             # Ensure figure is closed even if show fails
             plt.close(fig)
             return None # Indicate failure to show

    return anim


# --- Core Simulation Logic (No Plotting/Animation) ---
def run_simulation_core(L, n, dt, m, c, f_initial_func, alpha, rktype='RK44'):
    """
    Runs the DG simulation and returns the full solution history.
    Args:
        L, n, dt, m, c: Simulation parameters.
        f_initial_func: Initial condition function handle.
        alpha: Upwind parameter.
        rktype: Time integration method (only 'RK44' supported).
    Returns:
        u_history: (m+1, N_dof) numpy array of solution history, or None if setup fails.
                   May contain NaNs if time integration becomes unstable.
    """
    p_equiv = 1 # P1 Lagrange
    N_dof = n * (p_equiv + 1) # n * 2 DoFs

    # Calculate local inverse mass matrix (M_k^{-1})
    dx = L / n
    if dx <= 1e-15:
        print(f"Error: dx={dx:.3e} too small in run_simulation_core for n={n}.")
        return None # Indicate failure
    try:
        M_inv_local = get_Minv_local_lagrangeP1(dx)
    except ValueError as e:
         print(f"Error creating local inverse mass matrix for n={n}: {e}")
         return None


    # Initialize solution history array
    u_history = np.zeros((m + 1, N_dof), dtype=np.float64)

    # Set initial condition
    try:
        u_history[0] = compute_coefficients_lagrangeP1(f_initial_func, L=L, n=n)
        if not np.all(np.isfinite(u_history[0])):
             print(f"Error: Initial condition computation resulted in NaN/Inf for n={n}.")
             return None
    except Exception as e:
         print(f"Error computing initial conditions for n={n}: {e}")
         return None


    # Perform time stepping
    if rktype == 'RK44':
        u_history = rk44_lagrange_local_Minv(
            u_history, spatial_operator_lagrangeP1, M_inv_local,
            dt, m, n, L, c, alpha)
        # rk44 function handles internal failures and returns history with NaNs
    else:
        print(f"Error: Time integration method '{rktype}' is not implemented.")
        return None # Indicate failure

    # Return the full history (may contain NaNs if RK failed)
    return u_history


# --- Main DG Solver Function (Lagrange P1 Wrapper - Calls Plotting/Core) ---
def advection1d_lagrangeP1_solver(L, n, dt, m, c, f_initial_func, alpha, rktype='RK44',
                                 anim=True, save=False, tend=0., plot_final=True):
    """
    Wrapper function to run the P1 Lagrange DG simulation and optionally
    plot/animate the results.
    Args:
        L, n, dt, m, c: Simulation parameters.
        f_initial_func: Initial condition function handle.
        alpha: Upwind parameter.
        rktype: Time integration method.
        anim: Boolean, whether to create animation.
        save: Boolean, whether to save animation (if anim=True).
        tend: Final time (for plotting/labeling).
        plot_final: Boolean, whether to plot final comparison if anim=False.
    Returns:
        tuple: (u_final_flat, l2_error, animation_object)
               u_final_flat: Final state vector (N_dof,) or None on failure.
               l2_error: L2 error at final time or np.nan on failure.
               animation_object: Matplotlib animation object or None.
    """

    # Run the core simulation
    u_history_flat = run_simulation_core(L, n, dt, m, c, f_initial_func, alpha, rktype)

    # Check if simulation core returned valid history
    if u_history_flat is None:
        print(f"\n--- Simulation setup failed for n={n} ---")
        return None, np.nan, None # Return failure indication

    # Check for NaNs in the final state specifically
    # Use m as index for final state (since history has size m+1)
    u_final_flat = u_history_flat[m].copy()
    simulation_succeeded = np.all(np.isfinite(u_final_flat))

    if not simulation_succeeded:
        print(f"\n--- Simulation failed (NaN/Inf detected in final state) for n={n} ---")
        # Still try to plot/animate up to failure point
    else:
        print(f"\n--- Simulation completed successfully for n={n} ---")


    # --- Post-processing ---
    animation_object = None
    if anim:
        print("\nAttempting to generate animation...")
        # Pass the full history, plot_function_lagrange handles NaNs internally
        # Add try-except around the call itself
        try:
            animation_object = plot_function_lagrange(
                u_history_flat, L=L, n=n, dt=dt, m=m, c=c,
                f_initial_func=f_initial_func, save=save, tend=tend)
            if animation_object is None and save:
                 print("Animation saving/showing failed.")
            elif animation_object is None and not save:
                 print("Animation showing failed.")

        except Exception as e_plot:
            print(f"\n--- Error during animation/plotting call for n={n} ---")
            print(f"Error: {e_plot}")
            import traceback
            traceback.print_exc()
            print("------------------------------------------------------\n")
            animation_object = None # Ensure it's None


    # Plot final comparison only if simulation succeeded and animation was off
    if plot_final and not anim and simulation_succeeded:
        print("\nPlotting final solution comparison...")
        n_plot_points_per_element = 50 # Fine resolution for plotting
        x_plot = np.linspace(0, L, n * n_plot_points_per_element + 1)
        try:
            u_h_final_lagrange = evaluate_dg_solution_lagrangeP1(x_plot, u_final_flat, L, n)
            u_ex_final = u_exact(x_plot, tend, L, c, f_initial_func)

            # Check if evaluation resulted in NaNs
            if not np.all(np.isfinite(u_h_final_lagrange)):
                 print("Warning: Evaluation of final DG solution resulted in NaNs.")
            if not np.all(np.isfinite(u_ex_final)):
                 print("Warning: Evaluation of final Exact solution resulted in NaNs.")


            plt.figure(figsize=(10, 6))
            plt.plot(x_plot, u_ex_final, 'r-', linewidth=3, alpha=0.7, label=f'Exact Solution at T={tend:.2f}')
            plt.plot(x_plot, u_h_final_lagrange, 'g-', linewidth=1.5, label=f'DG Solution (Lagrange P1, n={n}, RK44)')
            # Add element boundaries
            for k_elem in range(n + 1): plt.axvline(k_elem * L / n, color='gray', linestyle=':', linewidth=0.5)

            # Use global font sizes
            plt.xlabel("x", fontsize=ftSz2)
            plt.ylabel(f"u(x, T={tend:.1f})", fontsize=ftSz2)
            ic_name = f_initial_func.__name__.replace('ic_','')
            plt.title(f"DG Lagrange P1 vs Exact at T={tend:.2f} (IC: {ic_name}, n={n})", fontsize=ftSz1)
            plt.legend(fontsize=ftSz3)
            plt.grid(True, linestyle=':')
            # Calculate reasonable Y limits
            # Handle potential NaNs in evaluation
            valid_uh = u_h_final_lagrange[np.isfinite(u_h_final_lagrange)]
            valid_uex = u_ex_final[np.isfinite(u_ex_final)]
            ymin = -1.2 # Default fallback
            ymax = 1.2
            if len(valid_uh)>0 and len(valid_uex)>0:
                ymin = min(np.min(valid_uex), np.min(valid_uh)) - 0.2
                ymax = max(np.max(valid_uex), np.max(valid_uh)) + 0.2
            elif len(valid_uex)>0:
                ymin = np.min(valid_uex) - 0.2
                ymax = np.max(valid_uex) + 0.2
            elif len(valid_uh)>0:
                 ymin = np.min(valid_uh) - 0.2
                 ymax = np.max(valid_uh) + 0.2

            if abs(ymax - ymin) < 1e-6: ymax = ymin + 1.0 # Handle constant solutions
            plt.ylim(ymin, ymax)
            plt.show()
            plt.close() # Close the plot window

        except Exception as e_final_plot:
             print(f"\n--- Error during final plot generation for n={n} ---")
             print(f"Error: {e_final_plot}")
             print("--------------------------------------------------\n")


    # Calculate L2 error at final time (only if simulation succeeded)
    l2_error = np.nan
    if simulation_succeeded:
        try:
            l2_error = calculate_l2_error_lagrangeP1(u_final_flat, f_initial_func, L, n, c, tend)
        except Exception as e_l2:
            print(f"\n--- Error during L2 error calculation for n={n} ---")
            print(f"Error: {e_l2}")
            print("------------------------------------------------\n")
            l2_error = np.nan # Ensure it's NaN on error
    else:
        print(f"Skipping L2 error calculation for n={n} due to simulation failure.")


    # Return None for u_final_flat if sim failed, otherwise return the final state
    final_state_to_return = u_final_flat if simulation_succeeded else None
    return final_state_to_return, l2_error, animation_object


# --- Convergence Study Function ---
# CORRECTED plotting order and reporting asymptotic rate
def run_convergence_study(L, c, T_final, f_initial_func, alpha_flux, rk_method, n_values, cfl_target=None, dt_fixed=None):
    """
    Performs a convergence study for the DG method using P1 Lagrange elements.
    Runs simulations for a list of element counts (n_values) and calculates
    the L2 error against the exact solution. Plots error vs element size (h)
    and prints a formatted results table, reporting the asymptotic rate.

    Args:
        L, c, T_final: Domain length, advection speed, final time.
        f_initial_func: Function handle for the initial condition.
        alpha_flux: Upwind parameter for the spatial operator.
        rk_method: Time integration method (e.g., 'RK44').
        n_values: List or array of element counts [n1, n2, ...].
        cfl_target: Target CFL number (dt determined based on c, dx, cfl_target).
        dt_fixed: Fixed time step size (overrides cfl_target if provided).

    Returns:
        tuple: (h_values, l2_errors) containing numpy arrays of element sizes
               and corresponding L2 errors. Includes NaNs for failed runs.
    """
    print("\n--- Starting Convergence Study ---")
    is_discontinuous = 'square' in f_initial_func.__name__
    if not callable(f_initial_func):
        raise TypeError("f_initial_func must be a callable function.")
    if is_discontinuous:
        print(f"Warning: Using discontinuous IC '{f_initial_func.__name__}' for convergence study.")
        print("         Observed rates may be lower than theoretical smooth rates (O(h^2))")
        print("         and potentially less stable.")

    if cfl_target is None and dt_fixed is None:
        raise ValueError("Convergence study requires either cfl_target or dt_fixed to be specified.")
    if cfl_target is not None and dt_fixed is not None:
        print("Warning: Both cfl_target and dt_fixed provided for convergence study. Using cfl_target.")
        dt_fixed = None # Prioritize CFL target

    l2_errors = []
    h_values = []
    sim_success = [] # Track success/failure for each n

    for n_conv in n_values:
        print(f"\nRunning Convergence Study for n = {n_conv}")
        dx_conv = L / n_conv
        if dx_conv <= 1e-15:
             print(f"Warning: dx={dx_conv:.3e} too small for n={n_conv}. Skipping.")
             l2_errors.append(np.nan)
             h_values.append(np.nan) # Use NaN for h if skipped
             sim_success.append(False)
             continue
        h_values.append(dx_conv)

        # Determine dt and m for this n
        dt_conv = 0.0
        actual_cfl = np.nan
        if dt_fixed is not None:
            if dt_fixed <= 0: raise ValueError("dt_fixed must be positive.")
            dt_conv = dt_fixed
        else: # Use CFL target
            if cfl_target is None or cfl_target <= 0: raise ValueError("cfl_target must be positive.")
            # Calculate dt based on CFL condition: dt = CFL * dx / |c|
            if abs(c) > 1e-14: # Avoid division by zero if c is very small
                 dt_conv = cfl_target * dx_conv / abs(c)
            else: # Handle c=0 case (e.g., stationary solution)
                 num_steps_if_c_zero = 200 # Arbitrary choice
                 dt_conv = T_final / num_steps_if_c_zero
                 print(f"Warning: c=0, using fixed dt={dt_conv:.3e} ({num_steps_if_c_zero} steps).")

            if dt_conv <= 1e-15:
                 print(f"Warning: Calculated dt={dt_conv:.3e} too small for n={n_conv} based on CFL target. Skipping.")
                 l2_errors.append(np.nan)
                 # Use index -1 to access the last added h_value
                 h_values[-1] = np.nan # Mark corresponding h as NaN too
                 sim_success.append(False)
                 continue

        # Calculate number of steps m and adjust dt to reach T_final exactly
        m_conv = max(1, int(np.ceil(T_final / dt_conv)))
        dt_adjusted_conv = T_final / m_conv

        # Calculate the actual CFL number achieved with the adjusted dt
        if abs(c) > 1e-14 and dx_conv > 1e-14:
            actual_cfl = abs(c) * dt_adjusted_conv / dx_conv
        else:
            actual_cfl = 0.0 # CFL is not strictly defined if c=0 or dx=0

        print(f"  dx = {dx_conv:.4e}")
        print(f"  m = {m_conv}, dt = {dt_adjusted_conv:.4e}, Actual CFL = {actual_cfl:.3f}")

        # Run simulation core (no plotting needed here)
        u_history_conv = run_simulation_core(
            L, n_conv, dt_adjusted_conv, m_conv, c,
            f_initial_func, alpha_flux, rk_method
        )

        # Check if simulation failed
        if u_history_conv is None or not np.all(np.isfinite(u_history_conv[m_conv])):
             print(f"Simulation failed for n={n_conv}, cannot calculate L2 error.")
             l2_errors.append(np.nan)
             sim_success.append(False)
        else:
            # Simulation succeeded, calculate L2 error
            u_final_conv = u_history_conv[m_conv]
            l2_err_n = calculate_l2_error_lagrangeP1(u_final_conv, f_initial_func, L, n_conv, c, T_final)
            l2_errors.append(l2_err_n) # Appends NaN if error calc failed internally
            # Update success flag based on whether L2 error calculation itself failed
            sim_success.append(l2_err_n is not np.nan and np.isfinite(l2_err_n))

    # --- Process and Plot Convergence Results ---
    h_values = np.array(h_values)
    l2_errors = np.array(l2_errors)

    # Filter out non-finite errors/h_values AND points where error is too small for log plot
    valid_mask = np.isfinite(h_values) & np.isfinite(l2_errors) & (l2_errors > 1e-15)
    h_valid = h_values[valid_mask]
    l2_errors_valid = l2_errors[valid_mask]
    # Ensure n_values is treated as a numpy array for boolean indexing
    n_values_valid = np.array(n_values)[valid_mask] # Get corresponding n for valid points


    rates = []
    n_sorted = np.array([]) # Initialize as empty array
    h_sorted = np.array([])
    l2_errors_sorted = np.array([])
    if len(h_valid) > 1:
        # Sort by h descending to calculate rates between successive refinements
        sort_indices = np.argsort(h_valid)[::-1] # Indices for h largest to smallest
        h_sorted = h_valid[sort_indices]
        l2_errors_sorted = l2_errors_valid[sort_indices]
        n_sorted = n_values_valid[sort_indices] # Keep track of n for printing

        # Estimate convergence rate: Order = log(E_coarse/E_fine) / log(h_coarse/h_fine)
        # Add small value to avoid log(0) issues, although filtered above
        log_errors = np.log(l2_errors_sorted + 1e-30)
        log_h = np.log(h_sorted + 1e-30)

        # Calculate rates comparing consecutive points in the sorted list
        rates = (log_errors[:-1] - log_errors[1:]) / (log_h[:-1] - log_h[1:])

    # --- Print Results Table --- #
    print("\n--- Convergence Study Results (Lagrange P1) ---")
    print("  n    |    h       |   L2 Error   | Approx. Rate")
    print("-------|------------|--------------|--------------")

    if len(h_valid) > 0:
        # Print the first point (coarsest in the sorted list)
        print(f"{n_sorted[0]:>6d} | {h_sorted[0]:<10.6f} | {l2_errors_sorted[0]:<12.6e} |     -    ")
        # Print subsequent points and the rate obtained refining from the previous
        for i in range(len(rates)):
             # Ensure index i+1 is valid for sorted arrays
             if i + 1 < len(n_sorted):
                 # Format rate, handle potential NaN/Inf
                 rate_str = f"{rates[i]:<7.3f}" if np.isfinite(rates[i]) else "  N/A  "
                 print(f"{n_sorted[i+1]:>6d} | {h_sorted[i+1]:<10.6f} | {l2_errors_sorted[i+1]:<12.6e} | {rate_str}")
             else:
                 print(f"Warning: Mismatch in array lengths during results table printing.")
    else:
        # If no valid points, print message and original data if helpful
        print("  No valid data points found for convergence analysis.")
        print("\n  Original Data (for debugging):")
        for i in range(len(n_values)):
             n_orig = n_values[i]
             # Safely access h_values, l2_errors, sim_success with bounds check
             h_orig_str = f"{h_values[i]:.6f}" if i < len(h_values) and np.isfinite(h_values[i]) else 'N/A'
             err_orig_str = f"{l2_errors[i]:.6e}" if i < len(l2_errors) and np.isfinite(l2_errors[i]) else 'N/A'
             success_str = str(sim_success[i]) if i < len(sim_success) else 'N/A'
             print(f"    n={n_orig:<4d} | h={h_orig_str:<10s} | L2={err_orig_str:<12s} | Success={success_str}")

    print("---------------------------------")
    # --- End Print Results Table --- #

    # --- Report Asymptotic Rate (from finest grids) --- #
    if len(rates) > 0:
        asymptotic_rate = rates[-1] # Rate from the two finest grids
        if np.isfinite(asymptotic_rate):
            print(f"Asymptotic Observed Rate (finest grids): {asymptotic_rate:.3f}")
        else:
            print("Could not calculate a finite asymptotic rate (finest grids).")
    else:
        print("Could not calculate any convergence rates.")

    # Modify expected rate comment based on IC smoothness
    p_basis = 1
    expected_rate_smooth = p_basis + 1
    if is_discontinuous:
         print(f"(Note: Using discontinuous IC '{f_initial_func.__name__}'.")
         print(f" Expected rate is typically < {expected_rate_smooth}. Often O(h^0.5) to O(h^1) in L2 norm.)")
    else:
         print(f"(Expected rate for P{p_basis} elements is ~{expected_rate_smooth:.1f} for smooth solutions)")


    # --- Plotting ---
    if len(h_valid) > 0:
        plt.figure(figsize=(8, 6))

        # **** CORRECTED PLOTTING ORDER ****
        # Sort the valid data by h (ascending) for correct line connection
        plot_sort_indices = np.argsort(h_valid)
        h_plot_sorted = h_valid[plot_sort_indices]
        l2_errors_plot_sorted = l2_errors_valid[plot_sort_indices]
        # Plot the calculated L2 errors using sorted data
        plt.loglog(h_plot_sorted, l2_errors_plot_sorted, 'bo-', markerfacecolor='none', markersize=8, label='DG P1 L2 Error')
        # **** END CORRECTION ****

        # Plot reference line for expected SMOOTH rate (O(h^2)), add others if needed
        order_expected_smooth = p_basis + 1
        # Scale reference line to pass through the point with smallest h
        idx_finest = np.argmin(h_valid) # Index in the original valid arrays
        C_ref_smooth = l2_errors_valid[idx_finest] / (h_valid[idx_finest]**order_expected_smooth)
        # Generate h values for plotting the reference line spanning the range of valid h
        h_plot_ref = np.array([min(h_valid), max(h_valid)]) # Use min/max for range
        plt.loglog(h_plot_ref, C_ref_smooth * h_plot_ref**order_expected_smooth,
                   'r--', label=f'$\\mathcal{{O}}(h^{order_expected_smooth})$ Ref (Smooth IC)')

        # Optionally add a reference line for the expected discontinuous rate (e.g., O(h^1) or O(h^0.5))
        if is_discontinuous:
            # Let's check O(h^0.5) as that's theoretically common for L2 error with jumps
            order_expected_disc = 0.5
            C_ref_disc = l2_errors_valid[idx_finest] / (h_valid[idx_finest]**order_expected_disc)
            plt.loglog(h_plot_ref, C_ref_disc * h_plot_ref**order_expected_disc,
                       'm:', label=f'$\\mathcal{{O}}(h^{order_expected_disc:.1f})$ Ref (Example)')


        # Use global font sizes
        plt.xlabel("Element Size $h = L/n$", fontsize=ftSz2)
        plt.ylabel("$L_2$ Error at $T_{final}$", fontsize=ftSz2)
        ic_name = f_initial_func.__name__.replace('ic_', '')
        plt.title(f"DG P1 Lagrange Convergence (IC: {ic_name})", fontsize=ftSz1)
        # Keep invert_xaxis(): plots h decreasing L->R (standard)
        plt.gca().invert_xaxis()
        plt.grid(True, which='both', linestyle=':')
        plt.legend(fontsize=ftSz3)
        plt.show()
        plt.close() # Close the plot
    else:
        print("Skipping convergence plot as no valid data points were generated.")

    return h_values, l2_errors


# --- Exact Solution Function ---
def u_exact(x, t, L, c, initial_func):
    """
    Calculates the exact solution u(x,t) = u0(x-ct) for the 1D linear
    advection equation with periodic boundary conditions.
    Args:
        x: Position(s) (scalar or numpy array).
        t: Time.
        L: Domain length.
        c: Advection speed.
        initial_func: Function handle u0(x, L) for the initial condition.
    Returns:
        Exact solution value(s) at x, t.
    """
    x = np.asarray(x)
    # Calculate the characteristic origin position x0 = x - ct
    x_origin = x - c * t
    # Apply periodic boundary condition: map x_origin back into [0, L)
    # np.mod handles periodicity correctly for both positive and negative x_origin
    x_origin_periodic = np.mod(x_origin, L)
    # Evaluate the initial condition function at the periodic origin position
    return initial_func(x_origin_periodic, L)


# --- Comparison Plot for Different n (Lagrange) ---
def plot_comparison_n_lagrange(L, c, T_final, f_initial_func, alpha_flux, rk_method, n_values_to_compare, cfl_target):
    """
    Runs the DG Lagrange P1 simulation for several values of n and plots the
    final solutions at T_final against the exact solution on the same axes.

    Args:
        L, c, T_final: Domain, speed, final time.
        f_initial_func: Initial condition function handle.
        alpha_flux: Upwind parameter.
        rk_method: Time integration method.
        n_values_to_compare: List of element counts [n1, n2, ...].
        cfl_target: Target CFL number for determining dt for each n.
    """
    print(f"\n--- Plotting Comparison for Different n (Lagrange P1) ---")
    if not callable(f_initial_func):
        print("Error: Invalid initial condition function provided for comparison plot.")
        return
    # Handle empty list case
    if not n_values_to_compare:
        print("Warning: No n values provided for comparison plot.")
        return

    # Define plotting parameters
    n_plot_points_per_element = 30 # Resolution for plotting DG solution
    # Use finest n for base plot resolution, ensure at least some points
    n_fine_plot_basis = max(max(n_values_to_compare), 10)
    x_plot = np.linspace(0, L, n_fine_plot_basis * n_plot_points_per_element + 1)

    # --- Calculate Exact Solution and Plot Limits FIRST ---
    try:
        u_ex_final = u_exact(x_plot, T_final, L, c, f_initial_func)
        valid_uex = u_ex_final[np.isfinite(u_ex_final)]
        if len(valid_uex) == 0: raise ValueError("Exact solution is all NaN/Inf")
        ymin_plot = np.min(valid_uex) - 0.2
        ymax_plot = np.max(valid_uex) + 0.2
        if abs(ymax_plot - ymin_plot) < 1e-6: ymax_plot = ymin_plot + 1.0
    except Exception as e:
        print(f"Error calculating exact solution for plot: {e}")
        # Set default plot limits if exact solution fails
        ymin_plot, ymax_plot = -1.2, 1.2 # Default fallback

    # --- Setup Plot ---
    plt.figure(figsize=(12, 7)) # Slightly wider for multiple lines
    plt.plot(x_plot, u_ex_final, 'k--', linewidth=2.5, label=f'Exact T={T_final:.1f}')
    # Generate distinct colors
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_values_to_compare)))

    # --- Loop through n values, simulate, and plot ---
    all_sims_failed = True # Track if any sim succeeds
    for i, n_comp in enumerate(n_values_to_compare):
        print(f"\nRunning for comparison plot: n={n_comp}")
        dx_comp = L / n_comp
        dt_comp = 0.0
        actual_cfl_comp = np.nan

        if dx_comp <= 1e-15: print(f"Warn: dx={dx_comp:.3e} too small, skip n={n_comp}."); continue

        # Determine dt based on CFL
        if abs(c) > 1e-14:
            dt_comp = cfl_target * dx_comp / abs(c)
        else:
            dt_comp = T_final / 100.0 # Default dt if c=0
        if dt_comp <= 1e-15: print(f"Warn: dt={dt_comp:.3e} too small, skip n={n_comp}."); continue

        m_comp = max(1, int(np.ceil(T_final / dt_comp)))
        dt_adjusted_comp = T_final / m_comp

        if abs(c) > 1e-14 and dx_comp > 1e-14 : actual_cfl_comp = abs(c) * dt_adjusted_comp / dx_comp
        else: actual_cfl_comp = 0.0

        print(f"  dx={dx_comp:.3e}, dt={dt_adjusted_comp:.3e}, m={m_comp}, CFL={actual_cfl_comp:.3f}")

        # Run simulation core
        u_history_comp = run_simulation_core(L, n_comp, dt_adjusted_comp, m_comp, c, f_initial_func, alpha_flux, rk_method)

        label_n = f'DG P1 (n={n_comp})'
        # Check if simulation succeeded
        if u_history_comp is None or not np.all(np.isfinite(u_history_comp[m_comp])):
            print(f"  Simulation failed for n={n_comp}, cannot plot result.")
            # Add a placeholder entry to the legend for failed runs
            plt.plot([],[], linestyle='-', color=colors[i], label=f'{label_n} - FAILED')
        else:
            # Simulation succeeded, evaluate final state
            all_sims_failed = False # At least one sim worked
            u_final_flat_comp = u_history_comp[m_comp]
            u_h_final = evaluate_dg_solution_lagrangeP1(x_plot, u_final_flat_comp, L, n_comp)

            # Plot the evaluated DG solution
            if np.all(np.isfinite(u_h_final)):
                plt.plot(x_plot, u_h_final, linestyle='-', color=colors[i], linewidth=1.5, label=label_n)
                # Update plot limits if necessary based on this simulation's results
                valid_uh = u_h_final[np.isfinite(u_h_final)]
                if len(valid_uh)>0:
                     current_min = np.min(valid_uh) - 0.2
                     current_max = np.max(valid_uh) + 0.2
                     ymin_plot = min(ymin_plot, current_min)
                     ymax_plot = max(ymax_plot, current_max)
            else:
                print(f"  Warning: Evaluation of DG solution for n={n_comp} resulted in NaN/Inf. Plotting skipped.")
                plt.plot([],[], linestyle='-', color=colors[i], label=f'{label_n} - NaN Eval')

    # --- Finalize Plot ---
    if all_sims_failed:
        print("\nWarning: All simulations for the n-comparison plot failed.")

    # Use global font sizes ftSz1, ftSz2, ftSz3
    plt.xlabel("x", fontsize=ftSz2); plt.ylabel(f"u(x, T={T_final:.1f})", fontsize=ftSz2)
    ic_name = f_initial_func.__name__.replace('ic_','')
    plt.title(f"Comparison of DG Lagrange P1 Solutions (IC: {ic_name})", fontsize=ftSz1)
    # Adjust legend font size if many lines
    legend_fontsize = max(8, ftSz3 - (len(n_values_to_compare)//3))
    plt.legend(fontsize=legend_fontsize); plt.grid(True, linestyle=':')
    # Adjust final Y limits in case DG solutions exceeded initial range
    if abs(ymax_plot - ymin_plot) < 1e-6: ymax_plot = ymin_plot + 1.0
    plt.ylim(ymin_plot, ymax_plot);
    plt.xlim(0, L); plt.show()
    plt.close() # Close the plot


# ==============================================================
# --- Main Execution Block ---
# ==============================================================
if __name__ == "__main__":

    # --- Select Mode ---
    run_normal_simulation = True   # Run a single simulation with animation/final plot
    run_conv_study = True      # Run convergence study (error vs h)
    run_n_comparison_plot = True # Run plot comparing final solutions for different n

    # --- Configuration (Shared) ---
    L_ = 1.0          # Domain Length
    c_ = 1.0          # Advection Speed
    T_final = 1.0     # Final Time (e.g., one period for c=1, L=1)
    rk_method = 'RK44'
    upwind_param = 1.0 # Use 1.0 for upwind flux based on sign(c)

    # --- Plot P1 Lagrange Basis Functions (Runs Once) ---
    # (Code unchanged)
    print("Plotting P1 Lagrange Basis Functions...")
    xi_plot_basis = np.linspace(-1, 1, 200); L1_vals = L1(xi_plot_basis); L2_vals = L2(xi_plot_basis)
    plt.figure(figsize=(10, 6))
    plt.plot(xi_plot_basis, L1_vals, 'b-', lw=2, label=r'$L_1(\xi) = \phi_{k,1} = (1 - \xi)/2$ (Node k)')
    plt.plot(xi_plot_basis, L2_vals, 'g-', lw=2, label=r'$L_2(\xi) = \phi_{k,2} = (1 + \xi)/2$ (Node k+1)')
    plt.plot([-1], [1], 'bo', markersize=8, markerfacecolor='b'); plt.plot([1], [0], 'bo', markersize=8, markerfacecolor='w', markeredgecolor='b')
    plt.plot([-1], [0], 'go', markersize=8, markerfacecolor='w', markeredgecolor='g'); plt.plot([1], [1], 'go', markersize=8, markerfacecolor='g')
    plt.title("P1 Lagrange Shape Functions on Reference Element [-1, 1]", fontsize=ftSz1); plt.xlabel("Reference Coordinate $\\xi$", fontsize=ftSz2); plt.ylabel("Shape Function Value", fontsize=ftSz2)
    plt.xticks(np.linspace(-1, 1, 9)); plt.yticks(np.linspace(0, 1, 6)); plt.grid(True, linestyle=':'); plt.axhline(0, color='black', linewidth=0.5); plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(fontsize=ftSz3); plt.ylim(-0.1, 1.1); plt.show(); plt.close()
    # --- End Plot Basis Functions ---


    # --- Configuration and Run Normal Simulation ---
    f_initial_sim = None # Define outside the if block to potentially use later
    initial_condition_name = '' # Define outside the if block
    if run_normal_simulation:
        n_sim = 40 # Number of elements for the detailed run
        # **** Use SQUARE WAVE for normal sim ****
        initial_condition_name = 'square' # Options: 'sine' or 'square'
        cfl_target_sim = 0.2 # CFL target for this run
        animate_sim = True   # Generate animation?
        save_anim = False    # Save animation file? (Requires FFmpeg)
        plot_final_sim = True # Plot final state comparison if animation is off?

        # Select Initial Condition function
        if initial_condition_name == 'sine':
            f_initial_sim = ic_sine_wave
        elif initial_condition_name == 'square':
            f_initial_sim = ic_square_wave
        else:
            print(f"Error: Unknown initial_condition_name: {initial_condition_name}")
            f_initial_sim = None # Ensure it's None if invalid name

        if f_initial_sim is not None:
            # Calculate simulation parameters
            dx_sim = L_ / n_sim
            dt_cfl_sim = 0.0
            if abs(c_) > 1e-14:
                dt_cfl_sim = cfl_target_sim * dx_sim / abs(c_)
            else:
                dt_cfl_sim = T_final / 200.0 # Default dt steps if c=0
            if dt_cfl_sim <= 1e-15:
                print(f"Error: Calculated dt={dt_cfl_sim:.3e} is too small for normal simulation (n={n_sim}). Aborting.")
            else:
                m_sim = max(1, int(np.ceil(T_final / dt_cfl_sim)))
                dt_adjusted_sim = T_final / m_sim
                actual_cfl_sim = abs(c_) * dt_adjusted_sim / dx_sim if (abs(dx_sim) > 1e-14 and abs(c_) > 1e-14) else 0.0

                print(f"\n--- Running Normal Simulation ---")
                print(f"  n = {n_sim}, L = {L_}, c = {c_}, T_final = {T_final:.3f}")
                print(f"  IC: {initial_condition_name}, Upwind alpha: {upwind_param}")
                print(f"  Target CFL = {cfl_target_sim}, Actual CFL = {actual_cfl_sim:.3f}")
                print(f"  m = {m_sim}, dt = {dt_adjusted_sim:.6f}")

                # Run the solver
                u_final_sim, l2_err_sim, anim_obj = advection1d_lagrangeP1_solver(
                    L_, n_sim, dt_adjusted_sim, m_sim, c_,
                    f_initial_func=f_initial_sim, alpha=upwind_param, rktype=rk_method,
                    anim=animate_sim, save=save_anim, tend=T_final, plot_final=plot_final_sim
                )

                # Report result
                if u_final_sim is not None:
                    print(f"\nNormal Simulation Complete for n={n_sim}. Final L2 Error = {l2_err_sim:.6e}")
                else:
                    print(f"\nNormal Simulation FAILED for n={n_sim}.")
        else:
            print("Skipping normal simulation due to invalid initial condition.")


    # --- Configuration and Run Convergence Study ---
    if run_conv_study:
        # **** Use SQUARE WAVE for convergence study ****
        f_initial_conv = ic_square_wave
        # Example n values for convergence study
        n_values_conv = [5, 10, 20, 40, 80, 160]
        # Use a MORE CONSERVATIVE CFL for study to minimize stability/temporal error issues
        cfl_target_conv = 0.05 # Reduced CFL for convergence study accuracy

        print(f"\nConfiguring Convergence Study (IC: {f_initial_conv.__name__}, CFL: {cfl_target_conv})")
        # Run the study - function now handles asymptotic rate reporting and correct plotting
        h_conv, errors_conv = run_convergence_study(
            L_, c_, T_final, f_initial_conv, upwind_param, rk_method,
            n_values=n_values_conv,
            cfl_target=cfl_target_conv, # Use CFL-based dt
            dt_fixed=None
        )


    # --- Configuration and Run n-Comparison Plot ---
    if run_n_comparison_plot:
        # Choose the IC for the comparison plot
        # **** Use SQUARE WAVE for comparison plot ****
        # Check if normal sim ran with square wave, otherwise default to square wave
        if f_initial_sim is not None and callable(f_initial_sim) and initial_condition_name == 'square':
            f_initial_comp = f_initial_sim
            ic_name_comp = initial_condition_name # Get name from normal sim run
            print(f"\nUsing IC '{ic_name_comp}' from normal simulation for n-comparison plot.")
        else:
            # Fallback if normal sim didn't run or IC was different
            f_initial_comp = ic_square_wave # Default to square wave
            ic_name_comp = 'square_wave'
            print(f"\nUsing default IC '{ic_name_comp}' for n-comparison plot.")

        # Choose n values and CFL for the comparison plot
        n_values_comp = [5, 10, 20, 40] # Which n values to plot together
        cfl_target_comp = 0.1 # Use a reasonable CFL target

        print(f"\nConfiguring n-Comparison Plot (IC: {ic_name_comp}, CFL: {cfl_target_comp})")
        # Call the comparison plot function
        plot_comparison_n_lagrange(
            L_, c_, T_final, f_initial_comp, upwind_param, rk_method,
            n_values_to_compare=n_values_comp,
            cfl_target=cfl_target_comp
        )


    print("\n--- Script Finished ---")

# ==============================================================
# End of Script
# ==============================================================
