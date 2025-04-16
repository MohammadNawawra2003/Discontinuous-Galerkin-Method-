# dg_legendre_p1_1d_improved.py

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre, roots_legendre
from scipy.signal import square
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os
import math # For log in convergence rate calculation

# --- Provided table (likely CFL limits or related coefficients) ---
# Kept for reference, but CFL calculation below is more explicit.
table = [
    [1.0000, 1.0000, 1.2564, 1.3926, 1.6085], # p=0
    [0, 0.3333, 0.4096, 0.4642, 0.5348],    # p=1
    [0, 0, 0.2098, 0.2352, 0.2716],       # p=2
    # ... (rest of table omitted for brevity but could be included)
]

# --- Time Stepping Functions (used by core solver) ---
def rk44(u_history, Q, dt, m):
    """ Solves du/dt = Q u using classic RK44. u_history[0] is IC. """
    if Q is None:
        print("Error: Q matrix is None in rk44.")
        u_history[1:, :] = np.nan
        return u_history

    N_dof = u_history.shape[1]
    n_elements = N_dof // (Q.shape[0] // N_dof) # Infer n (assumes Q block structure size known implicitly)
                                                # This is a bit fragile, better to pass n if possible

    for i in tqdm(range(m), desc=f"RK44 Steps", unit="step"):
        u_i = u_history[i]
        try:
            K1 = Q.dot(u_i)
            K2 = Q.dot(u_i + K1 * dt / 2.)
            K3 = Q.dot(u_i + K2 * dt / 2.)
            K4 = Q.dot(u_i + K3 * dt)

            # Check for NaNs/Infs
            if not np.all(np.isfinite(K1)) or \
               not np.all(np.isfinite(K2)) or \
               not np.all(np.isfinite(K3)) or \
               not np.all(np.isfinite(K4)):
                print(f"\nWarning: Instability detected at time step {i+1}. Aborting RK step.")
                u_history[i + 1:, :] = np.nan
                return u_history # Stop integration

            u_history[i + 1] = u_i + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
        except Exception as e:
             print(f"\nError during RK step {i+1}: {e}")
             u_history[i + 1:, :] = np.nan # Mark as failed
             return u_history
    return u_history

# Other time steppers (fwd_euler, rk22) could be added here if needed


# --- DG Matrix Building Function ---
def build_Q_matrix(n, p, L, c, a):
    """ Builds the DG spatial operator matrix Q = -c/dx * M_inv * L. """
    if p < 0: raise ValueError("Polynomial degree p must be non-negative.")
    if n <= 0: raise ValueError("Number of elements n must be positive.")
    dx = L / n
    if dx <= 1e-15: raise ValueError(f"dx={dx} too small.")

    # Inverse Mass Matrix M_inv (Diagonal)
    # M_ii = integral( psi_i * psi_i * dx ) = (dx/2) * integral( P_i(xi)^2 dxi ) = (dx/2) * (2 / (2i+1)) = dx / (2i+1)
    # M_inv_ii = (2*i + 1) / dx
    inv_mass_diag_coeffs = np.arange(0, p + 1) # i = 0 to p
    inv_mass_matrix_diag_term = (2.0 * inv_mass_diag_coeffs + 1.0) # Element-local diagonal term
    inv_mass_matrix_diag = np.tile(inv_mass_matrix_diag_term, n)
    # Normalization factor from dx = jacobian * dxi = (dx/2) * 2
    # The projection used below includes (2i+1)/2 factor. Mass matrix M_ij = integral(Li Lj dx)
    # M_ij = delta_ij * (dx / (2i+1)). M_inv_ij = delta_ij * (2i+1) / dx
    # Let's use the formula from projection consistency: M_inv_ii = (2i+1)/2 * (2/dx) = (2i+1)/dx
    inv_mass_matrix_diag_global = (inv_mass_matrix_diag_term * (2.0/dx)) # Global diag term including 1/dx scaling
    # Wait, the projection scaling was (2i+1)/2. Mass matrix term is integral(Pi*Pi*dx) = (dx/2)*integral(Pi*Pi*dxi) = (dx/2)*(2/(2i+1))=dx/(2i+1).
    # M_inv is diag( (2i+1)/dx ).
    inv_mass_matrix_diag = np.tile(inv_mass_matrix_diag_term, n) * (1.0 / dx) # Correct scaling
    inv_mass_matrix = sp.diags([inv_mass_matrix_diag], [0], format='csr', dtype=np.float64)


    # Stiffness/Flux Matrix L (build block by block)
    if p == 0:
        # P0 basis: P0(xi)=1. dP0/dxi = 0.
        D = sp.csr_matrix((1, 1)) # No derivative term
        A = sp.csr_matrix([[1.0]]) # P0(1)P0(1)=1, P0(-1)P0(-1)=1, P0(1)P0(-1)=1, P0(-1)P0(1)=1
        B = sp.csr_matrix([[1.0]]) # P0(1)=1, P0(-1)=1 -> [[P0(1)], [P0(-1)]] -> B represents P_i(x)*P_j(boundary) -> P0(1)=1
        I = sp.csr_matrix([[1.0]]) # P0(1)P0(1)=1, P0(-1)P0(-1)=1
    else: # p >= 1
        # D matrix (derivative term contribution to L)
        # L_ij = - integral( P_i * dP_j/dx dx ) = - integral( P_i * dP_j/dxi dxi ) * (dxi/dx) * dx ??? No.
        # L_ij = integral( P_j * dP_i/dx dx ) -> from whiteboard derivation? Let's trust build_matrix structure.
        # integral( P_j * dP_i/dxi dxi ) -> D[i,j] in the original code seems to represent this.
        # D[i, j] = integral( Pi * dPj/dxi dxi ) is zero unless i > j and i+j is odd.
        # D[i, i-k] = 2 for k=1, 3, 5... <= i
        diags_D = []
        offsets_D = []
        for k_odd in range(1, p + 1, 2): # k = 1, 3, 5...
             diag_len = p + 1 - k_odd
             if diag_len > 0:
                 diags_D.append(2.0 * np.ones(diag_len))
                 offsets_D.append(-k_odd) # D[i, i-k]
        if not diags_D: # Handle p=0 case again, although caught above
             D = sp.csr_matrix((p + 1, p + 1))
        else:
             D = sp.diags(diags_D, offsets_D, shape=(p + 1, p + 1), format='csr')

        # A matrix: A[i,j] = P_i(1)P_j(1) - P_i(-1)P_j(-1)
        Pi_p1 = np.array([legendre(i)(1.0) for i in range(p+1)]) # All are 1.0
        Pi_m1 = np.array([legendre(i)(-1.0) for i in range(p+1)]) # (-1)^i
        A = np.outer(Pi_p1, Pi_p1) - np.outer(Pi_m1, Pi_m1) # A[i,j] = 1 - (-1)^i*(-1)^j = 1 - (-1)^(i+j)
        A = sp.csr_matrix(A)

        # B matrix: B[i,j] = P_i(1)P_j(1)
        B = np.outer(Pi_p1, Pi_p1) # B[i,j] = 1 * 1 = 1
        B = sp.csr_matrix(B)

        # I matrix: I[i,j] = P_i(-1)P_j(-1)
        I = np.outer(Pi_m1, Pi_m1) # I[i,j] = (-1)^i * (-1)^j = (-1)^(i+j)
        I = sp.csr_matrix(I)

        # Correction: Check original definition of A, B, I in Hesthaven/Warburton book or similar source.
        # From typical DG: Flux terms involve P_i(+-1).
        # Term: - [ v * (f*n) ]_bound -> - [ P_i * (c*u_hat*n) ]_bound
        # At x_right (xi=1, n=1): - P_i(1) * c * u_hat_right
        # At x_left (xi=-1, n=-1): + P_i(-1) * c * u_hat_left
        # Using upwind u_hat_right=u_right=u(xi=1), u_hat_left=u_left_neighbor(xi=1) if c>0
        # Need to express u_hat in terms of coefficients U_j of the neighboring element.
        # u_hat_right = Sum_j U_j_current * P_j(1)
        # u_hat_left = Sum_j U_j_neighbor * P_j(1)
        # Contribution to element k from right boundary flux: - c * P_i(1) * Sum_j U_j_k * P_j(1) = -c * A_tilde_ij * U_j_k   where A_tilde_ij = Pi(1)Pj(1)
        # Contribution to element k from left boundary flux: + c * P_i(-1) * Sum_j U_j_km1 * P_j(1)
        # Volume integral: integral( dP_i/dx * c * u dx ) = integral( dP_i/dxi * c * Sum(Uj Pj) dxi ) * (dxi/dx) * dx
        # = c * Sum_j Uj_k * integral( dP_i/dxi * Pj dxi )

        # Let's assume the original build_matrix structure was correct for its specific formulation:
        # L = Stiff Matrix such that M U' = -c/dx * L * U

        # Flux terms depend on upwind parameter 'a' (alpha)
        # mat_lft: contribution from left neighbor (periodic j=i-1 or j=n-1)
        # mat_rgt: contribution from right neighbor (periodic j=i+1 or j=0)
        # mat_ctr: diagonal block contribution
        # Revisit original A,B,I definitions if results are wrong. Using code's version for now:
        A_orig = np.ones((p + 1, p + 1)); A_orig[1::2, ::2] = -1.; A_orig[::2, 1::2] = -1. # Alternating +/- 1
        B_orig = np.ones((p + 1, p + 1)); B_orig[:, 1::2] = -1. # Alternating +/- 1 in columns
        I_orig = np.ones((p + 1, p + 1))

        A_orig = sp.csr_matrix(A_orig)
        B_orig = sp.csr_matrix(B_orig)
        I_orig = sp.csr_matrix(I_orig)

        # Terms based on upwind parameter 'a'
        # alpha = 1 means upwind for c>0 (flux comes from left state)
        # Need term multiplying U_{k-1} and term multiplying U_{k+1}
        # Using the logic from the code provided:
        mat_lft = -(1. + a) / 2. * B_orig.T # Coefficient for U_{k-1} term
        mat_rgt = +(1. - a) / 2. * B_orig  # Coefficient for U_{k+1} term
        mat_ctr = D.T + (1. + a) / 2. * A_orig - (1. - a) / 2. * I_orig # Coefficient for U_k


    # Assemble the global block matrix L
    L_stiff = sp.bmat([
        [(mat_lft if (j == i - 1) or (i == 0 and j == n - 1) else
          mat_ctr if j == i else
          mat_rgt if (j == i + 1) or (i == n - 1 and j == 0) else
          None) for j in range(n)]
        for i in range(n)
    ], format='bsr', dtype=np.float64) # Use BSR for efficiency if blocks are dense

    # Build Q = -c/dx * M_inv * L
    # Note the sign: dU/dt = Q*U. If U' = -c U_x, then Q involves -c.
    # If Q is defined such that U' = Q*U, then Q should be -c * (...)
    Q_mat = -c * inv_mass_matrix.dot(L_stiff) # Scaling by c is correct here
                                            # Scaling by 1/dx is handled in M_inv

    return Q_mat


# --- Initial Condition Functions ---
def ic_sine_wave(x, L):
    """ Smooth sine wave initial condition. """
    return np.sin(2 * np.pi * x / L)

def ic_square_wave(x, L):
    """ Discontinuous square wave initial condition. """
    # return square(2 * np.pi * x / L, duty=1./3.) # Original duty cycle
    return square(2 * np.pi * x / L, duty=0.5) # Standard 50% duty cycle square wave

# --- Initial Condition Projection (Legendre) ---
def compute_coefficients_legendre(f_initial_func, L, n, p):
    """ Computes initial DG Legendre coefficients by L2 projection. """
    num_quad = max(p + 1, 5) # Use p+1 Gauss points minimum, maybe more
    try:
        xi_quad, w_quad = roots_legendre(num_quad)
    except Exception as e:
         print(f"Error getting Legendre roots for num_quad={num_quad}: {e}")
         raise
    dx = L / n
    if dx <= 1e-15: raise ValueError(f"dx={dx} too small.")
    jacobian = dx / 2.0
    psi_basis = [legendre(i) for i in range(p + 1)] # Basis functions P_0, ..., P_p

    u_coeffs = np.zeros((n, p + 1), dtype=np.float64)
    for k in range(n): # Loop over elements
        x_left = k * dx
        x_quad_k = x_left + (xi_quad + 1.0) * jacobian # Map quad points to element k

        f_vals_at_quad = f_initial_func(x_quad_k, L) # Evaluate IC at physical quad points

        for i in range(p + 1): # Loop over basis functions P_i
            psi_i_vals_at_ref_quad = psi_basis[i](xi_quad) # Evaluate P_i at ref quad points
            # Projection formula: u_i = ( (2i+1)/2 ) * integral( f(x(xi)) * P_i(xi) dxi )
            # integral approx by sum_q w_q * f(x(xi_q)) * P_i(xi_q)
            integral_weighted = np.dot(w_quad, f_vals_at_quad * psi_i_vals_at_ref_quad)
            # Normalization factor (2i+1)/2 comes from orthogonality int(Pi*Pi dxi) = 2/(2i+1)
            u_coeffs[k, i] = ( (2.0 * i + 1.0) / 2.0 ) * integral_weighted

    return u_coeffs.reshape(n * (p + 1)) # Flatten


# --- Core Simulation Logic (No Plotting/Animation) - Legendre ---
def run_simulation_core_legendre(L, n, p, dt, m, c, f_initial_func, a, rktype='RK44'):
    """ Runs the Legendre DG simulation and returns the full coefficient history. """
    N_dof = n * (p + 1)
    u_history = np.zeros((m + 1, N_dof), dtype=np.float64)

    # Compute initial coefficients
    try:
        u_history[0] = compute_coefficients_legendre(f_initial_func, L, n, p)
    except Exception as e:
        print(f"Error computing initial coefficients for n={n}, p={p}: {e}")
        return None # Indicate failure

    # Build spatial operator matrix Q
    try:
        Q_mat = build_Q_matrix(n, p, L, c, a)
    except Exception as e:
        print(f"Error building Q matrix for n={n}, p={p}: {e}")
        return None # Indicate failure

    # Perform time stepping
    print(f"Starting time integration ({rktype}, n={n}, p={p})...")
    if rktype == 'RK44':
        u_history = rk44(u_history, Q_mat, dt, m)
    # Add other RK methods if needed (ensure they handle history array correctly)
    # elif rktype == 'RK22': u_history = rk22(u_history, Q_mat, dt, m)
    else:
        print(f"Error: Unsupported rktype '{rktype}' for Legendre core sim.")
        return None # Indicate failure
    print("Time integration finished.")

    # Return the full history (may contain NaNs if RK failed)
    return u_history


# --- Evaluate Legendre DG Solution ---
def evaluate_dg_solution_legendre(x_eval, coeffs_history_flat, L, n, p, time_step_index):
    """ Evaluates the Legendre DG solution at specific points x for a given time step. """
    N_dof = n * (p + 1)
    if time_step_index >= coeffs_history_flat.shape[0]:
        raise IndexError("time_step_index out of bounds for coeffs_history_flat")

    coeffs_flat_at_time = coeffs_history_flat[time_step_index]

    if coeffs_flat_at_time is None or not np.all(np.isfinite(coeffs_flat_at_time)):
        print(f"Warning: Evaluating Legendre DG solution with invalid coefficients at time step {time_step_index}.")
        return np.full_like(x_eval, np.nan, dtype=float)

    coeffs_element_wise = coeffs_flat_at_time.reshape((n, p + 1)).T # Shape (p+1, n)
    dx = L / n
    if dx <= 1e-15: raise ValueError("dx is too small.")
    u_h_eval = np.zeros_like(x_eval, dtype=float)
    psi_basis = [legendre(i) for i in range(p + 1)] # P_0 to P_p

    for i, x_val in enumerate(x_eval):
        # Determine element index k and local coordinate xi in [-1, 1]
        if x_val >= L: element_idx, xi_val = n - 1, 1.0
        elif x_val <= 0: element_idx, xi_val = 0, -1.0
        else:
            element_idx = int(np.floor(x_val / dx))
            element_idx = min(element_idx, n - 1)
            x_left = element_idx * dx
            xi_val = 2.0 * (x_val - x_left) / dx - 1.0
            xi_val = np.clip(xi_val, -1.0, 1.0)

        # Evaluate u_h(xi) = sum_j coeffs[j] * P_j(xi)
        psi_vals_at_xi = np.array([psi(xi_val) for psi in psi_basis]) # Shape (p+1,)
        coeffs_k = coeffs_element_wise[:, element_idx] # Shape (p+1,)
        u_h_eval[i] = np.dot(coeffs_k, psi_vals_at_xi)

    return u_h_eval

# --- L2 Error Calculation (Legendre) ---
def calculate_l2_error_legendre(coeffs_final_flat, f_initial_func, L, n, p, c, T_final):
    """ Calculates L2 error for Legendre DG solution at T_final. """
    print(f"Calculating L2 error (Legendre P{p}, n={n})...")

    if coeffs_final_flat is None or not np.all(np.isfinite(coeffs_final_flat)):
        print(f"Warning: Cannot calculate L2 error for n={n}, p={p} due to invalid final coefficients. Returning NaN.")
        return np.nan

    num_quad = p + 1 # Need 2p+1 degree exactness -> use p+1 points for Gauss-Legendre?
                     # Let's use more points for safety, e.g., 2p+1 points? Or just p+2?
    num_quad = max(p + 2, 5) # Ensure enough points, at least 5
    try:
        xi_quad, w_quad = roots_legendre(num_quad)
    except Exception as e:
        print(f"Error getting Legendre roots: {e}")
        return np.nan

    l2_error_sq_sum = 0.0
    dx = L / n
    if dx <= 1e-15:
        print(f"Warning: dx={dx} too small for L2 error calc (n={n}). Returning NaN.")
        return np.nan
    jacobian = dx / 2.0

    coeffs_element_wise = coeffs_final_flat.reshape((n, p + 1)).T # Shape (p+1, n)
    psi_basis = [legendre(i) for i in range(p + 1)]
    # Pre-evaluate basis functions at reference quadrature points
    psi_vals_at_ref_quad = np.array([psi(xi_quad) for psi in psi_basis]) # Shape (p+1, num_quad)

    u_exact_final_func = lambda x: u_exact(x, T_final, L, c, f_initial_func)

    for k in range(n):
        x_left = k * dx
        x_quad_k = x_left + (xi_quad + 1.0) * jacobian # Physical quad points in element k

        # Evaluate DG solution u_h at physical quadrature points using coefficients
        coeffs_k = coeffs_element_wise[:, k] # Coefficients for element k, shape (p+1,)
        # u_h(xi_q) = Sum_i coeffs_k[i] * P_i(xi_q)
        u_h_at_ref_quad = np.dot(coeffs_k, psi_vals_at_ref_quad) # Shape (num_quad,)

        # Evaluate exact solution u_ex at physical quadrature points
        u_ex_at_quad_k = u_exact_final_func(x_quad_k)

        # Calculate squared error at quadrature points
        error_sq_at_quad = (u_h_at_ref_quad - u_ex_at_quad_k)**2

        # Add contribution from element k to the total L2 error squared integral
        l2_error_sq_sum += np.sum(w_quad * error_sq_at_quad) * jacobian

    if l2_error_sq_sum < 0 or not np.isfinite(l2_error_sq_sum):
        print(f"Warning: Invalid L2 error sum ({l2_error_sq_sum}) before sqrt for n={n}, p={p}. Returning NaN.")
        return np.nan

    l2_error = np.sqrt(l2_error_sq_sum)
    print(f"L2 Error ||u_h - u_exact|| at T={T_final:.2f} (n={n}, p={p}) = {l2_error:.6e}")
    return l2_error

# --- Animation Function (Legendre) ---
def plot_function_legendre(u_coeffs_history_flat, L, n, p, dt, m, c, f_initial_func, save=False, tend=0.):
    """ Creates animation of the Legendre DG solution vs exact solution. """
    N_dof = n * (p + 1)
    n_plot_eval_per_elem = 20 # Points per element for smooth plotting
    x_plot_full = np.linspace(0., L, n * n_plot_eval_per_elem + 1)

    # Check for NaNs in history
    if np.any(np.isnan(u_coeffs_history_flat)):
        print("\nWarning: Simulation history contains NaNs. Animation may be incomplete.")
        first_nan_step = np.where(np.isnan(u_coeffs_history_flat))[0]
        m_plot = first_nan_step[0] if len(first_nan_step) > 0 else m
        print(f"Plotting animation frames up to step {m_plot}.")
    else:
        m_plot = m

    # Reconstruct solution u(x,t) from coefficients for plotting
    print("Reconstructing solution for animation...")
    v_plot = np.zeros((m_plot + 1, len(x_plot_full)))
    for time_idx in tqdm(range(m_plot + 1), desc="Reconstructing Frames"):
        try:
            v_plot[time_idx, :] = evaluate_dg_solution_legendre(
                x_plot_full, u_coeffs_history_flat, L, n, p, time_idx)
        except Exception as e:
            print(f"Error reconstructing frame {time_idx}: {e}. Skipping rest.")
            v_plot[time_idx:, :] = np.nan # Mark subsequent frames as invalid
            break # Stop reconstruction

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    ax.grid(True, linestyle=':')

    global ftSz1, ftSz2, ftSz3
    try: ftSz1
    except NameError: ftSz1, ftSz2, ftSz3 = 16, 14, 12
    plt.rcParams["text.usetex"] = False
    plt.rcParams['font.family'] = 'serif'

    time_template = r'$t = \mathtt{{{:.4f}}} \;[s]$'
    time_text = ax.text(0.75, 0.90, '', fontsize=ftSz1, transform=ax.transAxes)

    dg_line, = ax.plot([], [], color='b', lw=1.5, label=f'DG Solution (Legendre P{p}, n={n}, RK44)')
    exact_func_t = lambda x, t: u_exact(x, t, L, c, f_initial_func)
    exact, = ax.plot([], [], color='r', alpha=0.7, lw=3, zorder=0, label='Exact')
    initial_exact_y = exact_func_t(x_plot_full, 0)
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
        if t_idx < v_plot.shape[0] and np.all(np.isfinite(v_plot[t_idx, :])):
            current_time = t_idx * dt
            dg_line.set_data(x_plot_full, v_plot[t_idx, :])
            exact.set_ydata(exact_func_t(x_plot_full, current_time))
            time_text.set_text(time_template.format(current_time))
        return tuple([dg_line, exact, time_text])

    fps = 30
    num_frames_to_show = np.where(np.isnan(v_plot[:,0]))[0] # Find first NaN frame
    num_frames_to_show = num_frames_to_show[0] if len(num_frames_to_show) > 0 else m_plot + 1
    interval = max(1, int(1000.0 / fps))

    print("Creating animation...")
    anim = FuncAnimation(fig, animate, frames=num_frames_to_show, interval=interval, blit=False,
                         init_func=init, repeat=False)

    if save:
        output_dir = './figures'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        ic_name_str = f_initial_func.__name__.replace('ic_', '')
        anim_filename = os.path.join(output_dir, f"dg_advection_legendre_p{p}_n{n}_{ic_name_str}_RK44.mp4")
        print(f"Saving animation to {anim_filename}...")
        writerMP4 = FFMpegWriter(fps=fps)
        try: anim.save(anim_filename, writer=writerMP4); print("Animation saved.")
        except Exception as e: print(f"Error saving animation: {e}")
    else: plt.show()
    return anim


# --- Main DG Solver Function (Legendre Wrapper) ---
def advection1d_legendre_solver(L, n, p, dt, m, c, f_initial_func, a, rktype='RK44',
                               anim=True, save=False, tend=0., plot_final=True):
    """ Wrapper to run Legendre DG simulation and handle post-processing. """

    # Run the core simulation
    u_coeffs_history_flat = run_simulation_core_legendre(
        L, n, p, dt, m, c, f_initial_func, a, rktype)

    if u_coeffs_history_flat is None or np.any(np.isnan(u_coeffs_history_flat)):
        print(f"\n--- Simulation failed for n={n}, p={p} ---")
        return None, np.nan, None # Indicate failure

    u_coeffs_final_flat = u_coeffs_history_flat[m]

    # --- Post-processing ---
    animation_object = None
    if anim:
        animation_object = plot_function_legendre(
            u_coeffs_history_flat, L, n, p, dt, m, c, f_initial_func, save, tend)

    if plot_final and not anim:
        print("\nPlotting final solution comparison...")
        n_plot_points_per_element = 50
        x_plot = np.linspace(0, L, n * n_plot_points_per_element + 1)
        u_h_final = evaluate_dg_solution_legendre(
            x_plot, u_coeffs_history_flat, L, n, p, m) # Evaluate at final time m
        u_ex_final = u_exact(x_plot, tend, L, c, f_initial_func)

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, u_ex_final, 'r-', linewidth=3, alpha=0.7, label=f'Exact Solution at T={tend:.2f}')
        plt.plot(x_plot, u_h_final, 'b-', linewidth=1.5, label=f'DG Solution (Legendre P{p}, n={n}, RK44)')
        for k_elem in range(n + 1): plt.axvline(k_elem * L / n, color='gray', linestyle=':', linewidth=0.5)
        plt.xlabel("x", fontsize=ftSz2); plt.ylabel("u(x, T)", fontsize=ftSz2)
        plt.title(f"DG Legendre P{p} Solution vs Exact Solution at T={tend:.2f} (Final)", fontsize=ftSz1)
        plt.legend(fontsize=ftSz3); plt.grid(True, linestyle=':')
        ymin = min(u_ex_final.min(), u_h_final.min()) - 0.2
        ymax = max(u_ex_final.max(), u_h_final.max()) + 0.2
        plt.ylim(ymin, ymax)
        plt.show()

    # Calculate L2 error
    l2_error = calculate_l2_error_legendre(
        u_coeffs_final_flat, f_initial_func, L, n, p, c, tend)

    return u_coeffs_final_flat, l2_error, animation_object


# --- Convergence Study Function (Legendre) ---
def run_convergence_study_legendre(L, p, c, T_final, f_initial_func, upwind_param, rk_method, n_values, cfl_target=None, dt_fixed=None):
    """ Performs convergence study for Legendre DG. """
    print(f"\n--- Starting Convergence Study (Legendre P{p}) ---")
    if not callable(f_initial_func): raise TypeError("f_initial_func must be callable.")
    if cfl_target is None and dt_fixed is None: raise ValueError("Need cfl_target or dt_fixed.")
    if cfl_target is not None and dt_fixed is not None:
        print("Warning: Using cfl_target, ignoring dt_fixed.")
        dt_fixed = None

    l2_errors = []
    h_values = []

    for n_conv in n_values:
        print(f"\nRunning Convergence Study for n = {n_conv} (p={p})")
        dx_conv = L / n_conv
        if dx_conv <= 1e-15:
             print(f"Warning: dx={dx_conv} too small for n={n_conv}. Skipping."); l2_errors.append(np.nan)
             h_values.append(dx_conv if dx_conv > 0 else np.nan); continue
        h_values.append(dx_conv)

        # Determine dt and m
        dt_conv = 0.0
        if dt_fixed is not None:
            if dt_fixed <= 0: raise ValueError("dt_fixed must be positive.")
            dt_conv = dt_fixed
        else: # Use CFL
            if cfl_target is None or cfl_target <= 0: raise ValueError("cfl_target must be positive.")
            # CFL for DG Legendre P(p): dt <= C * dx / (c * (2p+1)) approx?
            # A simple scaling dt ~ dx/c often works for basic checks with safety factor
            if abs(c) > 1e-12:
                # The factor (2p+1) is sometimes used as a heuristic stability limit scaling
                # dt_conv = cfl_target * dx_conv / (abs(c) * (2.0*p + 1.0)) # More conservative heuristic
                dt_conv = cfl_target * dx_conv / abs(c) # Simpler scaling
            else: dt_conv = T_final / 100.0
            if dt_conv <= 1e-15:
                 print(f"Warning: dt={dt_conv} too small for n={n_conv}. Skipping."); l2_errors.append(np.nan); continue

        m_conv = max(1, int(np.ceil(T_final / dt_conv)))
        dt_adjusted_conv = T_final / m_conv
        actual_cfl_simple = abs(c) * dt_adjusted_conv / dx_conv if abs(dx_conv) > 1e-12 and abs(c) > 1e-12 else 0
        # actual_cfl_p_scaled = actual_cfl_simple * (2.0*p + 1.0) # Heuristic scaling

        print(f"  dx = {dx_conv:.4e}")
        print(f"  m = {m_conv}, dt = {dt_adjusted_conv:.4e}, Simple CFL = {actual_cfl_simple:.3f}")

        # Run simulation core
        u_coeffs_history = run_simulation_core_legendre(
            L, n_conv, p, dt_adjusted_conv, m_conv, c,
            f_initial_func, upwind_param, rk_method)

        # Check failure and calculate L2 error
        if u_coeffs_history is None or np.any(np.isnan(u_coeffs_history)):
            print(f"Simulation failed for n={n_conv}, cannot calculate L2 error.")
            l2_errors.append(np.nan)
        else:
            u_coeffs_final = u_coeffs_history[m_conv]
            l2_err_n = calculate_l2_error_legendre(
                u_coeffs_final, f_initial_func, L, n_conv, p, c, T_final)
            l2_errors.append(l2_err_n)

    # --- Plotting Convergence Results ---
    h_values = np.array(h_values)
    l2_errors = np.array(l2_errors)

    valid_mask = np.isfinite(h_values) & np.isfinite(l2_errors) & (l2_errors > 1e-15)
    h_valid = h_values[valid_mask]
    l2_errors_valid = l2_errors[valid_mask]

    rates = []
    if len(h_valid) > 1:
        log_errors = np.log(l2_errors_valid)
        log_h = np.log(h_valid)
        sort_indices = np.argsort(h_valid)[::-1] # Sort h descending
        h_sorted = h_valid[sort_indices]
        log_errors_sorted = log_errors[sort_indices]
        log_h_sorted = log_h[sort_indices]
        rates = (log_errors_sorted[:-1] - log_errors_sorted[1:]) / (log_h_sorted[:-1] - log_h_sorted[1:])

    print(f"\n--- Convergence Study Results (Legendre P{p}) ---")
    print("  n    |    h       |   L2 Error   | Approx. Rate")
    print("-------|------------|--------------|--------------")
    n_values_valid = [n for n, h in zip(n_values, h_values) if h in h_valid]
    n_print_order = [n_values_valid[i] for i in sort_indices]
    h_print_order = h_sorted
    l2_print_order = np.exp(log_errors_sorted)

    order_expected = p + 1 # Theoretical L2 convergence rate

    if len(n_print_order) > 0:
        print(f"{n_print_order[0]:>6d} | {h_print_order[0]:.6f} | {l2_print_order[0]:.6e} |     -    ")
        for i in range(len(rates)):
             print(f"{n_print_order[i+1]:>6d} | {h_print_order[i+1]:.6f} | {l2_print_order[i+1]:.6e} |   {rates[i]:.3f}  ")
    else: print("No valid points found for convergence analysis.")
    print("---------------------------------")
    if len(rates) > 0: print(f"Average Observed Rate: {np.mean(rates):.3f}")
    print(f"(Expected rate for P{p} elements is ~{order_expected:.1f} for smooth solutions)")

    plt.figure(figsize=(8, 6))
    plt.loglog(h_valid, l2_errors_valid, 'bo-', markerfacecolor='none', label=f'L2 Error (P{p})')
    if len(h_valid) > 0:
        C_ref = l2_errors_valid[0] / (h_valid[0]**order_expected)
        h_plot_ref = np.sort(h_valid)
        plt.loglog(h_plot_ref, C_ref * h_plot_ref**order_expected,
                   'r--', label=f'$\\mathcal{{O}}(h^{order_expected})$ Ref.')

    plt.xlabel("Element Size $h = L/n$", fontsize=ftSz2)
    plt.ylabel("$L_2$ Error at $T_{final}$", fontsize=ftSz2)
    plt.title(f"DG Legendre P{p} Convergence (IC: {f_initial_func.__name__})", fontsize=ftSz1)
    plt.gca().invert_xaxis(); plt.grid(True, which='both', linestyle=':')
    plt.legend(fontsize=ftSz3); plt.show()


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
# --- Main Execution Block (Legendre Version) ---
# ==============================================================
if __name__ == "__main__":

    # --- Select Mode ---
    run_normal_simulation = True
    run_conv_study = True

    # --- Configuration (Shared) ---
    L_ = 1.0
    c_ = 1.0
    T_final = 1.0
    p_ = 1           # << SET POLYNOMIAL DEGREE HERE >>
    rk_method = 'RK44'
    upwind_param = 1.0 # 1.0 for upwind (c>0)

    # --- Configuration for Normal Simulation ---
    if run_normal_simulation:
        n_sim = 40   # Number of elements
        initial_condition_name = 'sine' # 'sine' or 'square'

        # --- Time Stepping ---
        # Simple CFL scaling: dt ~ safety * dx / c
        safety_factor_sim = 0.1 # Be conservative, especially for higher p
        dx_sim = L_ / n_sim
        dt_cfl_sim = 0.0
        if abs(c_) > 1e-12:
            # Heuristic: dt ~ dx / (c * (2p+1)) -> add (2p+1) scaling?
            # dt_cfl_sim = safety_factor_sim * dx_sim / (abs(c_) * (2.0*p_ + 1.0))
            dt_cfl_sim = safety_factor_sim * dx_sim / abs(c_) # Simpler scaling
        else: dt_cfl_sim = T_final / 100.0
        if dt_cfl_sim <= 1e-15: raise ValueError(f"dt={dt_cfl_sim} too small.")

        m_sim = max(1, int(np.ceil(T_final / dt_cfl_sim)))
        dt_adjusted_sim = T_final / m_sim
        actual_cfl_sim = abs(c_) * dt_adjusted_sim / dx_sim if abs(dx_sim) > 1e-12 and abs(c_) > 1e-12 else 0

        print(f"\n--- Running Normal Simulation (Legendre P{p_}) ---")
        print(f"  n = {n_sim}, T_final = {T_final:.3f}, c = {c_}")
        print(f"  IC: {initial_condition_name}, Upwind alpha: {upwind_param}")
        print(f"  Safety Factor = {safety_factor_sim}, Simple CFL = {actual_cfl_sim:.3f}")
        print(f"  m = {m_sim}, dt = {dt_adjusted_sim:.6f}")

        if initial_condition_name == 'sine': f_initial_sim = ic_sine_wave
        elif initial_condition_name == 'square': f_initial_sim = ic_square_wave
        else: raise ValueError(f"Unknown IC name: {initial_condition_name}")

        # Run solver
        coeffs_final_sim, l2_err_sim, anim_obj = advection1d_legendre_solver(
            L_, n_sim, p_, dt_adjusted_sim, m_sim, c_,
            f_initial_func=f_initial_sim, a=upwind_param, rktype=rk_method,
            anim=True, save=False, tend=T_final, plot_final=True
        )
        if coeffs_final_sim is not None:
            print(f"\nSimulation Complete (P{p_}, n={n_sim}). Final L2 Error = {l2_err_sim:.6e}")
        else: print(f"\nSimulation FAILED (P{p_}, n={n_sim}).")


    # --- Configuration for Convergence Study ---
    if run_conv_study:
        f_initial_conv = ic_sine_wave # MUST use smooth IC
        n_values_conv = [5, 10, 20, 40, 80] # As per notes
        # Use a MORE CONSERVATIVE CFL for study
        safety_factor_conv = 0.05 # Even smaller safety factor
        cfl_target_conv = safety_factor_conv # Use safety factor directly as CFL target

        run_convergence_study_legendre(
            L_, p_, c_, T_final, f_initial_conv, upwind_param, rk_method,
            n_values=n_values_conv,
            cfl_target=cfl_target_conv, # Use CFL-based dt
            dt_fixed=None
        )


# ==============================================================
# End of Script
# ==============================================================
