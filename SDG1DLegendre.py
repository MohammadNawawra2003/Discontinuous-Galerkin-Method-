# dg_legendre_p1_1d_improved_v2.py
# Legendre DG code with IC selection, convergence study, refactoring,
# and a final plot comparing solutions for different 'n'.

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre, roots_legendre
from scipy.signal import square
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os
import math

# --- Time Stepping Functions (used by core solver) ---
def rk44(u_history, Q, dt, m):
    """ Solves du/dt = Q u using classic RK44. u_history[0] is IC. """
    if Q is None:
        print("Error: Q matrix is None in rk44.")
        u_history[1:, :] = np.nan
        return u_history

    N_dof = u_history.shape[1]
    p_approx = (Q.shape[0] // (u_history.shape[1] // (p_ + 1))) -1 # Infer p (Approximate)
    n_elements = N_dof // (p_approx + 1) # Infer n

    for i in tqdm(range(m), desc=f"RK44 Steps (n={n_elements}, p={p_approx})", unit="step", leave=False):
        u_i = u_history[i]
        try:
            K1 = Q.dot(u_i)
            K2 = Q.dot(u_i + K1 * dt / 2.)
            K3 = Q.dot(u_i + K2 * dt / 2.)
            K4 = Q.dot(u_i + K3 * dt)

            if not np.all(np.isfinite(K1)) or \
               not np.all(np.isfinite(K2)) or \
               not np.all(np.isfinite(K3)) or \
               not np.all(np.isfinite(K4)):
                print(f"\nWarning: Instability detected at time step {i+1} (n={n_elements}). Aborting.")
                u_history[i + 1:, :] = np.nan
                return u_history

            u_history[i + 1] = u_i + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
        except Exception as e:
             print(f"\nError during RK step {i+1} (n={n_elements}): {e}")
             u_history[i + 1:, :] = np.nan
             return u_history
    return u_history

# --- DG Matrix Building Function ---
def build_Q_matrix(n, p, L, c, a):
    """ Builds the DG spatial operator matrix Q = M_inv * L_spatial_operator. """
    # Note: Changed sign convention interpretation. L_spatial represents the spatial
    # operator such that M U' = L_spatial U. Then Q = M_inv * L_spatial.
    # Let's stick to the original code's implicit definition where Q = -c * M_inv * L_stiff
    # to maintain its behavior, assuming L_stiff was defined appropriately for that.

    if p < 0: raise ValueError("Polynomial degree p must be non-negative.")
    if n <= 0: raise ValueError("Number of elements n must be positive.")
    dx = L / n
    if dx <= 1e-15: raise ValueError(f"dx={dx} too small.")

    # Inverse Mass Matrix M_inv (Diagonal)
    # M_ij = delta_ij * (dx / (2i+1)) => M_inv_ii = (2i+1) / dx
    inv_mass_diag_coeffs = np.arange(0, p + 1) # i = 0 to p
    inv_mass_matrix_diag_term = (2.0 * inv_mass_diag_coeffs + 1.0) / dx # Scaled diagonal term
    inv_mass_matrix_diag_global = np.tile(inv_mass_matrix_diag_term, n)
    inv_mass_matrix = sp.diags([inv_mass_matrix_diag_global], [0], format='csr', dtype=np.float64)

    # Stiffness/Flux Matrix L_stiff (using original code's block structure)
    if p == 0:
        D = sp.csr_matrix((1, 1)); A = sp.csr_matrix([[1.]]); B = sp.csr_matrix([[1.]]); I = sp.csr_matrix([[1.]])
    else:
        diags_D = []; offsets_D = []
        for k_odd in range(1, p + 1, 2):
             diag_len = p + 1 - k_odd
             if diag_len > 0: diags_D.append(2.0 * np.ones(diag_len)); offsets_D.append(-k_odd)
        if not diags_D: D = sp.csr_matrix((p + 1, p + 1))
        else: D = sp.diags(diags_D, offsets_D, shape=(p + 1, p + 1), format='csr')

        # Using the A, B, I matrices as defined in the original code snippet provided
        A_orig = np.ones((p + 1, p + 1)); A_orig[1::2, ::2] = -1.; A_orig[::2, 1::2] = -1.
        B_orig = np.ones((p + 1, p + 1)); B_orig[:, 1::2] = -1.
        I_orig = np.ones((p + 1, p + 1))
        A = sp.csr_matrix(A_orig); B = sp.csr_matrix(B_orig); I = sp.csr_matrix(I_orig)

    # Assemble blocks based on original code logic
    mat_lft = -(1. + a) / 2. * B.T
    mat_rgt = +(1. - a) / 2. * B
    # Original D.T seems related to integral( dPi/dx * Pj dx ) term after integration by parts
    mat_ctr = D.T + (1. + a) / 2. * A - (1. - a) / 2. * I

    L_stiff = sp.bmat([
        [(mat_lft if (j == i - 1) or (i == 0 and j == n - 1) else
          mat_ctr if j == i else
          mat_rgt if (j == i + 1) or (i == n - 1 and j == 0) else
          None) for j in range(n)]
        for i in range(n)
    ], format='bsr', dtype=np.float64)

    # Build Q = -c * M_inv * L_stiff (Following original code's structure)
    # Note: The scaling by 1/dx was absorbed into M_inv definition
    Q_mat = -c * inv_mass_matrix.dot(L_stiff)

    return Q_mat


# --- Initial Condition Functions ---
def ic_sine_wave(x, L):
    """ Smooth sine wave initial condition. """
    return np.sin(2 * np.pi * x / L)

def ic_square_wave(x, L):
    """ Discontinuous square wave initial condition. """
    return square(2 * np.pi * x / L, duty=0.5)

# --- Initial Condition Projection (Legendre) ---
def compute_coefficients_legendre(f_initial_func, L, n, p):
    """ Computes initial DG Legendre coefficients by L2 projection. """
    num_quad = max(p + 2, 5) # Use sufficient quadrature points
    try: xi_quad, w_quad = roots_legendre(num_quad)
    except Exception as e: print(f"Error Legendre roots: {e}"); raise
    dx = L / n
    if dx <= 1e-15: raise ValueError(f"dx={dx} too small.")
    jacobian = dx / 2.0
    psi_basis = [legendre(i) for i in range(p + 1)]

    u_coeffs = np.zeros((n, p + 1), dtype=np.float64)
    for k in range(n):
        x_left = k * dx
        x_quad_k = x_left + (xi_quad + 1.0) * jacobian
        f_vals_at_quad = f_initial_func(x_quad_k, L)
        for i in range(p + 1):
            psi_i_vals_at_ref_quad = psi_basis[i](xi_quad)
            integral_weighted = np.dot(w_quad, f_vals_at_quad * psi_i_vals_at_ref_quad)
            u_coeffs[k, i] = ( (2.0 * i + 1.0) / 2.0 ) * integral_weighted # Normalization
    return u_coeffs.reshape(n * (p + 1))


# --- Core Simulation Logic (No Plotting/Animation) - Legendre ---
def run_simulation_core_legendre(L, n, p, dt, m, c, f_initial_func, a, rktype='RK44'):
    """ Runs the Legendre DG simulation and returns the full coefficient history. """
    N_dof = n * (p + 1)
    u_history = np.zeros((m + 1, N_dof), dtype=np.float64)

    try: u_history[0] = compute_coefficients_legendre(f_initial_func, L, n, p)
    except Exception as e: print(f"Error IC proj (n={n},p={p}): {e}"); return None

    try: Q_mat = build_Q_matrix(n, p, L, c, a)
    except Exception as e: print(f"Error build Q (n={n},p={p}): {e}"); return None

    print(f"Starting time integration ({rktype}, n={n}, p={p})...", end=' ', flush=True)
    if rktype == 'RK44': u_history = rk44(u_history, Q_mat, dt, m)
    else: print(f"Error: Unsupported rktype '{rktype}'"); return None
    print("Integration finished.")
    return u_history


# --- Evaluate Legendre DG Solution ---
def evaluate_dg_solution_legendre(x_eval, coeffs_history_flat, L, n, p, time_step_index):
    """ Evaluates the Legendre DG solution from coefficient history at a time index. """
    N_dof = n * (p + 1)
    if time_step_index >= coeffs_history_flat.shape[0]: raise IndexError("time_step_index OOB")

    coeffs_flat_at_time = coeffs_history_flat[time_step_index]
    if not np.all(np.isfinite(coeffs_flat_at_time)):
        print(f"Warn: Evaluating Legendre DG w/ invalid coeffs @ t_idx={time_step_index}.")
        return np.full_like(x_eval, np.nan, dtype=float)

    coeffs_element_wise = coeffs_flat_at_time.reshape((n, p + 1)).T # Shape (p+1, n)
    dx = L / n
    if dx <= 1e-15: raise ValueError("dx too small.")
    u_h_eval = np.zeros_like(x_eval, dtype=float)
    psi_basis = [legendre(i) for i in range(p + 1)]

    for i, x_val in enumerate(x_eval):
        if x_val >= L: element_idx, xi_val = n - 1, 1.0
        elif x_val <= 0: element_idx, xi_val = 0, -1.0
        else:
            element_idx = min(int(np.floor(x_val / dx)), n - 1)
            x_left = element_idx * dx
            xi_val = np.clip(2.0 * (x_val - x_left) / dx - 1.0, -1.0, 1.0)

        psi_vals_at_xi = np.array([psi(xi_val) for psi in psi_basis])
        coeffs_k = coeffs_element_wise[:, element_idx]
        u_h_eval[i] = np.dot(coeffs_k, psi_vals_at_xi)
    return u_h_eval


# --- L2 Error Calculation (Legendre) ---
def calculate_l2_error_legendre(coeffs_final_flat, f_initial_func, L, n, p, c, T_final):
    """ Calculates L2 error for Legendre DG solution at T_final. """
    print(f"Calculating L2 error (Legendre P{p}, n={n})...", end=' ', flush=True)
    if coeffs_final_flat is None or not np.all(np.isfinite(coeffs_final_flat)):
        print(f"Warn: Invalid final coeffs. Returning NaN."); return np.nan

    num_quad = max(p + 2, 5);
    try: xi_quad, w_quad = roots_legendre(num_quad)
    except Exception as e: print(f"Error Legendre roots: {e}"); return np.nan

    l2_error_sq_sum = 0.0; dx = L / n
    if dx <= 1e-15: print(f"Warn: dx={dx} too small (n={n}). NaN."); return np.nan
    jacobian = dx / 2.0

    coeffs_element_wise = coeffs_final_flat.reshape((n, p + 1)).T
    psi_basis = [legendre(i) for i in range(p + 1)]
    psi_vals_at_ref_quad = np.array([psi(xi_quad) for psi in psi_basis])

    u_exact_final_func = lambda x: u_exact(x, T_final, L, c, f_initial_func)

    for k in range(n):
        x_left = k * dx
        x_quad_k = x_left + (xi_quad + 1.0) * jacobian
        coeffs_k = coeffs_element_wise[:, k]
        u_h_at_ref_quad = np.dot(coeffs_k, psi_vals_at_ref_quad)
        u_ex_at_quad_k = u_exact_final_func(x_quad_k)
        error_sq_at_quad = (u_h_at_ref_quad - u_ex_at_quad_k)**2
        l2_error_sq_sum += np.sum(w_quad * error_sq_at_quad) * jacobian

    if l2_error_sq_sum < 0 or not np.isfinite(l2_error_sq_sum):
        print(f"Warn: Invalid L2 sum {l2_error_sq_sum} (n={n},p={p}). NaN."); return np.nan

    l2_error = np.sqrt(l2_error_sq_sum)
    print(f"L2 Error= {l2_error:.6e}")
    return l2_error


# --- Animation Function (Legendre) ---
def plot_function_legendre(u_coeffs_history_flat, L, n, p, dt, m, c, f_initial_func, save=False, tend=0.):
    """ Creates animation of the Legendre DG solution vs exact solution. """
    N_dof = n * (p + 1)
    n_plot_eval_per_elem = 20
    x_plot_full = np.linspace(0., L, n * n_plot_eval_per_elem + 1)

    if np.any(np.isnan(u_coeffs_history_flat)):
        print("\nWarn: History contains NaNs. Animation may be incomplete.")
        first_nan_step = np.where(np.isnan(u_coeffs_history_flat))[0]
        m_plot = first_nan_step[0] if len(first_nan_step) > 0 else m
        print(f"Plotting up to step {m_plot}.")
    else: m_plot = m

    print("Reconstructing solution for animation...")
    v_plot = np.zeros((m_plot + 1, len(x_plot_full)))
    for time_idx in tqdm(range(m_plot + 1), desc="Reconstructing Frames", leave=False):
        try: v_plot[time_idx, :] = evaluate_dg_solution_legendre(x_plot_full, u_coeffs_history_flat, L, n, p, time_idx)
        except Exception as e: print(f"Error reconstr. frame {time_idx}: {e}. Skip."); v_plot[time_idx:, :] = np.nan; break

    fig, ax = plt.subplots(1, 1, figsize=(10, 6)); fig.tight_layout(pad=3.0); ax.grid(True, linestyle=':')

    global ftSz1, ftSz2, ftSz3
    try: ftSz1
    except NameError: ftSz1, ftSz2, ftSz3 = 16, 14, 12
    plt.rcParams["text.usetex"] = False; plt.rcParams['font.family'] = 'serif'

    time_template=r'$t = \mathtt{{{:.4f}}} \;[s]$'; time_text = ax.text(0.75, 0.90, '', fontsize=ftSz1, transform=ax.transAxes)
    dg_line, = ax.plot([], [], color='b', lw=1.5, label=f'DG (Legendre P{p}, n={n})')
    exact_func_t = lambda x, t: u_exact(x, t, L, c, f_initial_func)
    exact, = ax.plot([], [], color='r', alpha=0.7, lw=3, zorder=0, label='Exact')
    initial_exact_y = exact_func_t(x_plot_full, 0)
    ymin = min(initial_exact_y) - 0.3; ymax = max(initial_exact_y) + 0.3
    ax.set_ylim(ymin, ymax); ax.set_xlim(0, L)
    ax.set_xlabel(r"$x$", fontsize=ftSz2); ax.set_ylabel(r"$u(x,t)$", fontsize=ftSz2); ax.legend(fontsize=ftSz3)

    def init():
        dg_line.set_data([], []); exact.set_data(x_plot_full, initial_exact_y); time_text.set_text(time_template.format(0))
        return tuple([dg_line, exact, time_text])

    def animate(t_idx):
        if t_idx < v_plot.shape[0] and np.all(np.isfinite(v_plot[t_idx, :])):
            current_time = t_idx * dt
            dg_line.set_data(x_plot_full, v_plot[t_idx, :]); exact.set_ydata(exact_func_t(x_plot_full, current_time))
            time_text.set_text(time_template.format(current_time))
        return tuple([dg_line, exact, time_text])

    fps = 30
    num_frames_to_show = np.where(np.isnan(v_plot[:,0]))[0]
    num_frames_to_show = num_frames_to_show[0] if len(num_frames_to_show) > 0 else m_plot + 1
    interval = max(1, int(1000.0 / fps))

    print("Creating animation...")
    anim = FuncAnimation(fig, animate, frames=num_frames_to_show, interval=interval, blit=False, init_func=init, repeat=False)

    if save:
        output_dir = './figures'; os.makedirs(output_dir, exist_ok=True)
        ic_name_str = f_initial_func.__name__.replace('ic_', '')
        anim_filename = os.path.join(output_dir, f"dg_advection_legendre_p{p}_n{n}_{ic_name_str}_RK44.mp4")
        print(f"Saving animation to {anim_filename}...")
        try: anim.save(anim_filename, writer=FFMpegWriter(fps=fps)); print("Saved.")
        except Exception as e: print(f"Error saving animation: {e}"); plt.show()
    else: plt.show()
    return anim


# --- Main DG Solver Function (Legendre Wrapper) ---
def advection1d_legendre_solver(L, n, p, dt, m, c, f_initial_func, a, rktype='RK44',
                               anim=True, save=False, tend=0., plot_final=True):
    """ Wrapper to run Legendre DG simulation and handle post-processing. """
    u_coeffs_history_flat = run_simulation_core_legendre(L, n, p, dt, m, c, f_initial_func, a, rktype)

    if u_coeffs_history_flat is None or np.any(np.isnan(u_coeffs_history_flat)):
        print(f"\n--- Simulation failed (P{p}, n={n}) ---")
        return None, np.nan, None

    u_coeffs_final_flat = u_coeffs_history_flat[m]
    animation_object = None
    if anim: animation_object = plot_function_legendre(u_coeffs_history_flat, L, n, p, dt, m, c, f_initial_func, save, tend)

    if plot_final and (not anim or np.any(np.isnan(u_coeffs_history_flat))): # Plot if no anim or if anim failed
        print("\nPlotting final solution comparison...")
        n_plot_points_per_element = 50
        x_plot = np.linspace(0, L, n * n_plot_points_per_element + 1)
        u_h_final = evaluate_dg_solution_legendre(x_plot, u_coeffs_history_flat, L, n, p, m)
        u_ex_final = u_exact(x_plot, tend, L, c, f_initial_func)

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, u_ex_final, 'r-', linewidth=3, alpha=0.7, label=f'Exact Sol. T={tend:.2f}')
        # Only plot DG if it's finite
        if np.all(np.isfinite(u_h_final)):
             plt.plot(x_plot, u_h_final, 'b-', linewidth=1.5, label=f'DG (Legendre P{p}, n={n})')
        else:
             plt.plot([], [], 'b-', label=f'DG (Legendre P{p}, n={n}) - FAILED') # Placeholder for legend

        for k_elem in range(n + 1): plt.axvline(k_elem * L / n, color='gray', linestyle=':', linewidth=0.5)
        plt.xlabel("x", fontsize=ftSz2); plt.ylabel("u(x, T)", fontsize=ftSz2)
        plt.title(f"DG Legendre P{p} Solution vs Exact at T={tend:.2f}", fontsize=ftSz1)
        plt.legend(fontsize=ftSz3); plt.grid(True, linestyle=':')
        ymin = min(u_ex_final.min(), np.nanmin(u_h_final) if not np.all(np.isnan(u_h_final)) else -1.5) - 0.2
        ymax = max(u_ex_final.max(), np.nanmax(u_h_final) if not np.all(np.isnan(u_h_final)) else 1.5) + 0.2
        plt.ylim(ymin, ymax); plt.show()

    l2_error = calculate_l2_error_legendre(u_coeffs_final_flat, f_initial_func, L, n, p, c, tend)
    return u_coeffs_final_flat, l2_error, animation_object


# --- Convergence Study Function (Legendre) ---
def run_convergence_study_legendre(L, p, c, T_final, f_initial_func, upwind_param, rk_method, n_values, cfl_target=None, dt_fixed=None):
    """ Performs convergence study for Legendre DG. """
    print(f"\n--- Starting Convergence Study (Legendre P{p}) ---")
    if not callable(f_initial_func): raise TypeError("f_initial_func must be callable.")
    if cfl_target is None and dt_fixed is None: raise ValueError("Need cfl_target or dt_fixed.")
    if cfl_target is not None and dt_fixed is not None: print("Warn: Using cfl_target, ignore dt_fixed."); dt_fixed = None

    l2_errors = []; h_values = []
    order_expected = p + 1

    for n_conv in n_values:
        print(f"\nRun Conv Study: n={n_conv}, p={p}")
        dx_conv = L / n_conv
        if dx_conv <= 1e-15: print(f"Warn: dx={dx_conv} skip."); l2_errors.append(np.nan); h_values.append(dx_conv if dx_conv > 0 else np.nan); continue
        h_values.append(dx_conv)

        dt_conv = 0.0
        if dt_fixed is not None: dt_conv = dt_fixed
        else:
            if cfl_target is None or cfl_target <= 0: raise ValueError("cfl_target must be > 0.")
            if abs(c) > 1e-12: dt_conv = cfl_target * dx_conv / abs(c) # Simple scaling sufficient w/ safety factor
            else: dt_conv = T_final / 100.0
            if dt_conv <= 1e-15: print(f"Warn: dt={dt_conv} skip."); l2_errors.append(np.nan); continue

        m_conv = max(1, int(np.ceil(T_final / dt_conv)))
        dt_adjusted_conv = T_final / m_conv
        actual_cfl_simple = abs(c) * dt_adjusted_conv / dx_conv if abs(dx_conv) > 1e-12 and abs(c) > 1e-12 else 0
        print(f"  dx={dx_conv:.3e}, m={m_conv}, dt={dt_adjusted_conv:.3e}, CFL_simple={actual_cfl_simple:.3f}")

        u_coeffs_history = run_simulation_core_legendre(L, n_conv, p, dt_adjusted_conv, m_conv, c, f_initial_func, upwind_param, rk_method)

        if u_coeffs_history is None or np.any(np.isnan(u_coeffs_history)): print(f"Sim failed n={n_conv}."); l2_errors.append(np.nan)
        else: l2_errors.append(calculate_l2_error_legendre(u_coeffs_history[m_conv], f_initial_func, L, n_conv, p, c, T_final))

    h_values = np.array(h_values); l2_errors = np.array(l2_errors)
    valid_mask = np.isfinite(h_values) & np.isfinite(l2_errors) & (l2_errors > 1e-15)
    h_valid = h_values[valid_mask]; l2_errors_valid = l2_errors[valid_mask]

    rates = []
    if len(h_valid) > 1:
        sort_indices = np.argsort(h_valid)[::-1]; h_sorted = h_valid[sort_indices]; log_errors_sorted = np.log(l2_errors_valid[sort_indices]); log_h_sorted = np.log(h_sorted)
        rates = (log_errors_sorted[:-1] - log_errors_sorted[1:]) / (log_h_sorted[:-1] - log_h_sorted[1:])

    print(f"\n--- Convergence Study Results (Legendre P{p}) ---")
    print("  n    |    h       |   L2 Error   | Approx. Rate")
    print("-------|------------|--------------|--------------")
    n_values_valid = [n for n, h in zip(n_values, h_values) if h in h_valid]
    n_print_order = [n_values_valid[i] for i in sort_indices]
    h_print_order = h_sorted; l2_print_order = np.exp(log_errors_sorted)

    if len(n_print_order) > 0:
        print(f"{int(n_print_order[0]):>6d} | {h_print_order[0]:.6f} | {l2_print_order[0]:.6e} |     -    ")
        for i in range(len(rates)): print(f"{int(n_print_order[i+1]):>6d} | {h_print_order[i+1]:.6f} | {l2_print_order[i+1]:.6e} |   {rates[i]:.3f}  ")
    else: print("No valid points found for convergence analysis.")
    print("---------------------------------")
    if len(rates) > 0: print(f"Average Observed Rate: {np.mean(rates):.3f}")
    print(f"(Expected rate for P{p} is ~{order_expected:.1f} for smooth solutions)")

    plt.figure(figsize=(8, 6))
    plt.loglog(h_valid, l2_errors_valid, 'bo-', markerfacecolor='none', label=f'L2 Error (P{p})')
    if len(h_valid) > 0:
        C_ref = l2_errors_valid[0] / (h_valid[0]**order_expected)
        h_plot_ref = np.sort(h_valid)
        plt.loglog(h_plot_ref, C_ref * h_plot_ref**order_expected, 'r--', label=f'$\\mathcal{{O}}(h^{order_expected})$ Ref.')

    plt.xlabel("Element Size $h = L/n$", fontsize=ftSz2); plt.ylabel("$L_2$ Error at $T_{final}$", fontsize=ftSz2)
    plt.title(f"DG Legendre P{p} Convergence (IC: {f_initial_func.__name__})", fontsize=ftSz1)
    plt.gca().invert_xaxis(); plt.grid(True, which='both', linestyle=':')
    plt.legend(fontsize=ftSz3); plt.show()


# --- Comparison Plot for Different n (Legendre) ---
def plot_comparison_n_legendre(L, p, c, T_final, f_initial_func, upwind_param, rk_method, n_values_to_compare, cfl_target):
    """ Plots the final DG Legendre solution for different n against the exact solution. """
    print(f"\n--- Plotting Comparison for Different n (Legendre P{p}) ---")

    n_plot_points_per_element = 50
    # Use the finest n to determine the plot resolution
    n_fine_plot = max(n_values_to_compare) if n_values_to_compare else 10
    x_plot = np.linspace(0, L, n_fine_plot * n_plot_points_per_element + 1)
    u_ex_final = u_exact(x_plot, T_final, L, c, f_initial_func)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_ex_final, 'k--', linewidth=2.5, label=f'Exact T={T_final:.1f}') # Use black dashed for exact

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_values_to_compare))) # Color map

    for i, n_comp in enumerate(n_values_to_compare):
        print(f"Running for comparison plot: n={n_comp}")
        dx_comp = L / n_comp
        dt_comp = 0.0
        if abs(c) > 1e-12: dt_comp = cfl_target * dx_comp / abs(c)
        else: dt_comp = T_final / 100.0
        if dt_comp <= 1e-15: print(f"Warn: dt={dt_comp} skip n={n_comp}."); continue

        m_comp = max(1, int(np.ceil(T_final / dt_comp)))
        dt_adjusted_comp = T_final / m_comp
        print(f"  dt={dt_adjusted_comp:.3e}, m={m_comp}")

        coeffs_history = run_simulation_core_legendre(
            L, n_comp, p, dt_adjusted_comp, m_comp, c, f_initial_func, upwind_param, rk_method)

        if coeffs_history is None or np.any(np.isnan(coeffs_history)):
            print(f"Sim failed n={n_comp}, cannot plot.")
            plt.plot([],[], linestyle='-', color=colors[i], label=f'DG P{p} (n={n_comp}) - FAILED')
        else:
            u_h_final = evaluate_dg_solution_legendre(
                x_plot, coeffs_history, L, n_comp, p, m_comp)
            if np.all(np.isfinite(u_h_final)):
                 plt.plot(x_plot, u_h_final, linestyle='-', color=colors[i], linewidth=1.5, label=f'DG P{p} (n={n_comp})')
            else: # Should be caught above, but fallback
                 plt.plot([],[], linestyle='-', color=colors[i], label=f'DG P{p} (n={n_comp}) - NaN Eval')


    plt.xlabel("x", fontsize=ftSz2); plt.ylabel(f"u(x, T={T_final:.1f})", fontsize=ftSz2)
    plt.title(f"Comparison of DG Legendre P{p} Solutions for Different n", fontsize=ftSz1)
    plt.legend(fontsize=ftSz3); plt.grid(True, linestyle=':')
    ymin = u_ex_final.min() - 0.2; ymax = u_ex_final.max() + 0.2 # Base ylim on exact
    plt.ylim(ymin, ymax); plt.show()


# --- Matplotlib Global Settings ---
ftSz1, ftSz2, ftSz3 = 16, 14, 12 # Slightly smaller defaults
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'

# --- Exact Solution Function ---
def u_exact(x, t, L, c, initial_func):
    """ Calculates the exact solution u(x,t) = u0(x-ct) with periodic wrapping. """
    x = np.asarray(x); x_origin = np.mod(x - c * t, L)
    return initial_func(x_origin, L)

# ==============================================================
# --- Main Execution Block (Legendre Version) ---
# ==============================================================
if __name__ == "__main__":

    # --- Plot Basis Functions P0, P1 (Only relevant if p>=1) ---
    p_basis_plot = 1 # Choose p to plot basis for
    print(f"Plotting P0..P{p_basis_plot} Legendre Basis Functions...")
    xi_plot_basis = np.linspace(-1, 1, 200)
    plt.figure(figsize=(8, 5))
    colors_basis = plt.cm.viridis(np.linspace(0, 1, p_basis_plot + 1)) # Using Viridis colormap
    for i in range(p_basis_plot + 1):
         Pi_func = legendre(i)
         Pi_vals = Pi_func(xi_plot_basis)
         # Construct the label using LaTeX formatting
         label_base = f'P_{i}(\\xi)' # Base label part
         # Append equation part
         if i == 0: label_eq = '=1'
         elif i == 1: label_eq = '=\\xi' # Use LaTeX command for the symbol xi
         else: label_eq = '' # No equation part for higher orders for now
         # Enclose the entire label in $...$ for math mode rendering
         label = f'${label_base}{label_eq}$'
         plt.plot(xi_plot_basis, Pi_vals, color=colors_basis[i], lw=2, label=label)

    plt.title(f"Legendre Basis Functions (p=0..{p_basis_plot}) on Ref Element [-1, 1]")
    # Ensure x-axis label also uses the symbol
    plt.xlabel("Ref Coordinate $\\xi$")
    plt.ylabel("Basis Function Value")
    plt.xticks([-1, 0, 1]); plt.yticks([-1, 0, 1])
    plt.grid(True, linestyle=':'); plt.axhline(0, color='black', lw=0.5); plt.axvline(0, color='black', lw=0.5)
    plt.legend(fontsize=12); plt.show()


    # --- Select Mode ---
    run_normal_simulation = True
    run_conv_study = True
    run_n_comparison_plot = True # <<< New Flag for Comparison Plot

    # --- Configuration (Shared) ---
    L_ = 1.0; c_ = 1.0; T_final = 1.0
    p_ = 1           # << SET POLYNOMIAL DEGREE HERE (e.g., 0, 1, 2) >>
    rk_method = 'RK44'
    upwind_param = 1.0 # 1.0 for upwind (c>0)

    # --- Configuration for Normal Simulation ---
    if run_normal_simulation:
        n_sim = 40
        initial_condition_name = 'sine' # 'sine' or 'square'
        safety_factor_sim = 0.1 # CFL safety factor

        dx_sim = L_ / n_sim; dt_cfl_sim = 0.0
        if abs(c_) > 1e-12: dt_cfl_sim = safety_factor_sim * dx_sim / abs(c_)
        else: dt_cfl_sim = T_final / 100.0
        if dt_cfl_sim <= 1e-15: raise ValueError(f"dt={dt_cfl_sim} too small.")
        m_sim = max(1, int(np.ceil(T_final / dt_cfl_sim))); dt_adjusted_sim = T_final / m_sim
        actual_cfl_sim = abs(c_) * dt_adjusted_sim / dx_sim if abs(dx_sim)>1e-12 and abs(c_)>1e-12 else 0

        print(f"\n--- Running Normal Simulation (Legendre P{p_}) ---")
        print(f"  n={n_sim}, T={T_final:.3f}, c={c_}, p={p_}")
        print(f"  IC: {initial_condition_name}, Upwind alpha: {upwind_param}")
        print(f"  Safety Factor={safety_factor_sim}, Simple CFL={actual_cfl_sim:.3f}")
        print(f"  m={m_sim}, dt={dt_adjusted_sim:.6f}")

        if initial_condition_name == 'sine': f_initial_sim = ic_sine_wave
        elif initial_condition_name == 'square': f_initial_sim = ic_square_wave
        else: raise ValueError(f"Unknown IC name: {initial_condition_name}")

        coeffs_final_sim, l2_err_sim, anim_obj = advection1d_legendre_solver(
            L_, n_sim, p_, dt_adjusted_sim, m_sim, c_,
            f_initial_func=f_initial_sim, a=upwind_param, rktype=rk_method,
            anim=True, save=False, tend=T_final, plot_final=True
        )
        if coeffs_final_sim is not None: print(f"\nSim OK (P{p_}, n={n_sim}). L2 Error={l2_err_sim:.6e}")
        else: print(f"\nSim FAILED (P{p_}, n={n_sim}).")


    # --- Configuration for Convergence Study ---
    if run_conv_study:
        f_initial_conv = ic_sine_wave # MUST use smooth IC
        n_values_conv = [5, 10, 20, 40, 80]
        safety_factor_conv = 0.05 # Use MORE CONSERVATIVE CFL for study
        cfl_target_conv = safety_factor_conv

        # Run for the globally set p_ value
        run_convergence_study_legendre(
            L_, p_, c_, T_final, f_initial_conv, upwind_param, rk_method,
            n_values=n_values_conv, cfl_target=cfl_target_conv, dt_fixed=None
        )

    # --- Configuration for Comparison Plot ---
    if run_n_comparison_plot:
        f_initial_comp = ic_sine_wave # Use smooth or non-smooth for visual comparison
        n_values_comp = [5, 10, 20, 40] # Choose which n values to plot
        safety_factor_comp = 0.1 # Use a reasonable safety factor like normal sim

        plot_comparison_n_legendre(
             L_, p_, c_, T_final, f_initial_comp, upwind_param, rk_method,
             n_values_to_compare=n_values_comp,
             cfl_target=safety_factor_comp # Use CFL target based on safety factor
        )


# ==============================================================
# End of Script
# ==============================================================
