import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre, roots_legendre # Added roots_legendre for L2 error calc later
from scipy.signal import square
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

# --- Provided table (likely CFL limits or related coefficients) ---
table = [
    [1.0000, 1.0000, 1.2564, 1.3926, 1.6085], # p=0
    [0, 0.3333, 0.4096, 0.4642, 0.5348],    # p=1
    [0, 0, 0.2098, 0.2352, 0.2716],       # p=2
    [0, 0, 0.1301, 0.1454, 0.1679],       # p=3 <= We use this row
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
    # Returns the solution history u at all time steps (shape m+1, N_dof)
    return u

def rk22(u, Q, dt, m):
    """ Solves du/dt = Q u using RK22 (Midpoint/Heun). u[0] is initial condition. """
    for i in range(m):
        u_mid = u[i] + dt / 2. * Q.dot(u[i])
        u[i + 1] = u[i] + dt * Q.dot(u_mid)
        # The original formula was u[i+1] = u[i] + dt * Q.dot(u[i] + dt / 2. * Q.dot(u[i]))
        # Let's use the more standard RK2 Heun form:
        # k1 = Q.dot(u[i])
        # k2 = Q.dot(u[i] + dt * k1)
        # u[i+1] = u[i] + dt/2. * (k1 + k2)
        # Or midpoint:
        # k1 = Q.dot(u[i])
        # k2 = Q.dot(u[i] + dt/2. * k1)
        # u[i+1] = u[i] + dt * k2
        # Let's stick to the midpoint version used above for consistency unless specified otherwise.
    # Returns the solution history u at all time steps (shape m+1, N_dof)
    return u

def rk44(u, Q, dt, m):
    """ Solves du/dt = Q u using classic RK44. u[0] is initial condition. """
    for i in range(m):
        K1 = Q.dot(u[i])
        K2 = Q.dot(u[i] + K1 * dt / 2.)
        K3 = Q.dot(u[i] + K2 * dt / 2.)
        K4 = Q.dot(u[i] + K3 * dt)
        u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
    # Returns the solution history u at all time steps (shape m+1, N_dof)
    return u


# --- DG Matrix Building Function ---
def build_matrix(n, p, a):
    """ Builds the DG stiffness matrix L for periodic BCs. """
    # D matrix (derivative term)
    diags_of_D = [2 * np.ones(p - 2 * i) for i in range((p + 1) // 2)]
    offsets_of_D = -np.arange(1, p + 1, 2)
    # Handle p=0 case explicitly
    if p == 0:
        D = sp.bsr_matrix((1, 1)) # 1x1 zero matrix
    else:
        D = sp.diags([np.zeros(p + 1)] + diags_of_D, np.r_[0, offsets_of_D], format='bsr') # Ensure bsr

    # A matrix (Alternating +/- 1)
    A = np.ones((p + 1, p + 1))
    A[1::2, ::2] = -1.
    A[::2, 1::2] = -1.
    A = sp.bsr_matrix(A)

    # B matrix (Alternating +/- 1 in columns)
    B = np.ones((p + 1, p + 1))
    B[:, 1::2] = -1.
    B = sp.bsr_matrix(B)

    # I matrix (All ones - used for identity term scaling)
    # Correction: Should be Identity matrix, not all ones for the term -(1-a)/2 * I
    # However, the code structure implies it's using P_i(1) or P_i(-1) properties.
    # Let's keep the original I matrix as all ones based on the code's formula structure.
    I = np.ones((p + 1, p + 1))
    I = sp.bsr_matrix(I)

    # Flux terms depend on parameter 'a' (upwind parameter)
    # a=1 -> Upwind flux for c>0 (takes from left)
    # a=-1 -> Downwind flux for c>0 (takes from right)
    # a=0 -> Centered flux (average)
    mat_lft = -(1. + a) / 2. * B.T # Coefficient for left neighbor contribution
    mat_rgt = +(1. - a) / 2. * B  # Coefficient for right neighbor contribution
    # Diagonal block: includes derivative, and interface terms from left and right
    mat_ctr = D.T + (1. + a) / 2. * A - (1. - a) / 2. * I

    # Assemble the global block matrix L with periodic boundary conditions
    blocks = []
    for i in range(n):
        this_row = []
        for j in range(n):
            if (j == i - 1) or (i == 0 and j == n - 1): # Left neighbor (with wrap around)
                this_row.append(mat_lft)
            elif j == i:                                 # Diagonal block
                this_row.append(mat_ctr)
            elif (j == i + 1) or (i == n - 1 and j == 0): # Right neighbor (with wrap around)
                this_row.append(mat_rgt)
            else:                                        # Zero block
                this_row.append(None) # None interpreted as zero block by sp.bmat
        blocks.append(this_row)

    L_stiff = sp.bmat(blocks, format='bsr', dtype=np.float64) # Specify dtype
    return L_stiff


# --- Initial Condition Projection ---
def compute_coefficients(f, L, n, p):
    """ Computes initial DG coefficients by L2 projection using 5-point Gauss quadrature. """
    # 5-point Gauss-Legendre weights and points on [-1, 1]
    w_quad = np.array([+0.5688888888888889, +0.4786286704993665, +0.4786286704993665, +0.2369268850561891, +0.2369268850561891])
    s_quad = np.array([+0.0000000000000000, -0.5384693101056831, +0.5384693101056831, -0.9061798459386640, +0.9061798459386640])
    dx = L / n
    psi_basis = [legendre(i) for i in range(p + 1)] # Legendre basis polynomials

    u_coeffs = np.zeros((n, p + 1)) # Array to store coefficients (n_elements, p+1)
    for k in range(n): # Loop over elements
        x_left = k * dx
        # Map quadrature points s_l from [-1, 1] to physical coordinates xsi in element k
        x_quad_k = x_left + (s_quad + 1.0) * dx / 2.0

        for i in range(p + 1): # Loop over basis functions P_i
            # Evaluate f at physical quadrature points
            f_vals = f(x_quad_k, L) # Pass L if needed by f
            # Evaluate P_i at reference quadrature points
            psi_vals = psi_basis[i](s_quad)
            # Compute integral using quadrature: sum(w_q * f(x_q) * P_i(s_q))
            integral = np.dot(w_quad, f_vals * psi_vals)

            # Apply normalization factor (2i+1)/2 and jacobian scaling (dx/2)
            # Formula for coefficient: u_k^i = ((2i+1)/2) / (dx/2) * ∫_elem f(x) P_i(ξ(x)) dx
            #                               = (2i+1)/dx * ∫_elem f(x) P_i(ξ(x)) dx
            # The integral approx ∫ f(x(s)) P_i(s) * (dx/2) ds = integral_value * (dx/2)
            # So u_k^i = (2i+1)/dx * integral_value * (dx/2) = (2i+1)/2 * integral_value
            # Let's re-check the original code's logic.
            # Original: u[k, i] += w[l] * 1./2. * f(xsi, L) * psi[i](s[l]) -> This is missing jacobian dx/2
            # Original: u[k, i] *= (2 * i + 1) -> This applies (2i+1) factor
            # It seems the original compute_coefficients was missing the jacobian dx/2 inside the loop
            # and the factor 1/2. Let's use the formula u_k^i = ( (2i+1)/2 ) * integral_value
            # where integral_value is the quadrature sum of f(x(s)) * P_i(s) * w_q
            # No, the factor is u_k^i = (2i+1) / norm(P_i)^2 * integral(f * P_i * dx)
            # norm(P_i)^2 = integral(P_i^2 * dx) = (dx/2) * integral(P_i^2 * ds) = (dx/2) * (2/(2i+1)) = dx/(2i+1)
            # So u_k^i = (2i+1) / (dx/(2i+1)) * integral(f * P_i * dx) ?? No.
            # L2 projection: find u_h = sum u_i phi_i such that integral((u - u_h) * phi_j * dx) = 0 for all j
            # integral(u * phi_j * dx) = integral( (sum u_i phi_i) * phi_j * dx )
            # integral(u * phi_j * dx) = sum u_i * integral(phi_i * phi_j * dx)
            # integral(u * phi_j * dx) = u_j * integral(phi_j^2 * dx)  (using orthogonality)
            # integral(u * phi_j * dx) = u_j * (dx / (2j+1))
            # So, u_j = ( (2j+1) / dx ) * integral(u * phi_j * dx)
            # integral(u * phi_j * dx) = integral_{-1}^1 u(x(s)) phi_j(s) * (dx/2) ds
            # integral approx = sum_q w_q * u(x(s_q)) * phi_j(s_q) * (dx/2)
            # u_j = ( (2j+1) / dx ) * (dx/2) * sum_q w_q * u(x(s_q)) * phi_j(s_q)
            # u_j = ( (2j+1) / 2 ) * sum_q w_q * u(x(s_q)) * phi_j(s_q)
            u_coeffs[k, i] = (2 * i + 1) / 2.0 * integral

    # Reshape to a flat vector (N_dof = n * (p+1)) for the solver
    return u_coeffs.reshape(n * (p + 1))


# --- Main DG Solver Function (Method of Lines) ---
def advection1d(L, n, dt, m, p, c, f, a, rktype, anim=False, save=False, tend=0.):
    """
    Solves the 1D advection equation using DG with Method of Lines.

    Args:
        L: Domain length
        n: Number of elements
        dt: Time step
        m: Number of time steps
        p: Polynomial degree
        c: Advection speed
        f: Initial condition function f(x, L)
        a: Upwind parameter for build_matrix (1.0 for upwind if c>0)
        rktype: Time integration method ('ForwardEuler', 'RK22', 'RK44')
        anim: Boolean flag to generate animation
        save: Boolean flag to save animation
        tend: Final time (used for animation plotting)

    Returns:
        Array of DG coefficients over time, shape (p+1, n, m+1)
    """
    # Build necessary matrices
    # Inverse mass matrix M^-1 (diagonal for Legendre basis)
    # M_ij = integral(phi_i * phi_j * dx) = delta_ij * (dx / (2i+1))
    # (M^-1)_ii = (2i+1) / dx
    # The code uses inv_mass_matrix = diag(2i+1) and then Q = -c * (n/L) * M_inv * L_stiff
    # where n/L = 1/dx. So Q = -c * (1/dx) * diag(2i+1) * L_stiff.
    # This matches the required scaling (2i+1)/dx in M^-1.
    inv_mass_matrix_diag_term = np.arange(1, 2 * p + 2, 2) # 1, 3, 5, ... (2p+1)
    inv_mass_matrix_diag = np.tile(inv_mass_matrix_diag_term, n)
    inv_mass_matrix = sp.diags([inv_mass_matrix_diag], [0], format='bsr', dtype=np.float64) # Specify dtype

    # Stiffness/Flux matrix L
    stiff_matrix = build_matrix(n, p, a) # This is L = S-F

    # RHS matrix Q for ODE system dU/dt = Q U
    Q = -c * (n / L) * inv_mass_matrix.dot(stiff_matrix) # Note n/L = 1/dx

    # --- Time Integration ---
    N_dof = n * (p + 1)
    # Ensure u_history has the correct float type
    u_history = np.zeros((m + 1, N_dof), dtype=np.float64) # Array to store solution vectors at each time step

    # Set initial condition
    u_history[0] = compute_coefficients(f, L=L, n=n, p=p)

    # Perform time stepping
    print(f"Starting time integration ({rktype})...")
    if rktype == 'ForwardEuler':
        fwd_euler(u_history, Q, dt, m)
    elif rktype == 'RK22':
        rk22(u_history, Q, dt, m)
    elif rktype == 'RK44':
        # Use tqdm for RK44 progress bar
        for i in tqdm(range(m), desc=f"RK44 Steps", unit="step"):
            K1 = Q.dot(u_history[i])
            K2 = Q.dot(u_history[i] + K1 * dt / 2.)
            K3 = Q.dot(u_history[i] + K2 * dt / 2.)
            K4 = Q.dot(u_history[i] + K3 * dt)
            u_history[i + 1] = u_history[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.
        # rk44(u_history, Q, dt, m) # Keep the original function structure if preferred
    else:
        print(f"Error: The integration method '{rktype}' is not recognized.")
        print("       Should be 'ForwardEuler', 'RK22', or 'RK44'")
        raise ValueError
    print("Time integration finished.")

    # Reshape the result for easier handling: (p+1, n_elements, n_timesteps+1)
    u_final_reshaped = u_history.T.reshape((n, p + 1, m + 1))
    u_final_reshaped = np.swapaxes(u_final_reshaped, 0, 1) # Swap axes 0 and 1 -> (p+1, n, m+1)

    # Optional animation
    if anim:
        # Define font sizes for plotting if not globally defined
        global ftSz1, ftSz2, ftSz3
        try:
           ftSz1, ftSz2, ftSz3 # Check if they exist
        except NameError:
           print("Setting default font sizes for animation.")
           ftSz1, ftSz2, ftSz3 = 20, 17, 14 # Default values
           plt.rcParams["text.usetex"] = True
           plt.rcParams['font.family'] = 'serif'

        plot_function(u_final_reshaped, L=L, n=n, dt=dt, m=m, p=p, c=c, f=f, save=save, tend=tend)

    return u_final_reshaped


# --- Animation Function ---
# Note: This function is kept for potential future use but is not called when anim=False
def plot_function(u, L, n, dt, m, p, c, f, save=False, tend=0.):
    """ Creates animation of the DG solution vs exact solution. """
    n_plot = 100 # Points per element for smooth plotting
    v = np.zeros((n, m + 1, n_plot + 1)) # Reconstructed solution values
    r = np.linspace(-1, 1, n_plot + 1) # Reference element coordinates
    psi = np.array([legendre(i)(r) for i in range(p + 1)]).T # Basis functions evaluated at plot points
    dx = L / n
    full_x = np.linspace(0., L, n * n_plot + 1) # Global x coordinates for plotting

    # Reconstruct solution u(x,t) from coefficients for all times and elements
    print("Reconstructing solution for animation...")
    for time_idx in tqdm(range(m + 1), desc="Reconstructing Frames"):
        for elem_idx in range(n):
            # u has shape (p+1, n, m+1)
            coeffs_at_time_elem = u[:, elem_idx, time_idx]
            v[elem_idx, time_idx, :] = np.dot(psi, coeffs_at_time_elem)

    fig, ax = plt.subplots(1, 1, figsize=(8., 4.5))
    fig.tight_layout()
    ax.grid(ls=':')

    # Ensure font sizes are defined
    global ftSz1, ftSz2, ftSz3
    try:
       ftSz1, ftSz2, ftSz3
    except NameError:
       ftSz1, ftSz2, ftSz3 = 20, 17, 14
       plt.rcParams["text.usetex"] = True # Assume tex is available
       plt.rcParams['font.family'] = 'serif'

    time_template = r'$t = \mathtt{{{:.3f}}} \;[s]$' # Use more precision for time display
    time_text = ax.text(0.815, 0.92, '', fontsize=ftSz1, transform=ax.transAxes)

    # Create lines for each element's solution segment
    lines = [ax.plot([], [], color='C0')[0] for _ in range(n)]
    # Create line for exact solution
    # Need to handle periodic boundary for exact solution display
    exact_func = lambda x, t: f(np.mod(x - c * t, L), L) # Periodically wrapped exact solution
    exact, = ax.plot([], [], color='C1', alpha=0.5, lw=5, zorder=0, label='Exact')
    # Set plot limits based on initial exact solution
    initial_exact_y = exact_func(full_x, 0)
    ymin = min(initial_exact_y) - 0.5 * (max(initial_exact_y)-min(initial_exact_y)) if max(initial_exact_y)!=min(initial_exact_y) else min(initial_exact_y)-0.5
    ymax = max(initial_exact_y) + 0.5 * (max(initial_exact_y)-min(initial_exact_y)) if max(initial_exact_y)!=min(initial_exact_y) else max(initial_exact_y)+0.5
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r"$x$", fontsize=ftSz2)
    ax.set_ylabel(r"$u(x,t)$", fontsize=ftSz2)
    ax.legend()
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.11, top=0.995)

    def init():
        """ Initializes the animation plot. """
        exact.set_data(full_x, exact_func(full_x, 0))
        time_text.set_text(time_template.format(0))
        for k, line in enumerate(lines):
            # Set x data for each element's line segment
            x_elem = np.linspace(k * dx, (k + 1) * dx, n_plot + 1)
            line.set_data(x_elem, v[k, 0, :])
        return tuple([*lines, exact, time_text])

    def animate(t_idx):
        """ Updates the plot for frame t_idx. """
        current_time = t_idx * dt
        exact.set_ydata(exact_func(full_x, current_time))
        time_text.set_text(time_template.format(current_time))
        for k, line in enumerate(lines):
            line.set_ydata(v[k, t_idx, :])
        # pbar_anim.update(1) # Update animation progress bar
        return tuple([*lines, exact, time_text])

    # Animation setup
    fps = 25 # Frames per second for animation
    num_frames = m + 1
    interval = 1000 / fps # Interval in ms

    print("Creating animation...")
    # pbar_anim = tqdm(total=num_frames, desc="Animating Frames") # Separate progress bar for animation
    # init() # Call init manually before FuncAnimation can be helpful
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=False,
                         init_func=init, repeat=False) # Use init_func here

    if save:
        anim_filename = f"./figures/dg_advection_p{p}_n{n}_{rktype}.mp4"
        # Ensure figure directory exists
        import os
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
    else:
        plt.xlim(0,1)
        plt.show()
        

    # pbar_anim.close()
    return


# --- Matplotlib Global Settings ---
# Moved potentially missing definitions here for safety if plot_function is called directly
ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = False  
plt.rcParams['font.family'] = 'serif'


# ==============================================================
# --- Main Execution Block ---
# ==============================================================
if __name__ == "__main__":

    # --- Configuration for Baseline Simulation ---
    L_ = 1.0         # Domain Length [0, L]
    n_ = 20          # Number of spatial elements (Matches paper Sec 8.1)
    p_ = 3           # Polynomial degree (Matches paper Sec 8.1)
    c_ = 1.0         # Advection speed (a=1 in paper Sec 8.1)

    # --- Time Stepping Parameters ---
    # Using the CFL constraint from the provided table and formula
    # For p=3, the 4th value (index 3) in the table is 1.3926 (Matches paper Sec 8.1)
    # Column index 3 corresponds to the 4th column value for p=3
    try:
        CFL_limit = table[p_][3] # Should be 1.3926 for p=3 (using table value from original code)
                                # Check if this is the correct column for RK44 stability
    except IndexError:
        print(f"Error: p={p_} or column index 3 is out of bounds for the provided table.")
        exit()

    safety_factor = 0.5     # Safety factor for stability
    dt_ = safety_factor * CFL_limit / abs(c_) * (L_ / n_) # Time step calculation (use abs(c))

    # --- Final time T ---
    # Run for one full period so the exact solution returns to the initial state
    T_final = L_ / abs(c_) # T = 1.0 / 1.0 = 1.0
    # Ensure m_ is at least 1
    m_ = max(1, int(np.ceil(T_final / dt_))) # Number of time steps needed
    # Optional: Adjust dt slightly to exactly hit T_final
    dt_adjusted = T_final / m_
    print("(Using adjusted dt to exactly reach T_final)")

    # --- Print Configuration ---
    print(f"Configuration:")
    print(f"  Domain L = {L_}, Elements n = {n_}, Polynomial Degree p = {p_}")
    print(f"  Advection Speed c = {c_}")
    print(f"Time Discretization:")
    print(f"  CFL Limit (p={p_}, col_idx=3) = {CFL_limit}")
    print(f"  Safety Factor = {safety_factor}")
    print(f"  Initial dt estimate = {dt_:.6f}")
    print(f"  Target T_final = {T_final:.3f}")
    print(f"  Number of steps m = {m_}")
    print(f"  Adjusted dt = {dt_adjusted:.6f}")

    # --- Initial Condition ---
    # Square wave u(x,0) = square(2*pi*x/L, 1/3) (from paper Sec 8.1)
    # Need lambda function f(x, L) format for compute_coefficients
    f_initial = lambda x, L: square(2 * np.pi * x / L, 1./3.)

    # --- DG Parameters ---
    rk_method = 'RK44'       # As specified in paper Sec 8.1
    upwind_param = 1.0       # Use 1.0 for full upwind with c_>0 ('a' in build_matrix)

    # print(f"Running simulation with {rk_method}...") # Moved inside advection1d

    # --- Run Simulation ---
    # Call the main function, disable animation/saving for baseline validation
    u_coeffs_time_history = advection1d(L_, n_, dt_adjusted, m_, p_, c_,
                                       f=f_initial, a=upwind_param,
                                       rktype=rk_method, anim=True,
                                       save=False, tend=T_final)

    # --- Post-processing ---
    # print("Simulation finished.") # Moved inside advection1d

    # Extract coefficients at the final time step (index m)
    # Shape is (p+1, n, m+1), so final index is m
    # Correct indentation here:
    final_coeffs_per_element = u_coeffs_time_history[:, :, m_] # Shape (p+1, n)

    # --- Plotting (Action 1.3) ---
    print("Post-processing: Plotting final solution...")

    # Helper function to evaluate the DG solution at arbitrary points x
    def evaluate_dg_solution(x_eval, coeffs_element_wise, L, n, p):
        """ Evaluates the DG solution given by coefficients on each element. """
        dx = L / n
        u_h_eval = np.zeros_like(x_eval, dtype=float)
        psi_basis = [legendre(i) for i in range(p + 1)]

        for i, x_val in enumerate(x_eval):
            # Handle edge cases and find element index
            if x_val >= L: # Include L in the last element
                element_idx = n - 1
                xi_val = 1.0
            elif x_val <= 0: # Include 0 in the first element
                element_idx = 0
                xi_val = -1.0
            else:
                element_idx = int(np.floor(x_val / dx))
                element_idx = min(element_idx, n - 1) # Ensure index is within bounds [0, n-1]
                x_left = element_idx * dx
                # Map x_val to local coordinate xi in [-1, 1]
                xi_val = 2 * (x_val - x_left) / dx - 1.0
                # Clamp xi_val to handle potential floating point issues at boundaries
                xi_val = np.clip(xi_val, -1.0, 1.0)


            # Evaluate basis functions at xi_val
            psi_at_xi = np.array([psi(xi_val) for psi in psi_basis])

            # Compute solution as dot product of coefficients and basis values
            # coeffs_element_wise has shape (p+1, n)
            u_h_eval[i] = np.dot(psi_at_xi, coeffs_element_wise[:, element_idx])

        return u_h_eval

    # Define the exact solution function
    def u_exact(x, t, L, c, initial_func):
        """ Calculates the exact solution u(x,t) = u0(x-ct) with periodic wrapping. """
        # Calculate the position where the value originated at t=0, handling periodicity
        x_origin = np.mod(x - c * t, L) # Modulo L ensures periodicity
        return initial_func(x_origin, L) # Pass L if needed by f_initial

    # Generate points for plotting
    n_plot_points_per_element = 50 # Increase for smoother curves
    x_plot = np.linspace(0, L_, n_ * n_plot_points_per_element + 1)

    # Evaluate numerical and exact solutions at plot points
    u_h_final = evaluate_dg_solution(x_plot, final_coeffs_per_element, L_, n_, p_)
    u_ex_final = u_exact(x_plot, T_final, L_, c_, f_initial)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_ex_final, 'r-', linewidth=3, alpha=0.7, label=f'Exact Solution at T={T_final:.2f}')
    plt.plot(x_plot, u_h_final, 'b-', linewidth=1.5, label=f'DG Solution (p={p_}, n={n_}, RK44)') # Solid line for DG
    # Add element boundaries for clarity
    for k_elem in range(n_ + 1):
         plt.axvline(k_elem * L_ / n_, color='gray', linestyle=':', linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title(f"DG Solution vs Exact Solution at T={T_final:.2f} (Baseline)")
    plt.legend()
    plt.grid(True, linestyle=':')
    # Adjust ylim based on initial condition range (-1 to 1 for square wave)
    plt.ylim(-1.5, 1.5)
    plt.show()


    # --- L2 Error Calculation (Action 1.4) ---
    print("Post-processing: Calculating L2 error...")
    # Number of quadrature points (should be high enough for degree 2p accuracy)
    num_quad_points = p_ + 2 # Degree 2p+1 -> exact for poly degree 2*(p+2)-1 = 2p+3. p+1 -> exact for 2p+1. Let's use p+2.
    if num_quad_points < 1: num_quad_points = 1 # Ensure at least 1 point
    xi_quad, w_quad = roots_legendre(num_quad_points) # Points xi_q and weights w_q in [-1, 1]

    l2_error_sq_sum = 0.0
    dx = L_ / n_
    jacobian = dx / 2.0 # Jacobian for mapping [-1, 1] to element dx/dxi

    # Pre-calculate Legendre basis at quadrature points for efficiency
    psi_basis = [legendre(i) for i in range(p_ + 1)]
    psi_at_quad = np.array([[psi(xi) for psi in psi_basis] for xi in xi_quad]) # Shape (num_quad_points, p+1)

    for k in range(n_): # Loop over elements
        x_left = k * dx
        # Map quad points xi_q from [-1, 1] to physical coordinates x_q in element k
        x_quad_k = x_left + (xi_quad + 1) * jacobian

        # Evaluate numerical solution u_h at physical quadrature points x_q in element k
        # coeffs for element k has shape (p+1,)
        coeffs_k = final_coeffs_per_element[:, k]
        # u_h(x_q) = sum_i coeffs_k[i] * P_i(xi_q) = psi_at_quad @ coeffs_k
        u_h_at_quad_k = np.dot(psi_at_quad, coeffs_k) # Shape (num_quad_points,)

        # Evaluate exact solution u_exact at physical quadrature points x_q in element k
        u_ex_at_quad_k = u_exact(x_quad_k, T_final, L_, c_, f_initial)

        # Calculate squared error at quadrature points: (u_h(x_q) - u_ex(x_q))^2
        error_sq_at_quad = (u_h_at_quad_k - u_ex_at_quad_k)**2

        # Add contribution to integral: ∫_elem (err^2) dx = ∫_{-1}^1 (err(x(xi))^2) * jacobian * dxi
        # Approximated by sum_q [ w_q * err(x(xi_q))^2 * jacobian ]
        l2_error_sq_sum += np.sum(w_quad * error_sq_at_quad) * jacobian

    # Final L2 error is the square root of the total sum
    l2_error = np.sqrt(l2_error_sq_sum)
    print(f"L2 Error ||u_h - u_exact|| at T={T_final:.2f} = {l2_error:.6e}")

# ==============================================================
# End of Script
# ==============================================================
