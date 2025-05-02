"""
Space-Time Discontinuous Galerkin Method for Linear Advection Equation

Problem setup:
- Linear advection equation: du/dt + a * du/dx = 0
- Sine wave initial condition
- Domain length L=1
- Number of spatial elements Nx=20
- Number of temporal elements Nt=40 (to ensure stability for P=1)
- Polynomial order p=1
- Advection speed a=1
- Periodic boundary conditions
- Final time T=1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
# Need a linear solver
import numpy.linalg as npl # Keep import for comparison if needed, though will use sparse solver
# Import sparse matrix formats and sparse solvers
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import os
import math
# --- Import for 3D plotting ---
from mpl_toolkits.mplot3d import Axes3D
# --- Import for live animation ---
import matplotlib.animation as animation

# --- Matplotlib Global Settings ---
ftSz1, ftSz2, ftSz3 = 20, 17, 14
# Use 'Agg' backend for non-interactive plots during convergence study,
# but keep default for interactive plots/animation unless explicitly saving
# plt.rcParams["text.usetex"] = False # Keep original setting
plt.rcParams['font.family'] = 'serif' # Keep original setting
# Added: Potentially switch backend for speed/stability in non-interactive mode
# This line is commented out to strictly adhere to not changing existing settings
# except where required for the sparse solver or requested functionality.
# For robustness in automated scripts, this might be useful:
# import matplotlib
# if not plt.get_backend() == 'MacOSX': # Avoid changing interactive backends
#     matplotlib.use('Agg')

# --- Initial Condition Functions ---
def ic_sine_wave(x, L):
    """ Smooth sine wave initial condition (Good for convergence study). """
    return np.sin(2 * np.pi * x / L)

def exact_solution(x, t, L, a, f_init):
    """
    Exact solution for the linear advection equation with periodic boundary conditions

    Parameters:
    -----------
    x : ndarray
        x-coordinates
    t : float
        Time
    L : float
        Domain length
    a : float
        Advection speed
    f_init : function
        Initial condition function

    Returns:
    --------
    ndarray
        Exact solution values
    """
    # For linear advection, exact solution is initial condition shifted by a*t
    # Handle periodicity correctly for shifting
    x_shifted = np.fmod(x - a * t, L)
    x_shifted[x_shifted < 0] += L
    return f_init(x_shifted, L)

# --- Space-Time DG Implementation with Lagrange Basis ---
class SpaceTimeDG:
    def __init__(self, n_elements_spatial=20, n_elements_temporal=20, poly_order=3, domain_length=1.0, advection_speed=1.0, t_final=1.0):
        """
        Initialize the Space-Time DG solver for linear advection

        Parameters:
        -----------
        n_elements_spatial : int
            Number of elements in the spatial domain
        n_elements_temporal : int
            Number of temporal elements
        poly_order : int
            Polynomial order for the DG method
        domain_length : float
            Length of the spatial domain
        advection_speed : float
            Advection speed coefficient
        t_final : float
            Final time for the simulation
        """
        self.n_elements_spatial = n_elements_spatial
        self.n_elements_temporal = n_elements_temporal
        self.p = poly_order
        self.L = domain_length
        self.a = advection_speed
        self.T = t_final

        # Element size in space
        self.dx = self.L / self.n_elements_spatial

        # Element size in time
        self.dt = self.T / self.n_elements_temporal

        # Number of nodes per dimension (p+1 GLL nodes)
        self.n_nodes_1d = self.p + 1

        # Total number of nodes per space-time element
        self.n_nodes_per_element = self.n_nodes_1d ** 2

        # Total number of degrees of freedom
        self.n_dofs = self.n_elements_spatial * self.n_elements_temporal * self.n_nodes_per_element

        # Create space-time mesh (will be called again in solve after p is set)
        self.x_elements = np.linspace(0, self.L, self.n_elements_spatial + 1)
        self.t_elements = np.linspace(0, self.T, self.n_elements_temporal + 1)
        # Placeholder, will be filled by create_mesh
        self.ref_nodes_1d = None
        self.ref_nodes_x = None
        self.ref_nodes_t = None
        self.x_nodes = None
        self.t_nodes = None

        # Initialize solution vector (will be filled after solving)
        self.u = None

    def create_mesh(self):
        """Create the space-time mesh"""
        Nx = self.n_elements_spatial
        Nt = self.n_elements_temporal
        L = self.L
        T = self.T
        p = self.p
        d = self.n_nodes_1d # nodes per dim

        # Create 1D Gauss-Lobatto-Legendre nodes for reference element [-1, 1]
        # For p=1, nodes are -1, 1
        # For p=2, nodes are -1, 0, 1
        # For p=3, nodes are -1, -sqrt(1/5), sqrt(1/5), 1
        # Note: For p > 1, using roots_legendre for GLL nodes is more robust.
        # For the specified p=1, the hardcoded nodes are correct.
        # For consistency and future expansion, using roots_legendre is better.
        if p == 1:
             self.ref_nodes_1d = np.array([-1.0, 1.0])
        else:
            # GLL nodes are the roots of (1-xi^2)P_p'(xi), which are -1, 1, and roots of P_p'(xi).
            # For p>1, we can use roots of Legendre polynomial of degree p-1, plus -1 and 1.
            # scipy.special.roots_legendre gives roots of P_p(x). Need roots of P_{p-1}'(x) or similar
            # A standard way for GLL is roots of P_p'(xi) and endpoints -1, 1.
            # Let's stick to simple hardcoded nodes for p=1, 2, 3 based on typical GLL.
            if p == 2:
                 self.ref_nodes_1d = np.array([-1.0, 0.0, 1.0])
            elif p == 3:
                 self.ref_nodes_1d = np.array([-1.0, -np.sqrt(1/5.0), np.sqrt(1/5.0), 1.0])
            else:
                 # For higher p, using a dedicated function for GLL nodes is recommended.
                 # The Chebyshev nodes provide a simple approximation, but are not GLL.
                 # Let's raise an error or implement a proper GLL finder if p > 3.
                 # For the P=1 case specified, the first branch is sufficient.
                 raise NotImplementedError(f"GLL nodes not implemented for poly_order p={p} > 3.")


        # Create 2D tensor product nodes for reference element [-1, 1]^2
        self.ref_nodes_x, self.ref_nodes_t = np.meshgrid(self.ref_nodes_1d, self.ref_nodes_1d)
        self.ref_nodes_x = self.ref_nodes_x.flatten()
        self.ref_nodes_t = self.ref_nodes_t.flatten()

        # Create physical space-time mesh element boundaries
        self.x_elements = np.linspace(0, L, Nx + 1)
        self.t_elements = np.linspace(0, T, Nt + 1)

        # Create physical nodes for each space-time element
        self.x_nodes = np.zeros((Nt, Nx, self.n_nodes_per_element))
        self.t_nodes = np.zeros((Nt, Nx, self.n_nodes_per_element))

        for i in range(Nt):  # time element index (0 to Nt-1)
            for j in range(Nx):  # space element index (0 to Nx-1)
                # Map from reference element [-1, 1]^2 to physical element [x_j, x_{j+1}] x [t_i, t_{i+1}]
                x_left = self.x_elements[j]
                x_right = self.x_elements[j+1]
                t_bottom = self.t_elements[i]
                t_top = self.t_elements[i+1]

                # Map reference nodes to physical nodes
                self.x_nodes[i, j, :] = 0.5 * (x_right - x_left) * (self.ref_nodes_x + 1) + x_left
                self.t_nodes[i, j, :] = 0.5 * (t_top - t_bottom) * (self.ref_nodes_t + 1) + t_bottom


    def lagrange_basis_1d(self, xi, node_idx):
        """
        Evaluate the 1D Lagrange basis function L_k(xi) at point xi, where L_k is 1 at ref_nodes_1d[node_idx] and 0 at others.

        Parameters:
        -----------
        xi : float or ndarray
            Coordinate(s) in reference element [-1, 1]
        node_idx : int
            Index of the node (0 to p) corresponding to L_k

        Returns:
        --------
        float or ndarray
            Value(s) of the basis function L_k(xi)
        """
        value = 1.0
        for j in range(self.n_nodes_1d):
            if j != node_idx:
                denom = (self.ref_nodes_1d[node_idx] - self.ref_nodes_1d[j])
                if abs(denom) < 1e-12: # Avoid division by zero if nodes are not distinct
                     # This should not happen with distinct GLL nodes, but as a safeguard
                     if isinstance(xi, np.ndarray):
                          # Handle array input: if xi is close to node j, result is 0
                          # Otherwise continue product
                          value = np.where(np.abs(xi - self.ref_nodes_1d[j]) < 1e-12, 0.0, value * (xi - self.ref_nodes_1d[j]) / denom)
                     elif abs(xi - self.ref_nodes_1d[j]) < 1e-12:
                          # Scalar input: xi is a different node j, return 0
                          return 0.0
                     else:
                          # This indicates an issue with distinctness of nodes
                          raise ValueError(f"Reference nodes are not distinct or calculation error. Nodes: {self.ref_nodes_1d}")
                else:
                     value *= (xi - self.ref_nodes_1d[j]) / denom
        return value

    def lagrange_basis_1d_derivative(self, xi, node_idx):
        """
        Evaluate the 1D Lagrange basis function derivative L'_k(xi) at point xi.

        Parameters:
        -----------
        xi : float or ndarray
            Coordinate(s) in reference element [-1, 1]
        node_idx : int
            Index of the node (0 to p) corresponding to L_k

        Returns:
        --------
        float or ndarray
            Value(s) of the derivative L'_k(xi)
        """
        # L_k'(xi) = sum_{m!=k} [ Product_{j!=k, j!=m} (xi - xi_j) / (xi_k - xi_j) ] / (xi_k - xi_m)
        # Handle scalar and array xi
        if isinstance(xi, np.ndarray):
             dphi_dxi_1d = np.zeros_like(xi, dtype=float)
        else:
             dphi_dxi_1d = 0.0

        for m in range(self.n_nodes_1d):
             if m != node_idx:
                  prod = 1.0
                  denom_m = (self.ref_nodes_1d[node_idx] - self.ref_nodes_1d[m])
                  if abs(denom_m) < 1e-12: raise ValueError("Reference nodes are not distinct.")

                  for k_prod in range(self.n_nodes_1d):
                       if k_prod != node_idx and k_prod != m:
                           denom_k = (self.ref_nodes_1d[node_idx] - self.ref_nodes_1d[k_prod])
                           if abs(denom_k) < 1e-12: raise ValueError("Reference nodes are not distinct.")
                           # Need to handle xi being equal to a node value inside the product carefully
                           # For numerical integration, xi are typically quadrature points, not nodes.
                           # Assuming xi is not exactly one of the nodes ref_nodes_1d[k_prod] for k_prod != node_idx, m
                           prod *= (xi - self.ref_nodes_1d[k_prod]) / denom_k

                  dphi_dxi_1d += prod / denom_m
        return dphi_dxi_1d


    def lagrange_basis_2d(self, xi, eta, node_i, node_j):
        """
        Evaluate the 2D tensor product Lagrange basis function phi_{node_i, node_j}(xi, eta) = L_{node_i}(xi) * L_{node_j}(eta)

        Parameters:
        -----------
        xi : float or ndarray
            x-coordinate(s) in reference element [-1, 1]^2
        eta : float or ndarray
            t-coordinate(s) in reference element [-1, 1]^2
        node_i : int
            Index of the node in x-direction (0 to p)
        node_j : int
            Index of the node in t-direction (0 to p)

        Returns:
        --------
        float or ndarray
            Value(s) of the basis function
        """
        # Ensure xi and eta are treated as arrays if one is
        xi_arr = np.asarray(xi)
        eta_arr = np.asarray(eta)

        return self.lagrange_basis_1d(xi_arr, node_i) * self.lagrange_basis_1d(eta_arr, node_j)


    def lagrange_basis_2d_gradient(self, xi, eta, node_i, node_j):
        """
        Evaluate the gradient of the 2D tensor product Lagrange basis function
        phi_{node_i, node_j}(xi, eta) = L_{node_i}(xi) * L_{node_j}(eta)

        d/dxi (phi_{node_i, node_j}) = L'_{node_i}(xi) * L_{node_j}(eta)
        d/deta (phi_{node_i, node_j}) = L_{node_i}(xi) * L'_{node_j}(eta)

        Parameters:
        -----------
        xi : float or ndarray
            x-coordinate(s) in reference element [-1, 1]^2
        eta : float or ndarray
            t-coordinate(s) in reference element [-1, 1]^2
        node_i : int
            Index of the node in x-direction (0 to p)
        node_j : int
            Index of the node in t-direction (0 to p)

        Returns:
        --------
        tuple
            (d/dxi, d/deta) derivatives of the basis function (float or ndarray)
        """
        # Ensure xi and eta are treated as arrays if one is
        xi_arr = np.asarray(xi)
        eta_arr = np.asarray(eta)

        dphi_dxi = self.lagrange_basis_1d_derivative(xi_arr, node_i) * self.lagrange_basis_1d(eta_arr, node_j)
        dphi_deta = self.lagrange_basis_1d(xi_arr, node_i) * self.lagrange_basis_1d_derivative(eta_arr, node_j)

        return dphi_dxi, dphi_deta

    def get_quadrature_points_weights(self):
        """
        Get quadrature points and weights for numerical integration

        Returns:
        --------
        tuple
            (points_x, points_t, weights) for 2D numerical integration
        """
        # Using Gauss-Legendre quadrature points
        # We need enough points for the volume integral: deg(-u psi_t - a u psi_x) where u, psi are deg p
        # terms like u psi_t involves deg(p + p-1) = 2p-1.
        # terms like u psi_x involves deg(p + p-1) = 2p-1.
        # So we need quadrature exact for degree 2p-1. Gauss-Legendre with p points is exact for 2p-1.
        # Let's use p+1 points for safety and consistency with typical DG settings.
        n_quad_1d = self.p + 1 # Use p+1 points, sufficient for volume integrand degree 2p-1

        # Get 1D Gauss-Legendre quadrature points and weights
        points_1d, weights_1d = roots_legendre(n_quad_1d)

        # Create 2D tensor product quadrature
        points_x, points_t = np.meshgrid(points_1d, points_1d)
        points_x = points_x.flatten()
        points_t = points_t.flatten()

        # Compute 2D weights
        weights = np.outer(weights_1d, weights_1d).flatten()

        return points_x, points_t, weights

    def assemble_system(self, f_init):
        """
        Assemble the global system for the Space-Time DG method using the standard
        weak form with integration by parts and upwind fluxes (for a > 0).

        Weak form: ∫_K (-u psi_t - a u psi_x) dxdt + ∫_top u_h psi dS + ∫_bottom (-u^*) psi dS
                   + ∫_right (a u_h n_x) psi dS + ∫_left (a u^* n_x) psi dS = 0

        Upwind fluxes for a > 0:
        Top face (n_t=1): u^* = u_h
        Bottom face (n_t=-1): u^* = u_{prev_time} (or u_init for i=0)
        Right face (n_x=1): u^* = u_h
        Left face (n_x=-1): u^* = u_{neighbor_left}

        Terms to assemble (all on LHS initially):
        ∫_K (-u_m phi_m psi_l,t - a u_m phi_m psi_l,x) dxdt * u_{i,j,m} (Volume terms for current element)
        + ∫_top u_m phi_m(xi,1) psi_l(xi,1) dS * u_{i,j,m} (Top face term for current element)
        + ∫_bottom (-u^*) psi_l(xi,-1) dS (Bottom face term - u^* is from element (i-1,j) or IC)
        + ∫_right (a u_h n_x) psi dS + ∫_left (a u^* n_x) psi dS

        Move known/neighbor terms to RHS? No, in a global STDG matrix, neighbor terms stay on the LHS, linking DOFs across elements.
        The RHS comes from non-homogeneous boundary conditions or source terms. In this case, the IC acts like a non-homogeneous inflow BC on the bottom face of the first time row.

        Correct terms on LHS (multiplied by corresponding trial coefficients):
        ∫_K (-phi_m psi_l,t - a phi_m psi_l,x) dxdt * u_{i,j,m}
        + ∫_top phi_m(xi,1) psi_l(xi,1) dS * u_{i,j,m}
        + ∫_bottom (-phi'_m'(xi,1)) psi_l(xi,-1) dS * u_{i-1,j,m'}  <-- Coupling term from element (i-1,j)
        + ∫_right a phi_m(1, eta) psi_l(1, eta) dS * u_{i,j,m}
        + ∫_left -a phi_m^{neighbor_left}(1, eta) psi_l(-1, eta) dS * u_{i,j-1,m'} (or u_{i,Nx-1,m'} for j=0)

        The RHS `b` only receives contributions from the initial condition on the bottom boundary of the first time slice (i=0).
        Specifically, the term ∫_{bottom} (-u_init) psi_l dS which, when moved to the RHS (conceptually), becomes ∫_{bottom} u_init psi_l dS.

        Parameters:
        -----------
        f_init : function
            Initial condition function

        Returns:
        --------
        tuple
            (A, b) where A is the system matrix and b is the right-hand side
        """
        Nx = self.n_elements_spatial
        Nt = self.n_elements_temporal
        p = self.p
        n_nodes_1d = self.n_nodes_1d
        n_nodes_per_element = self.n_nodes_per_element
        n_dofs = self.n_dofs

        # Initialize system matrix (using LIL for easy filling) and right-hand side (dense)
        # CHANGED: Use sparse matrix format (LIL) instead of dense numpy array for A
        A = sparse.lil_matrix((n_dofs, n_dofs))
        b = np.zeros(n_dofs) # b should be zero for homogeneous equation with IC as BC

        # Get quadrature points and weights for numerical integration
        quad_x, quad_t, quad_weights = self.get_quadrature_points_weights()
        n_quad = len(quad_weights)

        # Get 1D quadrature points/weights for face integrals (using Legendre weights)
        # Needs to integrate product of two basis functions on the face, degree 2p.
        # Legendre quadrature with p+1 points is exact for degree 2p-1.
        # Using p+1 for safety and typical practice.
        n_face_quad_1d = self.p + 1 # Use p+1 points, sufficient for face integrand degree 2p

        face_quad_points_1d, face_quad_weights_1d = roots_legendre(n_face_quad_1d)


        print("Assembling Space-Time DG system (standard weak form, upwind fluxes, corrected temporal coupling)...")
        print(f"Spatial elements: {Nx}, Temporal elements: {Nt}, Poly Order: {p}")
        print(f"Total DOFs: {n_dofs}")
        print(f"Using sparse matrix assembly.")


        # Loop over space-time elements
        for i in tqdm(range(Nt), desc="Temporal Elements"): # time element index (0 to Nt-1)
            for j in range(Nx):  # space element index (0 to Nx-1)
                # Element indices
                elem_idx_global = i * Nx + j # Global index for this element (0 to Nt*Nx-1)

                # Element boundaries
                x_left = self.x_elements[j]
                x_right = self.x_elements[j+1]
                t_bottom = self.t_elements[i]
                t_top = self.t_elements[i+1]

                # Jacobian of the mapping from reference to physical element
                dx_phys = x_right - x_left
                dt_phys = t_top - t_bottom
                J_vol = (dx_phys / 2.0) * (dt_phys / 2.0) # Determinant of the Jacobian
                J_face_x = dt_phys / 2.0 # Jacobian for spatial face integral mapping reference [-1,1]_t to physical dt_phys
                J_face_t = dx_phys / 2.0 # Jacobian for temporal face integral mapping reference [-1,1]_x to physical dx_phys


                # Loop over test functions (psi_l, l = test_idx_local)
                for test_idx_local in range(n_nodes_per_element):
                    # Global row index in the system matrix A for this test function's equation
                    global_row_idx = elem_idx_global * n_nodes_per_element + test_idx_local

                    # Map local 2D index to 1D indices (row-major order for phi_kl = L_k(xi)L_l(eta), with l=temporal index)
                    # The original code had test_i = test_idx_local % n_nodes_1d, test_j = test_idx_local // n_nodes_1d
                    # This maps a 1D index `k` (from 0 to n_nodes_per_element - 1) to 2D indices `i`, `j`
                    # such that `k = j * n_nodes_1d + i`. So `i = k % n_nodes_1d`, `j = k // n_nodes_1d`.
                    # If trial/test bases are phi_{i_spatial, j_temporal}, where i_spatial is column index, j_temporal is row index
                    # in the 2D basis grid for an element (like phi_{00}, phi_{10}, phi_{01}, phi_{11} for p=1),
                    # and the global DOF ordering is (phi_{00}, phi_{10}, phi_{01}, phi_{11}) within an element block,
                    # then local_dof_idx corresponds to this 1D ordering.
                    # The 2D basis function phi_{i_spatial, j_temporal} is L_{i_spatial}(xi) * L_{j_temporal}(eta).
                    # So test_idx_local -> (test_i, test_j) means L_{test_i}(xi) * L_{test_j}(eta).
                    # The mapping should likely be: test_i (spatial index) = test_idx_local % n_nodes_1d,
                    # test_j (temporal index) = test_idx_local // n_nodes_1d. This matches the original code.
                    test_i = test_idx_local % n_nodes_1d # spatial index of test basis L_{test_i}(xi)
                    test_j = test_idx_local // n_nodes_1d # temporal index of test basis L_{test_j}(eta)


                    # --- Volume Integral Contribution (from current element trial functions) ---
                    # ∫_K (-u_m phi_m psi_l,t - a u_m phi_m psi_l,x) dxdt * u_{i,j,m}
                    # (Integral for test function psi_l over element (i,j), coupled with trial function phi_m from element (i,j))
                    for trial_idx_local in range(n_nodes_per_element):
                        # Global column index in the system matrix A for this trial function's coefficient
                        global_col_idx = elem_idx_global * n_nodes_per_element + trial_idx_local

                        # Map local 1D trial index to 2D indices (spatial, temporal)
                        trial_i = trial_idx_local % n_nodes_1d
                        trial_j = trial_idx_local // n_nodes_1d

                        # Evaluate basis functions and their derivatives at quadrature points (reference element)
                        phi_trial_q_vals = self.lagrange_basis_2d(quad_x, quad_t, trial_i, trial_j)
                        dphi_test_dxi_q_vals, dphi_test_deta_q_vals = self.lagrange_basis_2d_gradient(quad_x, quad_t, test_i, test_j)


                        # Sum over quadrature points for the volume integral
                        # Term: - phi_trial_q * dphi_test_deta_q * (2.0/dt_phys) - self.a * phi_trial_q * dphi_test_dxi_q * (2.0/dx_phys)
                        # Need to multiply by weight_q and J_vol
                        volume_integrand = - phi_trial_q_vals * dphi_test_deta_q_vals * (2.0/dt_phys) \
                                           - self.a * phi_trial_q_vals * dphi_test_dxi_q_vals * (2.0/dx_phys)

                        A[global_row_idx, global_col_idx] += \
                            np.sum(volume_integrand * quad_weights) * J_vol


                    # --- Face Integral Contributions ---

                    # Temporal Top Face (eta = 1): + ∫_{top} u_h psi_l dS_t
                    # (Integral for test function psi_l over top face of element (i,j), coupled with trial function phi_m from element (i,j))
                    # dS_t = (dx_phys/2) dxi
                    for trial_idx_local in range(n_nodes_per_element):
                         global_col_idx = elem_idx_global * n_nodes_per_element + trial_idx_local
                         trial_i = trial_idx_local % n_nodes_1d
                         trial_j = trial_idx_local // n_nodes_1d

                         # Evaluate basis functions at quadrature points on the top face (eta=1)
                         phi_trial_q_vals_top = self.lagrange_basis_2d(face_quad_points_1d, 1.0, trial_i, trial_j) # phi_m evaluated at eta=1
                         phi_test_q_vals_top = self.lagrange_basis_2d(face_quad_points_1d, 1.0, test_i, test_j) # psi_l evaluated at eta=1

                         # Sum over 1D quadrature points for the face integral
                         # Term: phi_trial_q * phi_test_q
                         # Need to multiply by weight_q_xi and J_face_t
                         face_top_integrand = phi_trial_q_vals_top * phi_test_q_vals_top

                         A[global_row_idx, global_col_idx] += \
                             np.sum(face_top_integrand * face_quad_weights_1d) * J_face_t # J_face_t = dx_phys/2.0


                    # Spatial Right Face (xi = 1): + ∫_{right} a u_h psi_l dS_x
                    # (Integral for test function psi_l over right face of element (i,j), coupled with trial function phi_m from element (i,j))
                    # dS_x = (dt_phys/2) deta
                    for trial_idx_local in range(n_nodes_per_element):
                         global_col_idx = elem_idx_global * n_nodes_per_element + trial_idx_local
                         trial_i = trial_idx_local % n_nodes_1d
                         trial_j = trial_idx_local // n_nodes_1d

                         # Evaluate basis functions at quadrature points on the right face (xi=1)
                         phi_trial_q_vals_right = self.lagrange_basis_2d(1.0, face_quad_points_1d, trial_i, trial_j) # phi_m evaluated at xi=1
                         phi_test_q_vals_right = self.lagrange_basis_2d(1.0, face_quad_points_1d, test_i, test_j) # psi_l evaluated at xi=1

                         # Sum over 1D quadrature points for the face integral
                         # Term: a * phi_trial_q * phi_test_q
                         # Need to multiply by weight_q_eta and J_face_x
                         face_right_integrand = self.a * phi_trial_q_vals_right * phi_test_q_vals_right

                         A[global_row_idx, global_col_idx] += \
                             np.sum(face_right_integrand * face_quad_weights_1d) * J_face_x # J_face_x = dt_phys/2.0


                    # --- Face Integral Contributions (from neighbor/IC trial functions) ---

                    # Temporal Bottom Face (eta = -1): + ∫_{bottom} (-u^*) psi_l dS_t
                    # dS_t = (dx_phys/2) dxi
                    if i == 0: # Initial Condition at t=0 (bottom face of element (0,j))
                        # u^* = u_init(x) at t=0. Term: ∫_{bottom} (-u_init) psi_l dS_t
                        # This is a known term on the RHS: + ∫_{bottom} u_init psi_l dS_t.
                        # Evaluate u_init at quadrature points on the bottom face (map xi to physical x)
                        x_left = self.x_elements[j]
                        x_right = self.x_elements[j+1]
                        dx_phys = x_right - x_left
                        x_q_phys = 0.5 * dx_phys * (face_quad_points_1d + 1) + x_left

                        u_init_q_vals = f_init(x_q_phys, self.L)

                        # Evaluate test function psi_l at quadrature points on the bottom face (eta=-1)
                        phi_test_q_vals_bottom = self.lagrange_basis_2d(face_quad_points_1d, -1.0, test_i, test_j) # psi_l evaluated at eta=-1

                        # Sum over 1D quadrature points for the integral
                        # Term for RHS: u_init_q * phi_test_q
                        # Need to multiply by weight_q_xi and J_face_t
                        rhs_integrand = u_init_q_vals * phi_test_q_vals_bottom

                        # Contribution to b (RHS)
                        b[global_row_idx] += \
                            np.sum(rhs_integrand * face_quad_weights_1d) * J_face_t # J_face_t = dx_phys/2.0


                    else: # i > 0, inflow from element (i-1, j) Top face (eta=1 of the neighbor)
                        # u^* = u_{i-1,j}(xi, 1). Term: ∫_{bottom} (-u_{i-1,j}(xi,1)) psi_l(xi,-1) dS_t.
                        # This involves trial DOFs from element (i-1, j).
                        neighbor_elem_idx_global = (i - 1) * Nx + j # Neighbor is directly below

                        for trial_idx_neighbor_local in range(n_nodes_per_element):
                            # Global column index for trial function from the *neighbor* element
                            global_col_idx_neighbor = neighbor_elem_idx_global * n_nodes_per_element + trial_idx_neighbor_local

                            # Map local 1D neighbor trial index to 2D indices
                            trial_i_neighbor = trial_idx_neighbor_local % n_nodes_1d
                            trial_j_neighbor = trial_idx_neighbor_local // n_nodes_1d

                            # Evaluate neighbor trial basis at its top face (eta=1) at quadrature points
                            phi_trial_neighbor_q_vals_top = self.lagrange_basis_2d(face_quad_points_1d, 1.0, trial_i_neighbor, trial_j_neighbor)

                            # Evaluate current test function psi_l at current element's bottom face (eta=-1) at quadrature points
                            phi_test_q_vals_bottom = self.lagrange_basis_2d(face_quad_points_1d, -1.0, test_i, test_j) # psi_l evaluated at eta=-1

                            # Sum over 1D quadrature points for the integral
                            # Term for LHS matrix A: (-1.0) * phi_trial_neighbor_q * phi_test_q
                            # Need to multiply by weight_q_xi and J_face_t
                            face_bottom_integrand = (-1.0) * phi_trial_neighbor_q_vals_top * phi_test_q_vals_bottom

                            # Add to A[test_dof_current_elem, trial_dof_neighbor_elem]
                            A[global_row_idx, global_col_idx_neighbor] += \
                               np.sum(face_bottom_integrand * face_quad_weights_1d) * J_face_t # J_face_t = dx_phys/2.0


                    # Spatial Left Face (xi = -1): ∫_{left} -a u^* psi_l dS_x
                    # dS_x = (dt_phys/2) deta
                    if j == 0: # Left spatial boundary, neighbor is element (i, Nx-1) due to periodic BC
                        neighbor_elem_idx_global = i * Nx + (Nx - 1)
                    else: # Interior left face, neighbor is element (i, j-1)
                        neighbor_elem_idx_global = i * Nx + (j - 1)

                    for trial_idx_neighbor_local in range(n_nodes_per_element):
                        # Global column index for trial function from the *neighbor* element
                        global_col_idx_neighbor = neighbor_elem_idx_global * n_nodes_per_element + trial_idx_neighbor_local

                        # Map local 1D neighbor trial index to 2D indices
                        trial_i_neighbor = trial_idx_neighbor_local % n_nodes_1d
                        trial_j_neighbor = trial_idx_neighbor_local // n_nodes_1d

                        # Evaluate neighbor trial basis at its right face (xi=1) at quadrature points
                        phi_trial_neighbor_q_vals_right = self.lagrange_basis_2d(1.0, face_quad_points_1d, trial_i_neighbor, trial_j_neighbor)

                        # Evaluate current test function psi_l at current element's left face (xi=-1) at quadrature points
                        phi_test_q_vals_left = self.lagrange_basis_2d(-1.0, face_quad_points_1d, test_i, test_j) # psi_l evaluated at xi=-1

                        # Sum over 1D quadrature points for the integral
                        # Term for LHS matrix A: (-self.a) * phi_trial_neighbor_q * phi_test_q
                        # Need to multiply by weight_q_eta and J_face_x
                        face_left_integrand = (-self.a) * phi_trial_neighbor_q_vals_right * phi_test_q_vals_left

                        # Add to A[test_dof_current_elem, trial_dof_neighbor_elem]
                        A[global_row_idx, global_col_idx_neighbor] += \
                            np.sum(face_left_integrand * face_quad_weights_1d) * J_face_x # J_face_x = dt_phys/2.0

        # Convert LIL matrix to CSR format for efficient solving
        # CHANGED: Convert to CSR format before solving
        A = A.tocsr()

        print("Assembly complete.")
        # --- NEW: Print Global System Matrix A and RHS vector b ---
        print("\n--- Global System Matrix (A) ---")
        print(f"Matrix size: {A.shape[0]}x{A.shape[1]} (sparse, CSR format)")
        # Print a representation of the sparse matrix. Default print shows non-zero entries.
        if A.shape[0] <= 20:
             print(A)
        else:
             print(f"Matrix is too large ({A.shape[0]}x{A.shape[1]}) to print in full. Showing a snippet of non-zero entries:")
             # Sparse matrix printing usually shows (row, col) value
             print(A[:min(A.shape[0], 30), :min(A.shape[1], 30)]) # Print top-left corner non-zeros up to a limit
             print("...")

        print("\n--- Global Right-Hand Side Vector (b) ---")
        print(f"Vector size: {b.shape[0]}")
        # Print snippets for large vectors
        if b.shape[0] <= 20:
            print(b)
        else:
            print(f"Vector is too large ({b.shape[0]}) to print in full. Showing snippet: {b[0:10]} ... {b[-10:]}")
        print("------------------------------------------")
        # --- End NEW ---

        return A, b

    def solve(self, f_init):
        """
        Assemble and solve the linear system Ax = b using a sparse solver.

        Parameters:
        -----------
        f_init : function
            Initial condition function
        """
        # --- Set poly_order to 1 and update element counts as specified in main ---
        # The __init__ now takes separate counts, so we use those.
        # The value from __init__ is used here.
        # However, the note in the prompt specifically says "poly_order p=1",
        # and the comment "Note: The poly_order is fixed to 1 inside the solve method." suggests this logic should remain.
        # Let's adhere to the prompt's implied structure by ensuring p=1 is used here.
        # We'll allow the __init__ to take any p, but force p=1 for the actual solve execution as per comment.
        # THIS MIGHT BE A SOURCE OF CONFUSION IF THE USER PROVIDES p != 1 IN __init__.
        # Sticking strictly to the prompt's comment: force p=1 here.
        # A better design would use the p from __init__.
        # Assuming the comment "The poly_order is fixed to 1 inside the solve method" is directive for this version.
        original_p = self.p # Save original p from __init__
        self.p = 1 # Force p=1 as per comment
        self.n_nodes_1d = self.p + 1
        self.n_nodes_per_element = self.n_nodes_1d ** 2
        # Total DOFs depend on Nx, Nt, and the new p=1.
        self.n_dofs = self.n_elements_spatial * self.n_elements_temporal * self.n_nodes_per_element

        print(f"Forcing polynomial order p=1 as specified in solve method logic.")
        print(f"Solving system with {self.n_elements_spatial} spatial elements and {self.n_elements_temporal} temporal elements for p=1.")
        print(f"Total DOFs: {self.n_dofs}")


        # Recreate mesh with updated p=1
        self.create_mesh()

        A, b = self.assemble_system(f_init)
        print("Solving the linear system using spsolve...")
        # Solve the system Au = b using a sparse solver
        # CHANGED: Use scipy.sparse.linalg.spsolve instead of numpy.linalg.solve
        try:
            # spsolve raises RuntimeError or ValueError on failure
            self.u = spsolve(A, b)
            if self.u is None: # spsolve can return None if factorization fails
                raise RuntimeError("Sparse solver spsolve returned None.")
            print("System solved successfully.")

            # --- NEW: Print Global Solution Vector U ---
            print("\n--- Global Solution Vector (U) ---")
            print(f"Vector size: {self.u.shape[0]}")
            # Print snippets for large vectors
            if self.u.shape[0] <= 20:
                print(self.u)
            else:
                 print(f"Vector is too large ({self.u.shape[0]}) to print in full. Showing snippet: {self.u[0:10]} ... {self.u[-10:]}")
            print("------------------------------------")
            # --- End NEW ---


        except (RuntimeError, ValueError) as e:
            print(f"Error solving linear system with spsolve: {e}")
            print("The sparse matrix might be singular or ill-conditioned.")
            print("Check assembly process, boundary conditions, and problem setup.")
            print("Consider using an iterative solver (e.g., gmres) with preconditioning for larger/harder problems.")
            self.u = None # Ensure self.u is None on failure
        except Exception as e:
            # Catch any other unexpected exceptions from spsolve
            print(f"An unexpected error occurred during spsolve: {e}")
            import traceback # Added traceback for debugging unexpected errors
            traceback.print_exc()
            self.u = None

        # Note: The original comment about restoring 'original_p' is skipped
        # as the rest of the code relies on the p value set by the solve method.


    def get_solution_at_time_level(self, time_step_idx):
        """
        Extract and evaluate the numerical solution at a specific discrete time level t = t_elements[time_step_idx].
        This corresponds to the solution on the top face of the elements in time row time_step_idx - 1
        (for time_step_idx > 0) or the initial condition projection (for time_step_idx = 0).
        Evaluates at the spatial GLL nodes for each element.

        Parameters:
        -----------
        time_step_idx : int
            Index of the time level from self.t_elements (0 to n_elements_temporal)

        Returns:
        --------
        tuple
            (x_plot, u_plot) numpy arrays, sorted by x-coordinate, evaluated at GLL nodes.
            Returns (None, None) if self.u is not available (e.g., solve failed).
        """
        # This function only extracts and evaluates. It shouldn't print
        # "Solution vector self.u is not available" if it's just called
        # during animation setup when u might be None initially.
        # The caller should handle the None check.
        if self.u is None:
             # print("Warning: Solution vector self.u is not available in get_solution_at_time_level.") # Suppress to avoid noise
             return None, None


        if not (0 <= time_step_idx <= self.n_elements_temporal):
            # Added check for time_step_idx > self.n_elements_temporal
            if time_step_idx > self.n_elements_temporal:
                 print(f"Error: time_step_idx ({time_step_idx}) must be <= n_elements_temporal ({self.n_elements_temporal})")
                 return None, None # Return None on invalid index

            raise ValueError(f"time_step_idx ({time_step_idx}) must be between 0 and {self.n_elements_temporal}")


        Nx = self.n_elements_spatial
        Nt = self.n_elements_temporal
        p = self.p
        n_nodes_1d = self.n_nodes_1d
        n_nodes_per_element = self.n_nodes_per_element

        x_plot_list = []
        u_plot_list = []

        # Evaluate solution at spatial GLL nodes for each element at the specified time level
        if time_step_idx == 0:
            # For t=0, we evaluate the solution on the bottom face of the first time row elements (i=0).
            # This corresponds to the projection of the IC onto the ST DG space on the bottom face.
            i_elem_row = 0 # Time element row 0
            eta_eval = -1.0
        else:
            # Get data from the top face (eta=1) of the corresponding time row elements (i = time_step_idx - 1)
            i_elem_row = time_step_idx - 1 # Time element row (0 to Nt-1)
            eta_eval = 1.0

        # Spatial GLL nodes are the same for all elements in a spatial column in the reference element [-1, 1]
        x_nodes_1d_ref = self.ref_nodes_1d

        # Reshape the solution vector to match the element and node structure
        # This assumes self.u is not None, which is checked at the start.
        # u_vals_reshaped = self.u.reshape(Nt, Nx, n_nodes_per_element) # No longer reshape global u directly

        for j_elem_col in range(Nx): # Spatial element index (0 to Nx-1)
            # Physical x-coordinates of the spatial GLL nodes for this element column
            # These correspond to mapping ref_nodes_1d for xi to physical x in element [x_j, x_{j+1}]
            x_nodes_1d_phys = 0.5 * (self.x_elements[j_elem_col+1] - self.x_elements[j_elem_col]) * (x_nodes_1d_ref + 1) + self.x_elements[j_elem_col]

            # Evaluate the local DG polynomial u_h at the spatial GLL nodes (ref_nodes_1d) on the specified temporal face (eta_eval).
            # We need the local coefficients for element (i_elem_row, j_elem_col).
            # The evaluation point is (xi_k, eta_eval) for k=0..p.
            xi_eval_points = x_nodes_1d_ref # Evaluating at the spatial reference nodes
            eta_eval_points = np.full_like(xi_eval_points, eta_eval) # At the chosen temporal face (eta_eval = -1 or 1)

            # Use the helper function to evaluate the polynomial
            u_vals_at_time = self.evaluate_element_solution(i_elem_row, j_elem_col, xi_eval_points, eta_eval_points)

            x_plot_list.extend(x_nodes_1d_phys)
            u_plot_list.extend(u_vals_at_time)

        # Convert lists to numpy arrays
        x_plot = np.array(x_plot_list)
        u_plot = np.array(u_plot_list)

        # Sort by x-coordinate, as nodes from different elements are combined
        sort_indices = np.argsort(x_plot)
        x_plot = x_plot[sort_indices]
        u_plot = u_plot[sort_indices]

        return x_plot, u_plot


    # --- ADDED: Helper method to get dense solution slice at a time level ---
    # This method is kept separate for potential future use (e.g., smoother animation)
    # but will not be used in the comparison plot to match the desired 'o-' style at nodes.
    def get_dense_solution_slice(self, time_step_idx, points_per_element_1d=10):
        """
        Extract and evaluate the numerical solution at a specific discrete time level t = t_elements[time_step_idx].
        Evaluates on a dense grid of points within each spatial element for smoother plotting.

        Parameters:
        -----------
        time_step_idx : int
            Index of the time level from self.t_elements (0 to n_elements_temporal)
        points_per_element_1d : int
            Number of evaluation points per spatial element in 1D (xi direction).
            Total points per element slice will be this number.

        Returns:
        --------
        tuple
            (x_plot, u_plot) numpy arrays, sorted by x-coordinate, evaluated on the dense grid.
            Returns (None, None) if self.u is not available (e.g., solve failed).
        """
        if self.u is None:
             # print("Warning: Solution vector self.u is not available in get_dense_solution_slice.") # Suppress to avoid noise
             return None, None

        if not (0 <= time_step_idx <= self.n_elements_temporal):
             # Added check for time_step_idx > self.n_elements_temporal
            if time_step_idx > self.n_elements_temporal:
                 print(f"Error: time_step_idx ({time_step_idx}) must be <= n_elements_temporal ({self.n_elements_temporal})")
                 return None, None # Return None on invalid index

            raise ValueError(f"time_step_idx ({time_step_idx}) must be between 0 and {self.n_elements_temporal}")

        Nx = self.n_elements_spatial
        Nt = self.n_elements_temporal

        x_plot_list = []
        u_plot_list = []

        # Time element row and eta evaluation point
        if time_step_idx == 0:
            i_elem_row = 0
            eta_eval = -1.0
        else:
            i_elem_row = time_step_idx - 1
            eta_eval = 1.0

        # Generate dense evaluation points in the reference xi direction
        # Add endpoints explicitly to avoid potential floating point issues near element boundaries
        xi_eval_points = np.linspace(-1.0, 1.0, points_per_element_1d)
        eta_eval_points = np.full_like(xi_eval_points, eta_eval)

        for j_elem_col in range(Nx): # Spatial element index (0 to Nx-1)
            # Map dense reference xi points to physical x in element [x_j, x_{j+1}]
            x_left = self.x_elements[j_elem_col]
            x_right = self.x_elements[j_elem_col+1]
            dx_phys = x_right - x_left
            x_dense_phys = 0.5 * dx_phys * (xi_eval_points + 1) + x_left

            # Evaluate the local polynomial u_h at these dense points (xi, eta_eval)
            u_vals_at_time = self.evaluate_element_solution(i_elem_row, j_elem_col, xi_eval_points, eta_eval_points)

            x_plot_list.extend(x_dense_phys)
            u_plot_list.extend(u_vals_at_time)

        # Convert lists to numpy arrays
        x_plot = np.array(x_plot_list)
        u_plot = np.array(u_plot_list)

        # Sort by x-coordinate (should already be sorted if processing elements left-to-right)
        # but sorting is robust if element processing order changes or nodes overlap slightly.
        sort_indices = np.argsort(x_plot)
        x_plot = x_plot[sort_indices]
        u_plot = u_plot[sort_indices]

        return x_plot, u_plot


    # --- plot_solution method (kept for static plots) ---
    def plot_solution(self, time_step_idx, f_init, plot_dir=".", show_plot=True):
        """
        Plots the numerical and exact solutions at a specific time level.
        Uses the GLL nodes for plotting the numerical solution.

        Parameters:
        -----------
        time_step_idx : int
            Index of the time level from self.t_elements (0 to n_elements_temporal) to plot.
        f_init : function
            Initial condition function used for the exact solution.
        plot_dir : str, optional
            Directory to save the plot. Defaults to current directory.
        show_plot : bool, optional
            Whether to display the plot using plt.show(). Defaults to True.
        """
        # Check if time_step_idx is valid
        if not (0 <= time_step_idx <= self.n_elements_temporal):
             print(f"Error: Invalid time_step_idx ({time_step_idx}) provided for plot_solution. Must be between 0 and {self.n_elements_temporal}.")
             return

        if self.u is None and time_step_idx > 0: # If t>0 and solve failed, numerical data is unavailable
             print("Cannot plot numerical solution as it is not available (solve failed?).")
             # Can still plot initial and exact solution at t=0 if needed, but the function assumes self.u exists for t > 0.
             # Let's handle the case where self.u is None but time_step_idx == 0 - this should ideally work
             if time_step_idx == 0:
                  # get_solution_at_time_level checks for self.u None internally
                  x_plot_num, u_plot_num = self.get_solution_at_time_level(time_step_idx) # Try to get it anyway
                  if x_plot_num is None: # If it failed even for t=0 (e.g. mesh creation failed)
                       if show_plot: plt.show() # Show empty plot if requested
                       else: plt.close()
                       return
             else: # t > 0 and self.u is None
                  x_plot_num, u_plot_num = None, None # Explicitly set to None


        else: # self.u is not None (solve succeeded) or time_step_idx == 0
             x_plot_num, u_plot_num = self.get_solution_at_time_level(time_step_idx)

             # Check again if getting solution at time level returned None (e.g. invalid index or other issue)
             if x_plot_num is None and time_step_idx > 0:
                  print(f"Warning: Could not retrieve numerical solution data at time step index {time_step_idx} for plotting.")
                  if show_plot: plt.show() # Show empty plot if requested
                  else: plt.close()
                  return # Cannot plot anything if x data is None

        t_plot = self.t_elements[time_step_idx]

        print(f"Generating plot at time t = {t_plot:.4f} (Time Step Index: {time_step_idx})")

        # Get exact solution
        x_exact_plot = np.linspace(0, self.L, 200) # Use more points for smooth exact solution plot
        u_exact_plot = exact_solution(x_exact_plot, t_plot, self.L, self.a, f_init)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot numerical solution - using markers at node points
        if x_plot_num is not None and u_plot_num is not None: # Only plot numerical if available
             # Add check for NaNs in numerical solution before plotting
             if np.any(np.isnan(u_plot_num)) or np.any(np.isinf(u_plot_num)):
                  print(f"Warning: Numerical solution at t={t_plot:.4f} contains NaN/Inf, skipping numerical plot line.")
                  # Still plot exact if available
             else:
                  ax.plot(x_plot_num, u_plot_num, 'o-', label=f'Numerical (p={self.p})', fillstyle='none', markersize=6)


        # Plot exact solution
        if np.any(np.isfinite(u_exact_plot)): # Only plot exact if finite values exist
             ax.plot(x_exact_plot, u_exact_plot, 'k--', label='Exact')
        else:
             print(f"Warning: Exact solution at t={t_plot:.4f} contains NaN/Inf, skipping exact plot line.")


        ax.set_title(f'Solution at t = {t_plot:.4f}', fontsize=ftSz1)
        ax.set_xlabel('x', fontsize=ftSz2)
        ax.set_ylabel('u(x, t)', fontsize=ftSz2)
        ax.legend(fontsize=ftSz2)
        ax.grid(True)
        ax.set_xlim(0, self.L)
        # Set y limits based on expected range (e.g., for sine wave)
        ax.set_ylim(-1.1, 1.1) # Sine wave range is [-1, 1]

        # Save plot (only if plot_dir is provided and not None/empty string)
        if plot_dir and plot_dir != ".": # Allow "." but avoid saving if plot_dir="" or None
             if not os.path.exists(plot_dir):
                 os.makedirs(plot_dir)
             plot_filename = os.path.join(plot_dir, f'solution_t_{time_step_idx:03d}.png')
             plt.savefig(plot_filename, bbox_inches='tight')
             print(f"Plot saved to {plot_filename}")

        # Display plot
        if show_plot:
            plt.show()
        else:
             plt.close(fig) # Close figure if not showing to free memory


    # --- Method to plot the 3D surface ---
    def plot_surface_3d(self, plot_dir="."):
        """
        Plots the numerical solution u(x,t) as a 3D surface.
        Uses the physical node coordinates and the corresponding solution coefficients.

        Parameters:
        -----------
        plot_dir : str, optional
            Directory to save the plot. Defaults to current directory.
        """
        if self.u is None:
             print("Cannot generate 3D surface plot: Solution vector self.u is not available (solve failed?).")
             return

        print("Generating 3D surface plot of u(x,t)...")

        # For Lagrange basis, the solution value at a node is the coefficient for that node's basis function.
        # The physical coordinates of all nodes are stored in self.x_nodes and self.t_nodes.
        # The coefficients (which are the values at these nodes) are in self.u.

        # Reshape the solution vector to match the element and node structure
        # u_vals_reshaped = self.u.reshape(self.n_elements_temporal, self.n_elements_spatial, self.n_nodes_per_element) # Cannot reshape if self.u is None

        # Flatten the node coordinates and solution values for plotting
        # Ensure node coordinates exist even if solve failed
        if self.x_nodes is None or self.t_nodes is None or self.u is None:
             print("Error: Node coordinates or solution data not available for 3D plot.")
             return

        x_flat = self.x_nodes.flatten()
        t_flat = self.t_nodes.flatten()
        u_flat = self.u.flatten() # Use the flattened solution vector directly

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use plot_trisurf with flattened nodes and values
        # Using plot_trisurf is suitable for scattered data points like the physical nodes.
        # Use antialiased=False to potentially speed up rendering for large meshes
        # Ensure u_flat does not contain NaNs or Infs if evaluate_element_solution could produce them
        if np.any(np.isnan(u_flat)) or np.any(np.isinf(u_flat)):
             print("Warning: Solution data contains NaN or Inf values, skipping 3D plot.")
             plt.close(fig) # Close the figure
             return

        try: # Add try-except around the plotting call itself
            surf = ax.plot_trisurf(x_flat, t_flat, u_flat, cmap=plt.cm.viridis, linewidth=0, antialiased=False)
            # Can also use plot_surface if the nodes form a structured grid, but plot_trisurf is more general.
        except Exception as e:
            print(f"Error during 3D plotting: {e}")
            print("This might happen with certain data structures or matplotlib versions.")
            plt.close(fig)
            return

        # Add labels and title
        ax.set_xlabel('x', fontsize=ftSz2)
        ax.set_ylabel('t', fontsize=ftSz2)
        ax.set_zlabel('u(x, t)', fontsize=ftSz2)
        ax.set_title('Numerical Solution u(x,t) Surface (Space-Time DG, P=1)', fontsize=ftSz1)

        # Add a color bar for the surface
        fig.colorbar(surf, ax=ax, label='u')

        # Set axis limits
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.T)
        # Set z limits based on expected range (e.g., for sine wave)
        ax.set_zlim(-1.1, 1.1)

        # Set view angle
        ax.view_init(elev=30, azim=-135)

        # Save plot (only if plot_dir is provided and not None/empty string)
        if plot_dir and plot_dir != ".":
             if not os.path.exists(plot_dir):
                 os.makedirs(plot_dir)
             plot_filename_3d = os.path.join(plot_dir, 'solution_3d_surface.png')
             plt.savefig(plot_filename_3d, bbox_inches='tight')
             print(f"3D surface plot saved to {plot_filename_3d}")


        # Display the plot
        plt.show()


    # --- ADDED: Method to create live animation ---
    def live_animate_solution(self, f_init, interval_ms=50):
        """
        Generates a live animation of the solution u(x,t) over time using Matplotlib animation.

        Parameters:
        -----------
        f_init : function
            Initial condition function (needed for exact solution).
        interval_ms : int, optional
            Delay between frames in milliseconds. Defaults to 50ms.
        """
        if self.u is None:
             print("Cannot create live animation: Solution vector self.u is not available (solve failed?).")
             return

        print("Creating live animation...")

        # Get initial data for plotting lines and setting limits
        # Use GLL nodes for animation (as used in plot_solution)
        x_plot_num_initial, u_plot_num_initial = self.get_solution_at_time_level(0)

        if x_plot_num_initial is None:
             print("Cannot create live animation: Failed to get initial solution data.")
             return

        # Setup the figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot initial state to create line objects
        line_num, = ax.plot(x_plot_num_initial, u_plot_num_initial, 'o-', label=f'Numerical (p={self.p})', fillstyle='none', markersize=6)

        # Get x-coordinates for the smooth exact solution plot once
        x_exact_plot = np.linspace(0, self.L, 200)
        # Plot initial exact solution to create the exact line object
        u_exact_plot_initial = exact_solution(x_exact_plot, self.t_elements[0], self.L, self.a, f_init)
        line_exact, = ax.plot(x_exact_plot, u_exact_plot_initial, 'k--', label='Exact')


        # Set plot limits and labels
        ax.set_xlim(0, self.L)
        ax.set_ylim(-1.1, 1.1) # Set y limits based on expected range (e.g., for sine wave)
        ax.set_xlabel('x', fontsize=ftSz2)
        ax.set_ylabel('u(x, t)', fontsize=ftSz2)
        ax.legend(fontsize=ftSz2)
        ax.grid(True)
        title = ax.set_title(f'Solution at t = {self.t_elements[0]:.4f}', fontsize=ftSz1)


        # Animation update function
        def animate(frame_idx):
            """Update function for the animation."""
            # frame_idx corresponds to the time step index from 0 to Nt
            if frame_idx > self.n_elements_temporal: # Safety check
                 return line_num, line_exact, title

            t_current = self.t_elements[frame_idx]

            # Get numerical solution data for the current time step (using GLL nodes)
            # We reuse the x_plot_num_initial as the x-coordinates don't change (at GLL nodes)
            x_num_current, u_plot_num_current = self.get_solution_at_time_level(frame_idx)

            # Get exact solution data for the current time step
            u_exact_plot_current = exact_solution(x_exact_plot, t_current, self.L, self.a, f_init)

            # Update the data for the plot lines
            if u_plot_num_current is not None:
                 # Check for NaNs/Infs before updating data
                 if np.any(np.isnan(u_plot_num_current)) or np.any(np.isinf(u_plot_num_current)):
                      # If NaNs detected, perhaps set ydata to NaN to show a gap or stop animating?
                      # For simplicity, just skip updating this frame's numerical line if invalid.
                      pass # line_num keeps its previous data
                 else:
                      line_num.set_ydata(u_plot_num_current)
            # Note: If numerical solution failed (u_plot_num_current is None), the line_num will keep its last valid data.
            # You could hide it if needed: line_num.set_visible(u_plot_num_current is not None)

            line_exact.set_ydata(u_exact_plot_current)

            # Update the title
            title.set_text(f'Solution at t = {t_current:.4f}')

            # Return the objects that were modified
            return line_num, line_exact, title # Return artist objects being updated


        # Create the animation object
        # frames: Sequence of values passed to the 'animate' function. Use range for time step indices.
        # blit=True means only redraw the parts of the plot that have changed. Improves performance.
        # repeat=False means the animation stops after the last frame.
        anim = animation.FuncAnimation(
            fig,                 # The figure to draw into
            animate,             # The function to call each frame
            frames=self.n_elements_temporal + 1, # Number of frames (from t=0 to t=T inclusive)
            interval=interval_ms, # Delay between frames in milliseconds
            blit=True,           # Use blitting for faster drawing (requires returning changed artists)
            repeat=False         # Don't repeat the animation
        )

        # Display the animation
        print("Displaying live animation...")
        plt.show()
        print("Live animation finished.")


    # --- ADDED: Helper method to evaluate solution inside an element ---
    def evaluate_element_solution(self, i_elem_row, j_elem_col, xi_points, eta_points):
        """
        Evaluates the DG polynomial solution u_h within a specific element (i, j)
        at given reference coordinates (xi, eta).

        Parameters:
        -----------
        i_elem_row : int
            Temporal element index (0 to Nt-1)
        j_elem_col : int
            Spatial element index (0 to Nx-1)
        xi_points : ndarray
            Array of xi coordinates in reference element [-1, 1]
        eta_points : ndarray
            Array of eta coordinates in reference element [-1, 1]

        Returns:
        --------
        ndarray
            Array of u_h values evaluated at the given points.
            Returns array of np.nan if self.u is not available or indices are invalid.
        """
        if self.u is None:
            # This method should ideally only be called if solve was successful,
            # but defensive check is good.
            # print("Warning: Solution vector self.u is not available in evaluate_element_solution.") # Avoid excessive printing in loops
            return np.full(len(xi_points), np.nan) # Return NaN for invalid results

        Nt = self.n_elements_temporal
        Nx = self.n_elements_spatial
        n_nodes_per_element = self.n_nodes_per_element
        n_nodes_1d = self.n_nodes_1d

        # Check for valid element indices
        if not (0 <= i_elem_row < Nt) or not (0 <= j_elem_col < Nx):
             print(f"Error: Invalid element indices ({i_elem_row}, {j_elem_col}) in evaluate_element_solution. Expected (0..{Nt-1}, 0..{Nx-1}).")
             return np.full(len(xi_points), np.nan)


        # Get local solution coefficients for the specified element
        elem_global_idx = i_elem_row * Nx + j_elem_col
        # Access coefficients from the global solution vector self.u
        # The coefficients for element (i, j) are at indices from elem_global_idx * n_nodes_per_element
        # up to (elem_global_idx + 1) * n_nodes_per_element - 1
        start_idx = elem_global_idx * n_nodes_per_element
        end_idx = start_idx + n_nodes_per_element

        if end_idx > len(self.u):
             print(f"Error: Solution vector size mismatch in evaluate_element_solution. Expected at least {end_idx} DOFs, but self.u has size {len(self.u)}.")
             return np.full(len(xi_points), np.nan)

        u_local_coeffs = self.u[start_idx : end_idx]

        # Initialize result array
        u_eval = np.zeros_like(xi_points, dtype=float)

        # Sum contributions from each local basis function
        for local_dof_idx in range(n_nodes_per_element):
            # Map local 1D index to 2D indices (spatial, temporal)
            # local_dof_idx maps to phi_{local_dof_idx%n_nodes_1d, local_dof_idx//n_nodes_1d}(xi, eta)
            trial_i = local_dof_idx % n_nodes_1d # spatial index of basis L_{trial_i}(xi)
            trial_j = local_dof_idx // n_nodes_1d # temporal index of basis L_{trial_j}(eta)

            # Evaluate the basis function at the given reference points
            try: # Add try-except around basis evaluation
                 phi_m_eval = self.lagrange_basis_2d(xi_points, eta_points, trial_i, trial_j)
            except Exception as e:
                 print(f"Error evaluating basis function L_{trial_i}L_{trial_j} for element ({i_elem_row}, {j_elem_col}) at points: {e}")
                 return np.full(len(xi_points), np.nan) # Return NaN on basis evaluation error


            # Add contribution: coefficient * basis_value
            u_eval += u_local_coeffs[local_dof_idx] * phi_m_eval

        return u_eval


    # --- ADDED: Method to calculate L2 error ---
    def calculate_l2_error(self, f_init):
        """
        Calculates the L2 norm of the error between the numerical solution u_h
        and the exact solution u_exact at the final time T.

        ||u_h - u_exact||_{L2} = sqrt( sum_{j=0}^{Nx-1} integral_{x_j}^{x_{j+1}} (u_h(x, T) - u_exact(x, T))^2 dx )
        The integral is computed using Gauss-Legendre quadrature within each spatial element.

        Parameters:
        -----------
        f_init : function
            Initial condition function used for the exact solution.

        Returns:
        --------
        float
            The L2 error at the final time T. Returns np.nan if solve failed.
        """
        if self.u is None:
             print("Warning: Solution vector self.u is not available. Cannot calculate L2 error.")
             return np.nan

        # print(f"Calculating L2 error at final time T = {self.T:.4f}...") # Moved print inside loop for better visibility per case

        Nx = self.n_elements_spatial
        # Use the last time row (index Nt-1) which corresponds to the solution near T=T
        i_elem_row_for_T = self.n_elements_temporal - 1
        t_eval = self.T # Evaluation time for exact solution

        # Quadrature for spatial integral over each element [x_j, x_{j+1}]
        # Integrand is (u_h - u_exact)^2. u_h is degree p=1 in space. (u_h)^2 is degree 2*p=2.
        # Need enough points to integrate polynomial of degree 2*p.
        # Gauss-Legendre with N points is exact for degree 2N-1.
        # To integrate degree 2p, we need 2N-1 >= 2p => 2N > 2p => N > p.
        # Using p+2 for safety: N=3 for p=1. GL(3) is exact for 2*3-1=5. Integrand degree for p=1 is 2*1=2.
        # N=p+1 points are sufficient for degree 2p-1 (volume integrals),
        # but (u_h-u_exact)^2 might be higher depending on u_exact.
        # For smooth problems, p+1 is usually okay, p+2 is safer.
        n_quad_1d_error = self.p + 2 # Using p+2 Gauss-Legendre points for error calculation
        quad_points_1d, quad_weights_1d = roots_legendre(n_quad_1d_error)

        l2_error_sq = 0.0

        # Loop over spatial elements
        for j_elem_col in range(Nx):
            x_left = self.x_elements[j_elem_col]
            x_right = self.x_elements[j_elem_col+1]
            dx_phys = x_right - x_left
            J_spatial_quad = dx_phys / 2.0 # Jacobian for the 1D spatial mapping [-1, 1] -> [x_left, x_right]

            # Evaluate the numerical solution u_h and the exact solution u_exact
            # at the quadrature points within the current spatial element [x_left, x_right]
            # at the fixed time T (which corresponds to eta=1 in the last time row element)

            # Map 1D spatial quadrature points (xi_q) to physical x-coordinates
            x_quad_phys = 0.5 * dx_phys * (quad_points_1d + 1) + x_left

            # All quadrature points are at the top face of the last time element row (eta=1)
            eta_quad_ref = np.full_like(quad_points_1d, 1.0)

            # Evaluate numerical solution u_h at the quadrature points (x_q, T)
            # Use the evaluate_element_solution helper method
            # Need to make sure i_elem_row calculation is correct for the last time step T
            # t_elements has Nt+1 entries, 0 to Nt. T is at index Nt.
            # The solution at T corresponds to the top face (eta=1) of elements in time row Nt-1.
            # So i_elem_row should indeed be Nt-1.
            if self.n_elements_temporal == 0:
                 # Cannot calculate error if Nt=0 (no time elements)
                 print("Warning: Cannot calculate L2 error for Nt=0.")
                 return np.nan
            i_elem_row_for_T = self.n_elements_temporal - 1
            u_h_quad = self.evaluate_element_solution(i_elem_row_for_T, j_elem_col, quad_points_1d, eta_quad_ref)

            # Check if u_h_quad contains NaN or Inf before using it
            if np.any(np.isnan(u_h_quad)) or np.any(np.isinf(u_h_quad)):
                 print(f"Warning: Numerical solution contains NaN/Inf in element ({i_elem_row_for_T}, {j_elem_col}) at T={t_eval:.4f}. Skipping L2 error calculation for this case.")
                 return np.nan # Return NaN for the entire L2 error calculation


            # Evaluate exact solution u_exact at the quadrature points (x_q, T)
            u_exact_quad = exact_solution(x_quad_phys, t_eval, self.L, self.a, f_init)

            # Sum the squared difference (u_h - u_exact)^2 weighted by quadrature weights and Jacobian
            integrand_sq = (u_h_quad - u_exact_quad)**2
            # Check for NaNs/Infs in the integrand before summing
            if np.any(np.isnan(integrand_sq)) or np.any(np.isinf(integrand_sq)):
                 print(f"Warning: L2 error integrand contains NaN/Inf in element ({i_elem_row_for_T}, {j_elem_col}) at T={t_eval:.4f}. Skipping L2 error calculation for this case.")
                 return np.nan # Return NaN for the entire L2 error calculation


            l2_error_sq += np.sum(integrand_sq * quad_weights_1d) * J_spatial_quad

        # Final check before sqrt
        if l2_error_sq < -1e-14 or not np.isfinite(l2_error_sq):
            print(f"Warning: Invalid L2 error sum ({l2_error_sq:.3e}) before sqrt. Returning NaN.")
            return np.nan
        elif l2_error_sq < 0: # Clamp small negative values due to FP errors
            l2_error_sq = 0.0

        l2_error = np.sqrt(l2_error_sq)
        # print(f"L2 Error at T={self.T:.4f} for Nx={Nx}, Nt={self.n_elements_temporal}: {l2_error:.6e}") # Moved print inside loop
        return l2_error


# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    # These parameters are for the initial run with plots and animation
    n_elements_spatial_initial = 20  # Number of spatial elements for initial run
    n_elements_temporal_initial = 40 # Number of temporal elements for initial run (to ensure stability for P=1)
    # Note: The solve method forces p=1 as per original comment.
    # Setting poly_order here affects initialization details but solve() overrides p for execution.
    poly_order_init = 1 # This value is overridden by solve() to 1
    domain_length = 1.0
    advection_speed = 1.0
    t_final = 1.0            # Final time

    print("--- Initial Simulation Run ---")
    print(f"Space-Time DG for u_t + {advection_speed} u_x = 0")
    print(f"Domain: [0, {domain_length}], Time: [0, {t_final}]")
    print(f"Initial Run Parameters:")
    print(f"Spatial Elements: {n_elements_spatial_initial}, Temporal Elements: {n_elements_temporal_initial}")
    # The actual polynomial order will be 1 due to the solve method's internal logic
    print(f"Poly Order (forced in solve): 1")
    dx_val = domain_length / n_elements_spatial_initial
    dt_val = t_final / n_elements_temporal_initial
    print(f"dx = {dx_val:.4f}, dt = {dt_val:.4f}")
    print(f"a * dt / dx (for initial run) = {advection_speed * dt_val / dx_val:.4f}")


    # --- Initialize Solver for Initial Run ---
    # Note: The poly_order is set in the __init__ but then confirmed/forced to 1 in solve().
    solver_initial = SpaceTimeDG(
        n_elements_spatial=n_elements_spatial_initial,
        n_elements_temporal=n_elements_temporal_initial,
        poly_order=poly_order_init, # This value is overridden by solve()
        domain_length=domain_length,
        advection_speed=advection_speed,
        t_final=t_final
    )

    # --- Define Initial Condition ---
    f_init = ic_sine_wave # Use the sine wave IC function

    # --- Solve the System for Initial Run ---
    # Note: The assemble_system method has been updated to implement
    # a more standard STDG weak form with upwind fluxes and uses separate element counts.
    # The solve method proceeds using the matrix assembled by assemble_system.
    # The poly_order is fixed to 1 inside the solve method.
    solver_initial.solve(f_init)

    # --- Plot Results for Initial Run ---
    if solver_initial.u is not None: # Only plot if solve was successful
        # Static plots (optional, can comment out if only animation is desired)
        plot_dir_static = "plots_stdg_fixed_stability_temporal_coupling_static" # Directory for static plots
        os.makedirs(plot_dir_static, exist_ok=True) # Ensure directory exists for saving

        # Plot initial condition (t=0) - saves and shows
        # Note: This plot at t=0 shows the projection of the IC onto the DG space.
        solver_initial.plot_solution(0, f_init, plot_dir=plot_dir_static, show_plot=True)

        # Plot final solution (t=T) - saves and shows
        solver_initial.plot_solution(solver_initial.n_elements_temporal, f_init, plot_dir=plot_dir_static, show_plot=True)

        # --- Plot the 3D surface ---
        solver_initial.plot_surface_3d(plot_dir=plot_dir_static)

        # --- ADDED: Create LIVE Animation ---
        # This will create a separate Matplotlib window and display the animation
        solver_initial.live_animate_solution(f_init, interval_ms=75) # Adjust interval as desired (ms per frame)

        print("\nStatic plots saved in:", os.path.abspath(plot_dir_static))
        print("Initial simulation run and outputs finished.")

    else:
        print("\nSkipping initial run plots and animation because the system solve failed.")

    print("\n--- Convergence Study (h-convergence, P=1 fixed) ---")
    # Define a sequence of resolutions (Nx, Nt) to test for convergence
    # Keeping Nt/Nx constant (=2) as in the original setup ensures fixed CFL_ST ~ a * dt / dx
    # Start from a coarse mesh and refine
    resolutions = [(10, 20), (20, 40), (40, 80), (80, 160)] # Example resolutions

    # Store data for the convergence plot and table
    nx_values = []
    h_values = []
    error_values = []
    # Store rates separately as they are computed
    computed_rates = []
    # Store solution data at GLL nodes for comparison plot
    gll_solutions_for_plotting = []

    # Define poly_order for the convergence study solvers (will be forced to 1)
    poly_order_conv = 1

    prev_error = None
    prev_h = None

    for Nx_conv, Nt_conv in resolutions:
        print(f"\nRunning convergence case: Nx={Nx_conv}, Nt={Nt_conv}, p=1")
        dx_conv = domain_length / Nx_conv
        dt_conv = t_final / Nt_conv
        print(f"dx = {dx_conv:.4f}, dt = {dt_conv:.4f}")
        print(f"a * dt / dx = {advection_speed * dt_conv / dx_conv:.4f}")


        # --- Initialize Solver for Convergence Study ---
        # A new solver instance is needed for each resolution
        solver_conv = SpaceTimeDG(
            n_elements_spatial=Nx_conv,
            n_elements_temporal=Nt_conv,
            poly_order=poly_order_conv, # This value is overridden by solve()
            domain_length=domain_length,
            advection_speed=advection_speed,
            t_final=t_final
        )

        # --- Solve the System for Convergence Study ---
        # NOTE: The assemble_system and solve methods will print A, b, and U here.
        solver_conv.solve(f_init)

        # --- Calculate and Store Error ---
        if solver_conv.u is not None: # Only calculate error if solve was successful
            error = solver_conv.calculate_l2_error(f_init)
            if not np.isnan(error): # Store error only if calculation was successful (solve worked and error calculation found no NaNs)
                h = domain_length / Nx_conv # Characteristic mesh size h = L/Nx
                                            # Using spatial h as the primary refinement parameter
                nx_values.append(Nx_conv)
                h_values.append(h)
                error_values.append(error)

                # Calculate approximate rate if we have a previous point
                if prev_error is not None and prev_h is not None:
                    # Rate r = log(error_new / error_old) / log(h_new / h_old)
                    # Ensure h_new != h_old and error_old != 0 for log
                    if h != prev_h and prev_error != 0:
                         # Add a small epsilon to error to avoid log(0) if error is machine zero
                         rate = math.log((error + 1e-16) / (prev_error + 1e-16)) / math.log(h / prev_h)
                         computed_rates.append(rate)
                    else:
                         # Handle cases where h didn't change or previous error was zero (perfect solution?)
                         # In a proper convergence study, h should always decrease, and error>0
                         # If error is truly 0, rate is effectively infinite (perfect solution)
                         computed_rates.append(float('inf'))
                # First point has no previous rate, handle this by computed_rates list size check later

                # Update previous values for the next iteration
                prev_error = error
                prev_h = h

                # --- Store Solution Data for Comparison Plot ---
                # We need to store the *final* solution evaluated at GLL nodes for this plot
                x_plot_num_gll, u_plot_num_gll = solver_conv.get_solution_at_time_level(solver_conv.n_elements_temporal)
                if x_plot_num_gll is not None:
                     # Check for NaNs/Infs in the retrieved plot data before storing
                     if np.any(np.isnan(x_plot_num_gll)) or np.any(np.isinf(x_plot_num_gll)) or \
                        np.any(np.isnan(u_plot_num_gll)) or np.any(np.isinf(u_plot_num_gll)):
                          print(f"Warning: GLL final solution data for Nx={Nx_conv} contains NaN/Inf after retrieval. Skipping storage for comparison plot.")
                     else:
                         gll_solutions_for_plotting.append({
                             'Nx': Nx_conv,
                             'x_plot': x_plot_num_gll,
                             'u_plot': u_plot_num_gll
                         })
                else:
                  print(f"Warning: GLL final solution data for Nx={Nx_conv} could not be retrieved for comparison plot.")


            else:
                 print(f"Skipping error calculation and solution storage for Nx={Nx_conv}, Nt={Nt_conv} due to failed solve or NaN error.")
                 # If a solve or error calculation failed, clear previous values to ensure rate calculation is sequential across *successful* runs
                 prev_error = None
                 prev_h = None
                 # Do NOT append to nx_values, h_values, error_values, gll_solutions_for_plotting if solve failed

        else:
             print(f"Skipping error calculation and solution storage for Nx={Nx_conv}, Nt={Nt_conv} because solve failed.")
             # If solve failed, clear previous values
             prev_error = None
             prev_h = None


    # --- Print Convergence Table ---
    if nx_values and error_values:
        print("\n--- Convergence Table (P=1 fixed) ---")
        # Header
        print(f"{'Nx':<8} {'h':<12} {'L2 Error':<15} {'Approx Rate':<15}")
        print("-" * 50)
        # Data rows
        for i in range(len(nx_values)):
            nx = nx_values[i]
            h = h_values[i]
            error = error_values[i]

            # Get the rate for this row (it's the rate *from* the previous row *to* this row)
            # The first row doesn't have a preceding rate, and computed_rates has len=len(nx_values)-1
            if i > 0 and i - 1 < len(computed_rates):
                rate = computed_rates[i-1]
                # Format rate, handle potential infinity if error became 0 (unlikely here)
                rate_str = f'{rate:<15.4f}' if np.isfinite(rate) else f"{'Inf':<15}"
            else:
                rate_str = f"{'N/A':<15}"

            print(f"{nx:<8} {h:<12.4e} {error:<15.6e} {rate_str}")
        print("-" * 50)
    else:
        print("\nNo data available to print convergence table (no successful solves).")

    # --- Plot Solution Comparison ---
    if gll_solutions_for_plotting: # Check the list containing GLL node data
        print("\nGenerating comparison plot of numerical solutions at T...")
        plt.figure(figsize=(10, 6))
        ax = plt.gca() # Get current axes

        # Plot exact solution once
        x_exact_plot = np.linspace(0, domain_length, 200)
        u_exact_plot_T = exact_solution(x_exact_plot, t_final, domain_length, advection_speed, f_init)
        ax.plot(x_exact_plot, u_exact_plot_T, 'k--', label='Exact', linewidth=2) # Increased linewidth for clarity

        # Plot numerical solutions from convergence study (GLL nodes)
        # Sort solutions by Nx for consistent plotting order (coarsest to finest)
        gll_solutions_for_plotting.sort(key=lambda item: item['Nx'])

        # Define distinct styles (color, marker, linestyle) for each resolution
        # Using common matplotlib markers and colors
        markers = ['o', 's', 'D', '^', 'v', '*', 'X', 'P'] # Cycle through markers
        colors = plt.cm.tab10.colors # Use a colormap for colors
        linestyles = ['-', '--', '-.', ':'] # Cycle through linestyles

        # Adjust base markersize for better visibility if needed
        base_markersize = 6

        for i, sol_data in enumerate(gll_solutions_for_plotting):
            nx = sol_data['Nx']
            # Assign style using modulo for cycling
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]

            # Plot using the stored GLL node data
            # Use fillstyle='none' for open markers on refinement steps, matching the image
            fillstyle = 'full' if i == 0 else 'none'

            ax.plot(sol_data['x_plot'], sol_data['u_plot'],
                    marker=marker, linestyle=linestyle, color=color,
                    label=f'Numerical (Nx={nx})', fillstyle=fillstyle, markersize=base_markersize)


        # Set plot properties
        ax.set_title(f'Comparison of Numerical Solutions at t = {t_final:.4f} (P=1)', fontsize=ftSz1)
        ax.set_xlabel('x', fontsize=ftSz2)
        ax.set_ylabel('u(x, t)', fontsize=ftSz2)
        ax.legend(fontsize=ftSz3) # Use slightly smaller font for multiple labels
        ax.grid(True)
        ax.set_xlim(0, domain_length)
        ax.set_ylim(-1.1, 1.1)

        plt.show()
        print("Comparison plot finished.")
    else:
        print("\nNo numerical solutions collected for comparison plot.")


    # --- Plot Convergence ---
    if h_values and error_values:
        print("\nGenerating convergence plot...")
        plt.figure(figsize=(10, 6))

        # Plot the computed error points
        plt.loglog(h_values, error_values, 'o-', label='Computed $L_2$ Error')

        # Add a reference line for the expected convergence rate p+1
        # For p=1, expected rate is 2
        if len(h_values) > 1:
            # Pick the coarsest resolution point among the *valid* h_values as a reference for the slope line
            # Get the index of the largest h (coarsest mesh) in the valid set
            idx_coarsest_valid = np.argsort(h_values)[-1] # Index for coarsest h
            h_ref = h_values[idx_coarsest_valid]
            error_ref = error_values[idx_coarsest_valid]


            # Plot line over the range of computed h (from smallest to largest h)
            h_plot_ref = np.array([min(h_values), max(h_values)]) # Plot line over the range of computed h
            # Get the actual p used in the solver instances (which is 1)
            # This relies on the last successful solver instance's p value.
            # Since p is forced to 1 inside solve(), this should be 1.
            rate_expected = 1 + 1 # Expected rate is p+1 = 1+1 = 2 for L2 error with P=1 basis
            # Ensure error_ref is not zero or too small for log scale calculation
            if error_ref > 1e-16:
                 C_ref = error_ref / (h_ref**rate_expected)
                 error_plot_ref = C_ref * h_plot_ref**rate_expected
                 plt.loglog(h_plot_ref, error_plot_ref, 'k--', label=f'Order $O(h^{rate_expected})$ Reference')
            else:
                 print("Warning: Reference error is zero or too small, skipping order reference line.")


        plt.title(f'Convergence Plot ($L_2$ Error at T={t_final:.1f}, P=1)', fontsize=ftSz1)
        plt.xlabel('Mesh Size $h = L/N_x$', fontsize=ftSz2)
        plt.ylabel('$||u_h - u_{exact}||_{L_2}$', fontsize=ftSz2)
        plt.grid(True, which="both", ls="--")
        plt.gca().invert_xaxis() # Show finer meshes (smaller h) to the right

        # Add legend and potentially axis limits
        plt.legend(fontsize=ftSz2)
        # Optional: Adjust plot limits for better visibility
        # plt.xlim([min(h_values)/2, max(h_values)*2])
        # plt.ylim([min(error_values)/10, max(error_values)*10])

        plt.show()

        print("\nConvergence study finished.")

    else:
        print("\nNo error values collected for convergence plot (all solves might have failed).")

    print("\nScript execution finished.")
