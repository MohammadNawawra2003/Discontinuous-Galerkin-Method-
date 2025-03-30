import numpy as np
from scipy.special import roots_legendre, jacobi # For edge quadrature and potential basis
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# Optional: Consider using quadpy for robust triangle quadrature
# try:
#     import quadpy
# except ImportError:
#     print("Warning: quadpy library not found. Using placeholder quadrature.")
#     quadpy = None

# --- Constants and Parameters ---
P_ORDER = 3 # Polynomial degree as specified
N_P = (P_ORDER + 1) * (P_ORDER + 2) // 2 # Number of basis functions/nodes for P=3 -> 10

# --- (Code Task 1.1) Reference Element Setup ---
# We use the reference triangle K_ref with vertices (0,0), (1,0), (0,1) in (xi, eta)

def basis_functions(xi, eta, p_order=P_ORDER):
    """
    Evaluates the P_ORDER-th degree basis functions at reference point (xi, eta).

    Args:
        xi (float): Reference coordinate xi.
        eta (float): Reference coordinate eta.
        p_order (int): Polynomial order.

    Returns:
        np.ndarray: Array of shape (N_P,) containing basis function values phi_j(xi, eta).

    NOTE: This is a PLACEHOLDER. Replace with actual basis function implementation
          (e.g., using Jacobi polynomials on transformed coordinates as in Hesthaven
           & Warburton Ch 7, or a nodal basis if using specific node locations).
          For P=3, N_P = 10.
    """
    n_p = (p_order + 1) * (p_order + 2) // 2
    # Check if point is approximately within the reference triangle
    if not (-1e-9 <= xi <= 1.0 + 1e-9 and -1e-9 <= eta <= 1.0 + 1e-9 and xi + eta <= 1.0 + 1e-9):
         # print(f"Warning: Point ({xi}, {eta}) potentially outside reference triangle.")
         # Depending on quadrature point generation, points might be exactly on boundary
         pass # Allow points very close to or on the boundary

    # --- !!! REPLACE THIS PLACEHOLDER IMPLEMENTATION !!! ---
    # Example using monomials (NOT ORTHOGONAL - just for structure)
    vals = np.zeros(n_p)
    count = 0
    for i in range(p_order + 1):
        for j in range(p_order - i + 1):
            if count < n_p:
                # Handle potential 0^0 case, define as 1
                term_xi = 1.0 if i == 0 and abs(xi) < 1e-15 else xi**i
                term_eta = 1.0 if j == 0 and abs(eta) < 1e-15 else eta**j
                vals[count] = term_xi * term_eta
                count += 1
            else: break # Should not happen if n_p is correct
    # For a real implementation, use orthogonal polynomials (e.g., Koornwinder, Dubiner)
    # or a nodal basis evaluated at (xi, eta).
    # --- !!! END PLACEHOLDER !!! ---

    # Normalize or ensure properties if using a specific basis type
    # E.g., for nodal: vals[j] = 1.0 if (xi, eta) is node j, else 0.0 (requires node coords)
    if count != n_p:
        raise ValueError(f"Basis function count mismatch: expected {n_p}, got {count}")
    return vals


def basis_gradients(xi, eta, p_order=P_ORDER):
    """
    Evaluates the gradients (d/dxi, d/deta) of the basis functions at (xi, eta).

    Args:
        xi (float): Reference coordinate xi.
        eta (float): Reference coordinate eta.
        p_order (int): Polynomial order.

    Returns:
        np.ndarray: Array of shape (N_P, 2) containing [dphi_j/dxi, dphi_j/deta].

    NOTE: This is a PLACEHOLDER. Replace with gradients of your chosen basis.
    """
    n_p = (p_order + 1) * (p_order + 2) // 2
    if not (-1e-9 <= xi <= 1.0 + 1e-9 and -1e-9 <= eta <= 1.0 + 1e-9 and xi + eta <= 1.0 + 1e-9):
         #print(f"Warning: Point ({xi}, {eta}) potentially outside reference triangle.")
         pass

    grads = np.zeros((n_p, 2))
    # --- !!! REPLACE THIS PLACEHOLDER IMPLEMENTATION !!! ---
    # Example using monomial gradients
    count = 0
    for i in range(p_order + 1):
        for j in range(p_order - i + 1):
            if count < n_p:
                # Handle potential 0^negative cases for derivatives
                term_xi_dxi = 0.0 if i == 0 else i * (xi**(i - 1) if i > 1 else 1.0)
                term_eta_dxi = 1.0 if j == 0 and abs(eta) < 1e-15 else eta**j

                term_xi_deta = 1.0 if i == 0 and abs(xi) < 1e-15 else xi**i
                term_eta_deta = 0.0 if j == 0 else j * (eta**(j - 1) if j > 1 else 1.0)

                # d/dxi
                grads[count, 0] = term_xi_dxi * term_eta_dxi
                # d/deta
                grads[count, 1] = term_xi_deta * term_eta_deta
                count += 1
            else: break
    # --- !!! END PLACEHOLDER !!! ---
    if count != n_p:
        raise ValueError(f"Basis gradient count mismatch: expected {n_p}, got {count}")
    return grads

# --- (Code Task 1.3) Quadrature ---

def get_volume_quadrature(degree):
    """
    Provides 2D quadrature points (xi, eta) and weights (w) for the reference triangle
    vertices (0,0), (1,0), (0,1), sufficient to integrate polynomials of the given degree exactly.

    Args:
        degree (int): The maximum polynomial degree to integrate exactly.

    Returns:
        tuple: (xi_coords, eta_coords, weights)
               - xi_coords (np.ndarray): xi coordinates of quadrature points.
               - eta_coords (np.ndarray): eta coordinates of quadrature points.
               - weights (np.ndarray): Quadrature weights (scaled to sum to 0.5 - area).

    NOTE: Using placeholder - Recommend using 'quadpy' for reliable points/weights.
          `scheme = quadpy.t2.get_good_scheme(degree)`
          `points = scheme.points` # Shape (N, 2) -> xi=points[:,0], eta=points[:,1]
          `weights = scheme.weights * 0.5` # Scale quadpy weights
    """
    # --- !!! REPLACE THIS PLACEHOLDER IMPLEMENTATION with quadpy or known rules !!! ---
    if degree <= 1: # Degree 1 rule (Midpoint rule, 3 points) - inaccurate for p=3
        print(f"Warning: Using low-degree quadrature (1) for requested degree {degree}.")
        weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0]) # Sums to 0.5 already
        xi_coords = np.array([0.5, 0.0, 0.5]) # Corrected midpoints: (0.5,0), (0,0.5), (0.5,0.5) -> need check
        eta_coords = np.array([0.0, 0.5, 0.5])
        # Corrected Midpoints: (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)
        xi_coords = np.array([0.5, 0.5, 0.0])
        eta_coords = np.array([0.0, 0.5, 0.5])
        return xi_coords, eta_coords, weights
    elif degree <=3: # Degree 3 rule (4 points) - still INSUFFICIENT for p=3 assembly (needs 2p=6)
         print(f"Warning: Using low-degree quadrature (3) for requested degree {degree}.")
         weights = np.array([ -27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0]) # Scale needed if sum != 0.5
         weights *= 0.5 / np.sum(weights) # Ensure sum is 0.5
         xi_coords  = np.array([1.0/3.0, 0.6, 0.2, 0.2])
         eta_coords = np.array([1.0/3.0, 0.2, 0.6, 0.2])
         return xi_coords, eta_coords, weights
    elif degree >= 6: # Placeholder for required degree
         print(f"Warning: Using placeholder quadrature (Degree 3) for required degree {degree}. Results inaccurate.")
         # REUSING DEGREE 3 RULE - REPLACE!
         weights = np.array([ -27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0])
         weights *= 0.5 / np.sum(weights)
         xi_coords  = np.array([1.0/3.0, 0.6, 0.2, 0.2])
         eta_coords = np.array([1.0/3.0, 0.2, 0.6, 0.2])
         return xi_coords, eta_coords, weights
    else:
         raise ValueError(f"Volume quadrature degree {degree} not implemented in placeholder")
    # --- !!! END PLACEHOLDER !!! ---


def get_edge_quadrature(degree):
    """
    Provides 1D Gauss-Legendre quadrature points (s) and weights (w)
    on the interval [-1, 1], sufficient to integrate polynomials of the given degree.

    Args:
        degree (int): The maximum polynomial degree to integrate exactly.

    Returns:
        tuple: (points, weights)
    """
    n_points = int(np.ceil((degree + 1) / 2))
    points, weights = roots_legendre(n_points)
    return points, weights # Defined on [-1, 1]


# --- (Code Task 1.2) Mapping ---

def get_element_mapping(verts):
    """
    Calculates Jacobian, inverse Jacobian, and determinant for the affine map
    from reference triangle K_ref((0,0),(1,0),(0,1)) to physical triangle K with vertices `verts`.

    Args:
        verts (np.ndarray): Array of shape (3, 2) with physical vertex coordinates
                            [[x0, t0], [x1, t1], [x2, t2]]. Order corresponds to ref vertices.

    Returns:
        tuple: (jacobian, inv_jacobian, det_jacobian)
    """
    v0, v1, v2 = verts[0, :], verts[1, :], verts[2, :] # Physical coords (Using v0,v1,v2 for clarity with formula)

    # Jacobian J = [[dx/dxi, dx/deta], [dt/dxi, dt/deta]]
    # Using map: P(xi,eta) = v0*(1-xi-eta) + v1*xi + v2*eta
    # dx/dxi = v1[0]-v0[0], dx/deta = v2[0]-v0[0]
    # dt/dxi = v1[1]-v0[1], dt/deta = v2[1]-v0[1]
    jacobian = np.array([
        [v1[0] - v0[0], v2[0] - v0[0]],
        [v1[1] - v0[1], v2[1] - v0[1]]
    ])

    det_jacobian = np.linalg.det(jacobian)
    # Check determinant sign - determines orientation. Should be positive if vertices ordered correctly.
    if abs(det_jacobian) < 1e-14:
        # Try calculating area using Shoelace formula as cross-check
        area = 0.5 * abs(v0[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-v1[1]))
        if area < 1e-14:
             raise ValueError(f"Degenerate element vertices: {verts}. Area is near zero.")
        else:
             print(f"Warning: Jacobian determinant {det_jacobian} is near zero for element {verts}. Area={area}")
             # Proceed cautiously, might indicate poor element shape

    # Inverse Jacobian using 2x2 formula: inv([[a,b],[c,d]]) = 1/det * [[d, -b], [-c, a]]
    inv_jacobian = (1.0 / det_jacobian) * np.array([
        [ jacobian[1, 1], -jacobian[0, 1]],
        [-jacobian[1, 0],  jacobian[0, 0]]
    ])

    return jacobian, inv_jacobian, det_jacobian


def map_gradients(dphi_dxi_eta, inv_jacobian):
    """
    Maps gradients from reference coordinates (xi, eta) to physical (x, t).

    Args:
        dphi_dxi_eta (np.ndarray): Gradients in ref coords, shape (N_P, 2).
                                     [[dphi/dxi, dphi/deta], ...]
        inv_jacobian (np.ndarray): Inverse Jacobian matrix (2, 2).

    Returns:
        np.ndarray: Gradients in physical coords, shape (N_P, 2).
                    [[dphi/dx, dphi/dt], ...]
    """
    # Formula: [d/dx, d/dt]^T = J_inv^T @ [d/dxi, d/deta]^T
    # For a scalar function phi: [dphi/dx, dphi/dt] = [dphi/dxi, dphi/deta] @ J_inv
    # Ensure correct shapes for matrix multiplication if dphi_dxi_eta is single grad (2,)
    if dphi_dxi_eta.ndim == 1:
        dphi_dxi_eta = dphi_dxi_eta.reshape(1, 2) # Make it a row vector for consistency
        dphi_dx_dt = dphi_dxi_eta @ inv_jacobian
        return dphi_dx_dt.flatten() # Return as a flat array (2,)
    else: # Assume shape (N_P, 2)
        dphi_dx_dt = dphi_dxi_eta @ inv_jacobian
        return dphi_dx_dt


# --- (Code Task 1.5) Mesh Connectivity (Conceptual Structure) ---

class Mesh:
    """Conceptual class to hold mesh information."""
    def __init__(self, vertices, elements, edges_info):
        """
        Args:
            vertices (np.ndarray): (num_verts, 2) array of vertex [x, t] coords.
            elements (np.ndarray): (num_elems, 3) array of vertex indices for each triangle.
            edges_info (dict): Dictionary mapping a unique edge identifier (e.g., tuple of sorted vertex indices)
                               to its details.
                 Example: edge_key=(min(v1,v2), max(v1,v2)) -> {
                     'vertices': (v_idx1, v_idx2), # Global vertex indices
                     'length': float,
                     'normal': np.array([nx, nt]), # Outward normal w.r.t elements[0] element
                     'elements': (elem_idx1, elem_idx2), # elem_idx2 = -1 for boundary
                     'local_indices': (local_edge_idx1, local_edge_idx2) # Edge index (0,1,2) within element
                 }
        """
        self.vertices = vertices
        self.elements = elements
        self.num_elements = elements.shape[0]
        self.edges_info = edges_info # This needs to be constructed carefully by mesh generator

        # Precompute element vertices for quick lookup
        self.element_vertices = np.array([vertices[el_verts] for el_verts in elements])

        # Precompute edge info per element for faster lookup during assembly
        self.element_edges = [[] for _ in range(self.num_elements)]
        if edges_info: # Check if edges_info is provided and not empty
            for edge_id, info in edges_info.items():
                el1_idx = info['elements'][0]
                loc1_idx = info['local_indices'][0]
                # Add edge info relative to the first element
                self.element_edges[el1_idx].append({
                    'edge_id': edge_id,
                    'local_idx': loc1_idx,
                    'normal': info['normal'], # Normal outward from el1
                    'length': info['length'],
                    'neighbor_element': info['elements'][1],
                    'neighbor_local_idx': info.get('local_indices', (None, None))[1] # Use get for safety
                })
                # If it's an internal edge, add info relative to the second element
                if info['elements'][1] != -1:
                    el2_idx = info['elements'][1]
                    loc2_idx = info.get('local_indices', (None, None))[1]
                    if loc2_idx is not None: # Ensure local index for neighbor exists
                         self.element_edges[el2_idx].append({
                            'edge_id': edge_id,
                            'local_idx': loc2_idx,
                            'normal': -info['normal'], # Normal outward from el2 (opposite)
                            'length': info['length'],
                            'neighbor_element': info['elements'][0],
                            'neighbor_local_idx': info['local_indices'][0]
                        })
                    else:
                        print(f"Warning: Missing local index for neighbor element {el2_idx} on edge {edge_id}")
        else:
             print("Warning: Mesh created without edge information.")


    def get_element_vertices(self, k):
        """Returns (3, 2) array of vertices for element k."""
        if k < 0 or k >= self.num_elements:
            raise IndexError(f"Element index {k} out of bounds (0 to {self.num_elements-1})")
        return self.element_vertices[k]

    def get_element_edge_info(self, k):
        """Returns a list of info dicts for the edges of element k."""
        if k < 0 or k >= self.num_elements:
            raise IndexError(f"Element index {k} out of bounds (0 to {self.num_elements-1})")
        if not self.element_edges[k]:
             print(f"Warning: No edge information available for element {k}")
        return self.element_edges[k]

# --- Numerical Flux ---

def lax_friedrichs_flux(u_minus, u_plus, normal, a):
    """
    Calculates the Local Lax-Friedrichs flux for 1D advection in space-time.

    Args:
        u_minus (float): Solution value inside the element.
        u_plus (float): Solution value outside the element (neighbor/BC).
        normal (np.ndarray): Outward unit normal vector [nx, nt], shape (2,).
        a (float): Advection speed.

    Returns:
        float: The numerical flux value.
    """
    nx, nt = normal[0], normal[1]
    F_minus_n = (a * nx + nt) * u_minus
    F_plus_n = (a * nx + nt) * u_plus

    # Stabilization parameter C = max |a*nx + nt|
    # For safety, we can overestimate C based on possible normals.
    # Or, calculate it precisely based on the specific edge normal.
    C_stab = abs(a * nx + nt) # Precise for this edge

    flux = 0.5 * (F_minus_n + F_plus_n) - 0.5 * C_stab * (u_plus - u_minus)
    return flux

# --- (Code Task 1.4) Local Matrix/Vector Assembly ---

def assemble_local_matrix(verts, a, p_order=P_ORDER):
    """
    Assembles the local matrix M^K for a single space-time element K.
    M_ij = ∫∫_K φ_j (a ∂φ_i/∂x + ∂φ_i/∂t) dx dt

    Args:
        verts (np.ndarray): Vertices of the physical element [[x0,t0],[x1,t1],[x2,t2]].
        a (float): Advection speed.
        p_order (int): Polynomial order.

    Returns:
        np.ndarray: Local matrix M^K, shape (N_P, N_P).
    """
    n_p = (p_order + 1) * (p_order + 2) // 2
    local_M = np.zeros((n_p, n_p))

    # Quadrature degree: need to integrate grad(phi_i)*phi_j -> deg (p-1)+p = 2p-1
    # Let's use a rule exact for degree 2*p_order for safety/generality
    quad_deg_vol = 2 * p_order
    try:
        xi_v, eta_v, w_v = get_volume_quadrature(quad_deg_vol)
    except ValueError as e:
        print(f"Error getting volume quadrature: {e}")
        return None # Or handle error appropriately
    num_quad_vol = len(w_v)

    # Mapping derivatives (constant for affine map)
    try:
        _, inv_jacobian, det_jacobian = get_element_mapping(verts)
    except ValueError as e:
        print(f"Error getting element mapping: {e}")
        return None # Or handle degenerate element

    for q in range(num_quad_vol):
        xi, eta, w = xi_v[q], eta_v[q], w_v[q]

        # Basis functions and gradients at quadrature point (reference coords)
        phi_vals = basis_functions(xi, eta, p_order)         # Shape (N_P,)
        dphi_dxi_eta = basis_gradients(xi, eta, p_order)     # Shape (N_P, 2)

        # Map gradients to physical coords
        dphi_dx_dt = map_gradients(dphi_dxi_eta, inv_jacobian) # Shape (N_P, 2)
        dphi_dx = dphi_dx_dt[:, 0]
        dphi_dt = dphi_dx_dt[:, 1]

        # Calculate integrand: phi_j * (a * dphi_i/dx + dphi_i/dt)
        # Using outer product: term_i = a * dphi_dx + dphi_dt
        # M_ij += term_i[i] * phi_vals[j] * w * det_jacobian
        term_i = a * dphi_dx + dphi_dt # Shape (N_P,)
        # integrand_matrix[i, j] = term_i[i] * phi_vals[j]
        integrand_matrix = np.outer(term_i, phi_vals) # Shape (N_P, N_P)

        local_M += integrand_matrix * w * det_jacobian # Weights sum to 0.5, detJ is 2*Area

    return local_M


def assemble_local_rhs(k_elem, verts, u_coeffs_k, mesh, U_global, a, p_order=P_ORDER):
    """
    Assembles the local right-hand side vector R^K for a single element K.
    R_i = ∫_{∂K} F̂_n(u_h^-, u_h^+) φ_i ds

    Args:
        k_elem (int): Index of the current element.
        verts (np.ndarray): Vertices of the physical element [[x0,t0],[x1,t1],[x2,t2]].
        u_coeffs_k (np.ndarray): Solution coefficients for element k, shape (N_P,).
        mesh (Mesh): Mesh object providing connectivity and geometry.
        U_global (np.ndarray): Global solution vector (e.g., shape (num_elems, N_P)).
                                Assumes accessible coefficients for neighbors.
        a (float): Advection speed.
        p_order (int): Polynomial order.

    Returns:
        np.ndarray: Local RHS vector R^K, shape (N_P,).
    """
    n_p = (p_order + 1) * (p_order + 2) // 2
    local_R = np.zeros(n_p)

    # Quadrature degree for edge: F_hat * phi_i -> approx deg p * deg p = 2p
    quad_deg_edge = 2 * p_order
    try:
        s_e, w_e = get_edge_quadrature(quad_deg_edge) # Points/weights on [-1, 1]
    except ValueError as e:
        print(f"Error getting edge quadrature: {e}")
        return None # Or handle error
    num_quad_edge = len(s_e)

    # Get info for the 3 edges of the current element
    edge_info_list = mesh.get_element_edge_info(k_elem)

    # Map reference triangle vertices (0,0), (1,0), (0,1) to edge indices
    # Edge 0: (0,0) -> (1,0) : xi varies 0 to 1, eta = 0
    # Edge 1: (1,0) -> (0,1) : xi = 1-s, eta = s (s from 0 to 1)
    # Edge 2: (0,1) -> (0,0) : xi = 0, eta = 1-s (s from 0 to 1)
    ref_edge_parametrization = [
        lambda s: (s, 0.0),         # Edge 0: Parameter s runs 0 to 1
        lambda s: (1.0 - s, s),     # Edge 1: Parameter s runs 0 to 1
        lambda s: (0.0, 1.0 - s)     # Edge 2: Parameter s runs 0 to 1
    ]

    for edge_info in edge_info_list:
        local_edge_idx = edge_info['local_idx'] # 0, 1, or 2
        normal = edge_info['normal']           # Outward normal for *this* element
        L_edge = edge_info['length']
        k_neighbor = edge_info['neighbor_element']

        # Get parametrization function for the current edge index
        get_ref_coords = ref_edge_parametrization[local_edge_idx]

        # Edge Jacobian for 1D integral on reference [-1, 1]: L_physical / L_reference = L_edge / 2.0
        edge_jacobian = L_edge / 2.0
        if abs(edge_jacobian) < 1e-14:
            print(f"Warning: Edge {k_elem},{local_edge_idx} has near-zero length {L_edge}. Skipping.")
            continue

        for q in range(num_quad_edge):
            sq, wq = s_e[q], w_e[q] # Point/weight on [-1, 1]
            s_param = (sq + 1.0) / 2.0 # Map Gauss point from [-1,1] to edge parameter [0,1]

            # Reference coords (xi, eta) on the edge using parametrization
            xi_q, eta_q = get_ref_coords(s_param)

            # Basis functions at the edge quadrature point in reference coords
            phi_vals_q = basis_functions(xi_q, eta_q, p_order) # Shape (N_P,)

            # Solution value from inside the current element ('minus' side)
            u_minus_q = np.dot(phi_vals_q, u_coeffs_k)

            # Solution value from the neighbor ('plus' side)
            if k_neighbor != -1:
                # Internal edge - get neighbor coefficients
                # Assuming U_global is structured [element, coeff_index]
                try:
                    u_coeffs_neighbor = U_global[k_neighbor, :]
                    # We need basis values *at the corresponding point* on the neighbor's edge
                    # This requires mapping (xi_q, eta_q) from current element's edge
                    # to the neighbor element's reference coordinates.
                    # For affine elements and straight edges, the reference basis function
                    # values *should* correspond IF the local node numbering aligns
                    # across the shared edge. This relies heavily on mesh generator & basis setup.
                    # ASSUMPTION: phi_vals_q calculated for k_elem edge applies to k_neighbor edge point too.
                    u_plus_q = np.dot(phi_vals_q, u_coeffs_neighbor)
                except IndexError:
                     print(f"Error accessing neighbor {k_neighbor} coefficients for element {k_elem}")
                     u_plus_q = u_minus_q # Fallback: Treat as boundary, needs review
                except Exception as e:
                     print(f"Error getting neighbor state: {e}")
                     u_plus_q = u_minus_q # Fallback

            else:
                # Boundary edge - apply boundary condition via flux
                # Requires specific BC implementation. Placeholder using zero inflow:
                char_speed_normal = a * normal[0] + normal[1]
                if char_speed_normal < -1e-9: # Characteristic points strictly inward
                    u_plus_q = 0.0 # Assume zero inflow state for u_plus
                else: # Characteristic points outward or tangential
                    u_plus_q = u_minus_q # Use internal state for u_plus (zero normal flux for BC)
                # print(f"BC Edge {k_elem},{local_edge_idx}, n=({normal[0]:.1f},{normal[1]:.1f}), char_spd={char_speed_normal:.1f}, u+={u_plus_q:.2f}")


            # Calculate numerical flux
            F_hat_q = lax_friedrichs_flux(u_minus_q, u_plus_q, normal, a)

            # Add contribution to RHS: Integral F_hat * phi_i * ds
            # ds = |J_edge| * d(s_e) where s_e is on [-1, 1]
            # Integral = Sum[ F_hat * phi_i * wq * |J_edge| ] over q points
            local_R += F_hat_q * phi_vals_q * wq * edge_jacobian

    return local_R

# --- Plotting Function ---
def plot_element_solution(verts, u_coeffs, p_order=P_ORDER, resolution=20):
    """
    Creates a contour plot of the DG solution within a single triangular element.

    Args:
        verts (np.ndarray): Vertices of the physical element [[x0,t0],[x1,t1],[x2,t2]].
        u_coeffs (np.ndarray): Solution coefficients for the element, shape (N_P,).
        p_order (int): Polynomial order.
        resolution (int): Number of points along each reference edge for plotting grid.
    """
    n_p = (p_order + 1) * (p_order + 2) // 2
    if len(u_coeffs) != n_p:
        raise ValueError(f"Incorrect number of coefficients. Expected {n_p}, got {len(u_coeffs)}")

    # 1. Create a grid of points in the reference triangle (xi, eta)
    xi_pts = np.linspace(0, 1, resolution)
    eta_pts = np.linspace(0, 1, resolution)
    xi_grid, eta_grid = np.meshgrid(xi_pts, eta_pts)

    # Filter points to be inside the reference triangle (xi + eta <= 1)
    mask = xi_grid + eta_grid <= 1.0 + 1e-9 # Add tolerance
    xi_flat = xi_grid[mask]
    eta_flat = eta_grid[mask]
    num_plot_pts = len(xi_flat)
    if num_plot_pts == 0:
        print("Warning: No points generated for plotting in reference triangle.")
        return

    # 2. Map reference points to physical (x, t)
    # Affine map: P(xi,eta) = v0*(1-xi-eta) + v1*xi + v2*eta
    v0, v1, v2 = verts[0, :], verts[1, :], verts[2, :]
    x_plot = v0[0] * (1 - xi_flat - eta_flat) + v1[0] * xi_flat + v2[0] * eta_flat
    t_plot = v0[1] * (1 - xi_flat - eta_flat) + v1[1] * xi_flat + v2[1] * eta_flat

    # 3. Evaluate the DG solution u_h at these points
    u_plot = np.zeros(num_plot_pts)
    for i in range(num_plot_pts):
        try:
            phi_vals_i = basis_functions(xi_flat[i], eta_flat[i], p_order) # Assumes basis_functions is defined
            u_plot[i] = np.dot(phi_vals_i, u_coeffs)
        except ValueError as e:
             print(f"Error evaluating basis/solution at ref pt ({xi_flat[i]}, {eta_flat[i]}): {e}")
             u_plot[i] = np.nan # Mark as invalid

    # 4. Create triangulation object for plotting
    try:
        # Create triangulation based on the reference points (more robust)
        triang_ref = tri.Triangulation(xi_flat, eta_flat)
        # Use this topology but with physical coordinates for plotting
        triang = tri.Triangulation(x_plot, t_plot, triangles=triang_ref.triangles)

    except Exception as e:
        print(f"Warning: Could not create plot triangulation: {e}")
        print("Plotting might fail or look incorrect.")
        # Fallback to scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_plot, t_plot, c=u_plot, cmap='viridis', s=10, vmin=np.nanmin(u_plot), vmax=np.nanmax(u_plot))
        plt.plot([verts[0,0], verts[1,0], verts[2,0], verts[0,0]],
                 [verts[0,1], verts[1,1], verts[2,1], verts[0,1]], 'r-', lw=1)
        plt.colorbar(label=f'u_h (P{p_order})')
        plt.xlabel('x (Space)')
        plt.ylabel('t (Time)')
        plt.title(f'DG Solution (Fallback Plot) in Element\nVertices: {verts.round(2).tolist()}')
        plt.axis('equal')
        plt.grid(True, linestyle=':')
        return

    # 5. Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Filter out potential NaN values before plotting
    valid_mask = ~np.isnan(u_plot)
    if np.any(valid_mask):
        contour = ax.tricontourf(triang, u_plot[valid_mask], cmap='viridis', levels=15) # Only plot valid data
        ax.triplot(triang, 'ko-', lw=0.5, markersize=2) # Show evaluation points/mesh
        fig.colorbar(contour, label=f'u_h (P{p_order})')
    else:
        print("Warning: No valid solution data to plot.")

    # Plot element boundary
    ax.plot([verts[0,0], verts[1,0], verts[2,0], verts[0,0]],
            [verts[0,1], verts[1,1], verts[2,1], verts[0,1]], 'r-', lw=2)

    # Add labels, title, colorbar
    ax.set_xlabel('x (Space)')
    ax.set_ylabel('t (Time)')
    ax.set_title(f'DG Solution (Random Coeffs) in Element\nVertices: {verts.round(2).tolist()}')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')

# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    print(f"STDG Setup: Polynomial Order P = {P_ORDER}, Num Basis Funcs N_P = {N_P}")

    # --- Create a dummy mesh (REPLACE with your mesh generator) ---
    # Example: 2 elements sharing an edge, plus boundary edges
    # Correct vertex ordering for reference map: v0=(0,0), v1=(1,0), v2=(0,1)
    # Element 0: Vertices (0,0), (0.5,0), (0.0,0.5) -> Indices 0, 1, 2
    # Element 1: Vertices (0.5,0), (0.5,0.5), (0.0,0.5) -> Map needs care. Let's use different vertices for clarity.
    # Element 1 Alt: Vertices (0.5,0), (1.0,0), (0.5,0.5) -> Indices 1, 3, 4 (if 3=(1,0), 4=(0.5,0.5))
    # Let's stick to the simple 2-element square split along diagonal
    dummy_verts = np.array([
        [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]
    ])
    # Elem 0 uses verts 0, 1, 2. Maps ref (0,0)->0, (1,0)->1, (0,1)->2
    # Elem 1 uses verts 1, 3, 2. Map ref (0,0)->1, (1,0)->3, (0,1)->2
    dummy_elems = np.array([
        [0, 1, 2], # Element 0: Lower left triangle
        [1, 3, 2]  # Element 1: Upper right triangle
    ])
    # Define edges based on VERTEX PAIRS (sorted tuple for uniqueness)
    # Need normals and neighbor info carefully defined.
    # Edge (0,1): Belongs to Elem 0 (local edge 0). Boundary. Normal [0, -1]
    # Edge (1,2): Belongs to Elem 0 (local edge 1), Elem 1 (local edge 2). Normal [1, 1]/sqrt(2) wrt Elem 0.
    # Edge (2,0): Belongs to Elem 0 (local edge 2). Boundary. Normal [-1, 0]
    # Edge (1,3): Belongs to Elem 1 (local edge 0). Boundary. Normal [1, 0]
    # Edge (3,2): Belongs to Elem 1 (local edge 1). Boundary. Normal [0, 1]
    # Local edges: 0 -> v0-v1, 1 -> v1-v2, 2 -> v2-v0 (counter-clockwise)
    dummy_edges = {
        (0, 1): {'vertices': (0, 1), 'length': 0.5, 'normal': np.array([0.0,-1.0]),
                 'elements': (0, -1), 'local_indices':(0, None)}, # Elem 0, Edge 0
        (1, 2): {'vertices': (1, 2), 'length': np.sqrt(0.5**2+0.5**2), 'normal': np.array([0.5,0.5])/np.sqrt(0.5**2+0.5**2), # Normal wrt Elem 0: (t2-t1, x1-x2) -> (0.5, 0.5)
                 'elements': (0, 1), 'local_indices':(1, 2)}, # Elem 0 Edge 1, Elem 1 Edge 2
        (0, 2): {'vertices': (2, 0), 'length': 0.5, 'normal': np.array([-1.0,0.0]), # Swapped vertices to match key (0,2)
                 'elements': (0, -1), 'local_indices':(2, None)}, # Elem 0, Edge 2
        (1, 3): {'vertices': (1, 3), 'length': 0.5, 'normal': np.array([0.0, -1.0]), # Normal wrt Elem 1: (t3-t1, x1-x3) -> (-0.5, 0.0)? No, this edge uses v1,v3,v2 map. v1=(0.5,0),v3=(0.5,0.5),v2=(0,0.5) WRONG
                 # Remap Elem 1: v0'=(0.5,0), v1'=(0.5,0.5), v2'=(0.0,0.5) - Uses global verts 1, 3, 2
                 # Edge 0': v0'-v1' = (1)-(3). Normal = (t3-t1, x1-x3) = (0.5, 0.0). Len=0.5. Boundary.
                 # Edge 1': v1'-v2' = (3)-(2). Normal = (t2-t3, x3-x2) = (0.0, 0.0)? No. (t2-t3, x3-x2) = (0.0, 0.0-0.5) = (0,-0.5). Len=0.5. Boundary.
                 # Edge 2': v2'-v0' = (2)-(1). Normal = (t1-t2, x2-x1) = (-0.5, -0.5). Len=sqrt(0.5). Internal. Matches -(normal of edge (1,2))
                 'normal': np.array([1.0, 0.0]), # Elem 1, Edge 0: Correct normal (0.5,0.5)-(0.5,0) -> [1,0] outward
                 'elements': (1, -1), 'local_indices':(0, None)},
        (2, 3): {'vertices': (3, 2), 'length': 0.5, 'normal': np.array([0.0, 1.0]), # Elem 1, Edge 1: (0.0,0.5)-(0.5,0.5) -> [0,1] outward
                 'elements': (1, -1), 'local_indices':(1, None)},
        # Edge (1, 2) handled above for Elem 1 as local edge 2
    }
    # Refined edge info needed for robust code
    # --- End Dummy Mesh ---
    try:
        mesh = Mesh(dummy_verts, dummy_elems, dummy_edges)
    except Exception as e:
        print(f"Error creating mesh object: {e}")
        exit()


    advection_speed = 1.0

    # Assume some global coefficient vector U exists
    try:
        U_dummy = np.random.rand(mesh.num_elements, N_P) # Shape (2, 10)
    except Exception as e:
        print(f"Error creating dummy U: {e}")
        exit()


    # --- Assemble and Plot for Element 0 ---
    k_test = 0
    try:
        verts_k = mesh.get_element_vertices(k_test)
        u_coeffs_k = U_dummy[k_test, :]
    except IndexError:
         print(f"Error: Cannot access element {k_test} data. Mesh definition likely incomplete.")
         exit()
    except Exception as e:
         print(f"Error getting element data: {e}")
         exit()


    print(f"\nAssembling for Element {k_test} with vertices:\n{verts_k}")

    local_M_k = assemble_local_matrix(verts_k, advection_speed, p_order=P_ORDER)
    if local_M_k is not None:
        print(f"\nLocal Matrix M^K (shape {local_M_k.shape}):\n{local_M_k[:3,:3]}...")

    print("\n assembling RHS requires proper neighbor lookup/BC")
    local_R_k = assemble_local_rhs(k_test, verts_k, u_coeffs_k, mesh, U_dummy, advection_speed, p_order=P_ORDER)
    if local_R_k is not None:
        print(f"\nLocal RHS R^K (shape {local_R_k.shape}):\n{local_R_k[:3]}...")

    # --- Add Plotting Call ---
    if local_R_k is not None: # Only plot if assembly seemed okay
        print("\nGenerating plot for element 0 with random coefficients...")
        try:
            plot_element_solution(verts_k, u_coeffs_k, p_order=P_ORDER)
            plt.show() # Display the plot
        except Exception as e:
            print(f"Error during plotting: {e}")
    # --- End Plotting Call ---

    # --- Next steps would be: ---
    # 1. Implement robust basis functions and quadrature.
    # 2. Implement robust mesh generation with FULL connectivity info.
    # 3. Loop through all elements to assemble GLOBAL M and R.
    # 4. Apply Initial Conditions (project initial state onto basis functions).
    # 5. Apply Boundary Conditions (modify global R, or handle in flux).
    # 6. Solve the global system M U = R.