import numpy as np
from scipy.special import roots_legendre, jacobi # Using SciPy for quadrature points
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# Consider using quadpy for more robust/varied triangle quadrature rules if needed
# import quadpy

# --- Configuration ---
P_ORDER = 3  # Polynomial degree for basis functions
N_DOF_PER_ELEMENT = (P_ORDER + 1) * (P_ORDER + 2) // 2  # Dofs per element (e.g., 10 for P=3)

# --- Reference Element Definition (Triangle: (0,0), (1,0), (0,1)) ---

def evaluate_reference_basis(xi, eta, p_order=P_ORDER):
    """
    Evaluates basis functions at a point (xi, eta) on the reference triangle.

    Args:
        xi (float): Reference coordinate (in [0, 1]).
        eta (float): Reference coordinate (in [0, 1]).
        p_order (int): Polynomial order.

    Returns:
        np.ndarray: Array of basis values, shape (N_DOF_PER_ELEMENT,).

    NOTE/TODO: This is a PLACEHOLDER using simple monomials (1, xi, eta, xi^2, ...).
               This is NOT an orthogonal or nodal basis. Replace with a proper
               implementation (e.g., using mapped Jacobi polynomials or a specific
               nodal basis set like warp-and-blend) for actual DG calculations.
    """
    num_dofs = (p_order + 1) * (p_order + 2) // 2
    # Check bounds with a small tolerance
    if not (-1e-9 <= xi <= 1.0 + 1e-9 and -1e-9 <= eta <= 1.0 + 1e-9 and xi + eta <= 1.0 + 1e-9):
        # Pass silently for now, could indicate issues with quadrature points
        pass

    basis_vals = np.zeros(num_dofs)
    current_dof = 0
    for degree_sum in range(p_order + 1):
        for i in range(degree_sum + 1):
            j = degree_sum - i
            if current_dof < num_dofs:
                # Use np.power for potentially better handling of 0^0 (though still check)
                term_xi = np.power(xi, i) if i > 0 else (1.0 if abs(xi) < 1e-15 and i == 0 else 1.0) # Handle 0^0=1
                term_eta = np.power(eta, j) if j > 0 else (1.0 if abs(eta) < 1e-15 and j == 0 else 1.0) # Handle 0^0=1
                basis_vals[current_dof] = term_xi * term_eta
                current_dof += 1
            else:
                # This break should technically not be needed if num_dofs is correct
                break
    if current_dof != num_dofs:
        raise RuntimeError(f"Internal logic error: Basis function count mismatch. Expected {num_dofs}, generated {current_dof}")
    return basis_vals

def evaluate_reference_basis_gradients(xi, eta, p_order=P_ORDER):
    """
    Evaluates gradients (d/dxi, d/deta) of basis functions at (xi, eta) on ref triangle.

    Args:
        xi (float): Reference coordinate xi.
        eta (float): Reference coordinate eta.
        p_order (int): Polynomial order.

    Returns:
        np.ndarray: Gradients array, shape (N_DOF_PER_ELEMENT, 2). Columns are [d/dxi, d/deta].

    NOTE/TODO: This is a PLACEHOLDER matching the monomial basis above.
               Replace with gradients corresponding to your chosen basis set.
    """
    num_dofs = (p_order + 1) * (p_order + 2) // 2
    if not (-1e-9 <= xi <= 1.0 + 1e-9 and -1e-9 <= eta <= 1.0 + 1e-9 and xi + eta <= 1.0 + 1e-9):
        pass # Allow points near boundary

    basis_grads = np.zeros((num_dofs, 2))
    current_dof = 0
    for degree_sum in range(p_order + 1):
        for i in range(degree_sum + 1):
            j = degree_sum - i
            if current_dof < num_dofs:
                # Pre-calculate powers carefully to handle 0^(negative) from derivative
                xi_pow_i = np.power(xi, i) if i > 0 else (1.0 if abs(xi) < 1e-15 and i == 0 else 1.0)
                eta_pow_j = np.power(eta, j) if j > 0 else (1.0 if abs(eta) < 1e-15 and j == 0 else 1.0)

                xi_pow_im1 = np.power(xi, i-1) if i > 1 else (1.0 if i == 1 else 0.0)
                eta_pow_jm1 = np.power(eta, j-1) if j > 1 else (1.0 if j == 1 else 0.0)

                # d/dxi (term is i * xi^(i-1) * eta^j)
                grad_xi = i * xi_pow_im1 * eta_pow_j if i > 0 else 0.0
                # d/deta (term is xi^i * j * eta^(j-1))
                grad_eta = xi_pow_i * j * eta_pow_jm1 if j > 0 else 0.0

                basis_grads[current_dof, 0] = grad_xi
                basis_grads[current_dof, 1] = grad_eta
                current_dof += 1
            else:
                break
    if current_dof != num_dofs:
        raise RuntimeError(f"Internal logic error: Basis gradient count mismatch. Expected {num_dofs}, generated {current_dof}")
    return basis_grads

# --- Quadrature Rules ---

def get_triangle_quadrature_rule(required_degree):
    """
    Returns 2D quadrature points and weights for the reference triangle (0,0),(1,0),(0,1).

    Args:
        required_degree (int): Integrate polynomials up to this degree exactly.

    Returns:
        tuple: (xi_coords, eta_coords, weights) - Weights scaled to sum to 0.5 (area).

    NOTE/TODO: Using placeholder. Strongly recommend using a library like 'quadpy'
               or known high-degree rules (e.g., Dunavant).
               `scheme = quadpy.t2.get_good_scheme(required_degree)`
               `points, weights = scheme.points, scheme.weights * 0.5`
    """
    print(f"Note: Requesting volume quadrature for degree {required_degree}.")
    if required_degree < 6: # Required degree for P=3 assembly is 2*P_ORDER = 6
         print(f"CRITICAL WARNING: Using placeholder quadrature insufficient for P={P_ORDER}. Results WILL BE WRONG.")
         # Fallback to a known degree 3 rule (4 points) - NOT ACCURATE ENOUGH
         weights = np.array([ -27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0])
         weights *= 0.5 / np.sum(weights) # Ensure sum is 0.5
         xi_coords  = np.array([1.0/3.0, 0.6, 0.2, 0.2])
         eta_coords = np.array([1.0/3.0, 0.2, 0.6, 0.2])
         return xi_coords, eta_coords, weights
    elif required_degree >= 6:
         print(f"CRITICAL WARNING: Using placeholder quadrature insufficient for P={P_ORDER}. Results WILL BE WRONG.")
         # Using a known degree 5 rule (7 points) - Still NOT accurate enough for 2P=6!
         # Replace this with a correct rule for degree 6 or higher from quadpy/literature.
         w_ = np.array([0.125939180544827,+0.132394152788506,+0.132394152788506,
                        +0.132394152788506,+0.058959942643300,+0.058959942643300,
                        +0.058959942643300])
         xi_ = np.array([1/3.,0.797426985353087,0.101286507323456,0.101286507323456,
                         0.059715871789770,0.470142064105115,0.470142064105115])
         eta_ = np.array([1/3.,0.101286507323456,0.797426985353087,0.101286507323456,
                          0.470142064105115,0.059715871789770,0.470142064105115])
         weights = w_ * 0.5 / np.sum(w_) # Scale weights to sum to 0.5
         return xi_, eta_, weights
    else:
         raise ValueError(f"Volume quadrature degree {required_degree} not implemented.")


def get_line_quadrature_rule(required_degree):
    """
    Returns 1D Gauss-Legendre quadrature points/weights on [-1, 1].

    Args:
        required_degree (int): Integrate polynomials up to this degree exactly.

    Returns:
        tuple: (points, weights) on interval [-1, 1].
    """
    num_points = int(np.ceil((required_degree + 1) / 2))
    if num_points < 1: num_points = 1 # Ensure at least one point
    points, weights = roots_legendre(num_points)
    return points, weights

# --- Coordinate Mapping ---

def calculate_affine_mapping(element_vertices):
    """
    Calculates Jacobian details for the map from ref triangle to physical element.

    Args:
        element_vertices (np.ndarray): (3, 2) array of physical vertex coordinates
                                        [[x0, t0], [x1, t1], [x2, t2]].

    Returns:
        tuple: (jacobian, inverse_jacobian, determinant)
    """
    v0, v1, v2 = element_vertices[0, :], element_vertices[1, :], element_vertices[2, :]

    # Jacobian J = [[dx/dxi, dx/deta], [dt/dxi, dt/deta]]
    jacobian = np.array([
        [v1[0] - v0[0], v2[0] - v0[0]],
        [v1[1] - v0[1], v2[1] - v0[1]]
    ])

    determinant = np.linalg.det(jacobian)

    # Check for degenerate or flipped elements
    if abs(determinant) < 1e-14:
        area = 0.5 * abs(v0[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-v1[1]))
        if area < 1e-14:
             raise ValueError(f"Degenerate element vertices (Area ~ 0): {element_vertices}")
        else:
             # Determinant near zero suggests sliver element, could cause numerical issues
             print(f"Warning: Jacobian determinant {determinant:.2e} near zero for element {element_vertices}. Area={area:.2e}")

    if determinant <= 0:
         print(f"Warning: Element {element_vertices} has non-positive Jacobian determinant {determinant:.2e}. Check vertex ordering (should be counter-clockwise).")
         # Depending on strictness, could raise error here

    # Inverse Jacobian
    inv_jacobian = np.array([
        [ jacobian[1, 1], -jacobian[0, 1]],
        [-jacobian[1, 0],  jacobian[0, 0]]
    ]) / determinant # Apply determinant division

    return jacobian, inv_jacobian, determinant


def transform_reference_gradients(ref_grads, inv_jacobian):
    """ Maps gradients from reference (xi, eta) to physical (x, t) coordinates. """
    # Input ref_grads shape: (N_DOF_PER_ELEMENT, 2)
    # Formula: [d/dx, d/dt] = [d/dxi, d/deta] @ J_inv
    phys_grads = ref_grads @ inv_jacobian
    return phys_grads # Shape: (N_DOF_PER_ELEMENT, 2)


# --- Mesh Data Structure (Conceptual) ---

class MeshData:
    """ Simple container for mesh data. Needs population from a mesh generator. """
    def __init__(self, vertices, elements, edge_map):
        """
        Args:
            vertices (np.ndarray): (N_verts, 2) coordinates [x, t].
            elements (np.ndarray): (N_elems, 3) vertex indices per element.
            edge_map (dict): Maps edge key (e.g., sorted vertex tuple) to info:
                             {'vertices':(v1,v2), 'length':L, 'normal':n,
                              'elements':(e1, e2), 'local_indices':(loc1, loc2)}
                              e2=-1 for boundary. Normal points out from e1.
        """
        self.vertices = vertices
        self.elements = elements
        self.num_elements = elements.shape[0]
        self.edge_map = edge_map # Assumes this map is complete and correct

        self.element_vertices_coords = np.array([vertices[el_verts] for el_verts in elements])
        self.edges_per_element = self._build_edge_list_per_element()

    def _build_edge_list_per_element(self):
        """ Helper to organize edge info by element index. """
        edges_by_elem = [[] for _ in range(self.num_elements)]
        if not self.edge_map:
            print("Warning (Mesh): Edge map is empty, cannot build edge list per element.")
            return edges_by_elem

        for edge_key, info in self.edge_map.items():
            try:
                el1_idx, el2_idx = info['elements']
                loc1_idx, loc2_idx = info['local_indices']

                # Info for element 1
                edges_by_elem[el1_idx].append({
                    'edge_key': edge_key,
                    'local_idx': loc1_idx,
                    'normal': info['normal'], # Normal outward from el1
                    'length': info['length'],
                    'neighbor_element': el2_idx,
                    'neighbor_local_idx': loc2_idx
                })
                # Info for element 2 (if internal)
                if el2_idx != -1:
                    edges_by_elem[el2_idx].append({
                        'edge_key': edge_key,
                        'local_idx': loc2_idx,
                        'normal': -info['normal'], # Normal outward from el2
                        'length': info['length'],
                        'neighbor_element': el1_idx,
                        'neighbor_local_idx': loc1_idx
                    })
            except KeyError as e:
                 print(f"Error processing edge {edge_key}: Missing key {e} in edge_map info.")
            except IndexError as e:
                 print(f"Error processing edge {edge_key}: Element index out of bounds? {e}")
        # Sanity check: each element should have 3 edges listed
        for k, edges in enumerate(edges_by_elem):
            if len(edges) != 3:
                print(f"Warning (Mesh): Element {k} has {len(edges)} edges associated, expected 3.")
        return edges_by_elem

    def get_vertices_for_element(self, element_index):
        """ Get coordinates of vertices for element k. """
        if not (0 <= element_index < self.num_elements):
            raise IndexError(f"Element index {element_index} out of range.")
        return self.element_vertices_coords[element_index]

    def get_edge_info_for_element(self, element_index):
        """ Get list of edge info dictionaries for element k. """
        if not (0 <= element_index < self.num_elements):
            raise IndexError(f"Element index {element_index} out of range.")
        return self.edges_per_element[element_index]

# --- Numerical Flux ---

def flux_lax_friedrichs(u_in, u_out, normal_vec, wave_speed_a):
    """ Calculates Lax-Friedrichs flux for u_t + a u_x = 0 in space-time. """
    nx, nt = normal_vec[0], normal_vec[1]
    # Physical flux component normal to the edge: F(u) . n = (a*u*nx + u*nt)
    flux_phys_in = (wave_speed_a * nx + nt) * u_in
    flux_phys_out = (wave_speed_a * nx + nt) * u_out

    # Max wave speed normal to edge (alpha in LF formula)
    # C = | F'(u) . n | = | (a*nx + nt) |
    max_normal_speed = abs(wave_speed_a * nx + nt)

    # LF Flux: 0.5 * (F_n(in) + F_n(out)) - 0.5 * C * (u_out - u_in)
    numerical_flux = 0.5 * (flux_phys_in + flux_phys_out) - 0.5 * max_normal_speed * (u_out - u_in)
    return numerical_flux

# --- Local Element Assembly ---

def compute_element_matrix_M(element_verts, wave_speed_a, poly_order=P_ORDER):
    """ Assembles local matrix M^K for ∫∫ φ_j (a ∂φ_i/∂x + ∂φ_i/∂t) dx dt. """
    num_dofs = (poly_order + 1) * (poly_order + 2) // 2
    element_M = np.zeros((num_dofs, num_dofs))

    # Quadrature degree needed: (P-1) + P = 2P-1. Use 2P for safety.
    quad_degree = 2 * poly_order
    try:
        xi_q, eta_q, w_q = get_triangle_quadrature_rule(quad_degree)
    except ValueError as e:
        print(f"Assembly failed: {e}")
        return None
    num_quad_points = len(w_q)

    try:
        _, inv_J, det_J = calculate_affine_mapping(element_verts)
    except ValueError as e:
        print(f"Assembly failed for element {element_verts}: {e}")
        return None

    for q_idx in range(num_quad_points):
        xi, eta, w = xi_q[q_idx], eta_q[q_idx], w_q[q_idx]

        # Evaluate basis and gradients at quadrature point (reference)
        basis_at_q = evaluate_reference_basis(xi, eta, poly_order)       # Shape (N_DOF,)
        grads_at_q_ref = evaluate_reference_basis_gradients(xi, eta, poly_order) # Shape (N_DOF, 2)

        # Transform gradients to physical coordinates (x, t)
        grads_at_q_phys = transform_reference_gradients(grads_at_q_ref, inv_J) # Shape (N_DOF, 2)
        dphi_dx = grads_at_q_phys[:, 0]
        dphi_dt = grads_at_q_phys[:, 1]

        # Compute M_ij = Sum_q [ weight_q * detJ * basis_j(q) * (a*d(basis_i)/dx + d(basis_i)/dt) ]
        term_i = wave_speed_a * dphi_dx + dphi_dt # Shape (N_DOF,)
        # contribution = np.outer(term_i, basis_at_q) # contribution[i, j] = term_i[i] * basis_at_q[j]
        # outer product gives M[i,j] = term_i[i] * basis_at_q[j]
        element_M += np.outer(term_i, basis_at_q) * w * det_J

    return element_M

def compute_element_rhs_R(elem_idx, element_verts, element_coeffs, mesh_data, global_coeffs_U, wave_speed_a, poly_order=P_ORDER):
    """ Assembles local RHS vector R^K for ∫_{∂K} F̂_n(u^-, u^+) φ_i ds. """
    num_dofs = (poly_order + 1) * (poly_order + 2) // 2
    element_R = np.zeros(num_dofs)

    # Quadrature degree for edge integrals: approx degree P * degree P = 2P
    edge_quad_degree = 2 * poly_order
    try:
        edge_qp, edge_qw = get_line_quadrature_rule(edge_quad_degree) # Points on [-1, 1]
    except ValueError as e:
        print(f"Assembly failed: {e}")
        return None
    num_edge_qp = len(edge_qp)

    # Get edge information for the current element
    edges_info = mesh_data.get_edge_info_for_element(elem_idx)
    if not edges_info: return element_R # Skip if no edge info available

    # Parametrization functions for edges 0, 1, 2 of reference triangle
    ref_edge_param_funcs = [
        lambda s: (s, 0.0),         # Edge 0 (xi=s, eta=0)
        lambda s: (1.0 - s, s),     # Edge 1 (xi=1-s, eta=s)
        lambda s: (0.0, 1.0 - s)     # Edge 2 (xi=0, eta=1-s)
        # Parameter 's' runs from 0 to 1 along the edge
    ]

    for edge in edges_info:
        local_idx = edge['local_idx']
        normal_phys = edge['normal'] # Physical outward normal for this element
        edge_len = edge['length']
        neighbor_idx = edge['neighbor_element']

        # Jacobian for 1D integral on edge: length_physical / length_reference = edge_len / 2.0
        edge_jacobian_1d = edge_len / 2.0
        if abs(edge_jacobian_1d) < 1e-14: continue # Skip zero-length edges

        param_func = ref_edge_param_funcs[local_idx]

        for q_idx in range(num_edge_qp):
            sq, wq = edge_qp[q_idx], edge_qw[q_idx] # Quadrature point/weight on [-1, 1]
            s_param = (sq + 1.0) / 2.0 # Map point from [-1, 1] to parameter [0, 1]

            # Get reference coords (xi, eta) on the edge
            xi_q, eta_q = param_func(s_param)

            # Evaluate basis functions at this point on the edge (in reference coords)
            basis_vals_at_q = evaluate_reference_basis(xi_q, eta_q, poly_order) # Shape (N_DOF,)

            # Evaluate solution from THIS element ('u_minus') at the point
            u_minus_at_q = np.dot(basis_vals_at_q, element_coeffs)

            # Evaluate solution from NEIGHBOR element ('u_plus') at the point
            u_plus_at_q = 0.0 # Default for boundary or error
            if neighbor_idx != -1:
                # Internal edge: Get neighbor's coefficients
                try:
                    neighbor_coeffs = global_coeffs_U[neighbor_idx, :]
                    # Evaluate neighbor's solution using *same* basis values at corresponding point
                    # ASSUMES basis functions align correctly across edge
                    u_plus_at_q = np.dot(basis_vals_at_q, neighbor_coeffs)
                except IndexError:
                    print(f"Error: Neighbor index {neighbor_idx} out of bounds accessing global_coeffs_U.")
                    u_plus_at_q = u_minus_at_q # Fallback: Use own state (can cause issues)
                except Exception as e:
                    print(f"Unexpected error getting neighbor state: {e}")
                    u_plus_at_q = u_minus_at_q # Fallback
            else:
                # Boundary edge: Apply Boundary Condition through u_plus
                # Placeholder: Zero inflow condition
                char_speed_normal = wave_speed_a * normal_phys[0] + normal_phys[1]
                if char_speed_normal < -1e-9: # Characteristic points strictly inward
                    u_plus_at_q = 0.0 # Prescribe external state (e.g., zero)
                else: # Characteristic points outward or tangential
                    u_plus_at_q = u_minus_at_q # Use internal state (effectively sets flux based on u_minus only)

            # Calculate the numerical flux value at the quadrature point
            flux_val_at_q = flux_lax_friedrichs(u_minus_at_q, u_plus_at_q, normal_phys, wave_speed_a)

            # Add contribution to the RHS vector R_i += Sum_q [ wq * jac_1d * flux_val * basis_i(q) ]
            element_R += flux_val_at_q * basis_vals_at_q * wq * edge_jacobian_1d

    return element_R

# --- Plotting ---
def plot_st_element_solution(element_verts, element_coeffs, poly_order=P_ORDER, plot_resolution=15):
    """ Creates a contour plot of the DG solution within one space-time triangle. """
    num_dofs = (poly_order + 1) * (poly_order + 2) // 2
    if len(element_coeffs) != num_dofs:
        raise ValueError(f"Coefficient array size mismatch. Expected {num_dofs}, got {len(element_coeffs)}")

    # 1. Create evaluation points in reference coordinates
    xi_linspace = np.linspace(0, 1, plot_resolution)
    eta_linspace = np.linspace(0, 1, plot_resolution)
    xi_grid, eta_grid = np.meshgrid(xi_linspace, eta_linspace)
    mask = xi_grid + eta_grid <= 1.0 + 1e-9 # Points within ref triangle
    xi_eval = xi_grid[mask]
    eta_eval = eta_grid[mask]
    num_eval_points = len(xi_eval)
    if num_eval_points == 0: return # Nothing to plot

    # 2. Map evaluation points to physical coords (x, t)
    v0, v1, v2 = element_verts[0,:], element_verts[1,:], element_verts[2,:]
    x_eval = v0[0]*(1-xi_eval-eta_eval) + v1[0]*xi_eval + v2[0]*eta_eval
    t_eval = v0[1]*(1-xi_eval-eta_eval) + v1[1]*xi_eval + v2[1]*eta_eval

    # 3. Evaluate solution u_h at physical evaluation points
    u_eval = np.zeros(num_eval_points)
    for i in range(num_eval_points):
        basis_at_pt = evaluate_reference_basis(xi_eval[i], eta_eval[i], poly_order)
        u_eval[i] = np.dot(basis_at_pt, element_coeffs)

    # 4. Plot using tricontourf
    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        # Triangulation based on reference points can be more robust
        tri_ref = tri.Triangulation(xi_eval, eta_eval)
        tri_phys = tri.Triangulation(x_eval, t_eval, triangles=tri_ref.triangles)

        contour_plot = ax.tricontourf(tri_phys, u_eval, cmap='viridis', levels=14)
        fig.colorbar(contour_plot, label=f'$u_h$ (P{poly_order})')
        ax.triplot(tri_phys, 'k-', lw=0.3, alpha=0.5) # Show triangulation lightly
    except Exception as e:
        print(f"Plotting warning: {e}. Using scatter plot fallback.")
        scatter_plot = ax.scatter(x_eval, t_eval, c=u_eval, cmap='viridis', s=15, vmin=np.min(u_eval), vmax=np.max(u_eval))
        fig.colorbar(scatter_plot, label=f'$u_h$ (P{poly_order})')

    # Outline the element
    ax.plot([v0[0], v1[0], v2[0], v0[0]], [v0[1], v1[1], v2[1], v0[1]], 'r-', lw=1.5)

    ax.set_xlabel('$x$ (Space)')
    ax.set_ylabel('$t$ (Time)')
    ax.set_title(f'STDG Solution in Element (Random Coefficients)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)
    return fig, ax


# --- Example Usage ---
if __name__ == "__main__":
    print(f"STDG Setup: Polynomial Order P={P_ORDER}, DOFs/Element N_P={N_DOF_PER_ELEMENT}")

    # --- Dummy Mesh Definition (CRITICAL: Replace with actual mesh generator output) ---
    #     v2 (0, 0.5)
    #     | \
    #     |   \ Elem 0
    #     |     \
    # v0 (0, 0)----v1 (0.5, 0)
    # Element 0: Vertices [0, 1, 2] -> Map ref (0,0)->v0, (1,0)->v1, (0,1)->v2
    # Local Edges (Counter-Clockwise): 0:v0-v1, 1:v1-v2, 2:v2-v0
    dummy_vertices_data = np.array([
        [0.0, 0.0], [0.5, 0.0], [0.0, 0.5]
    ])
    dummy_elements_data = np.array([
        [0, 1, 2] # Single element
    ])
    # Define edges: key=(sorted_vtx_idx_tuple), normal outward from first listed element
    dummy_edge_map_data = {
        (0, 1): { # Edge 0 (Local 0)
            'vertices': (0, 1), 'length': 0.5, 'normal': np.array([0.0, -1.0]),
            'elements': (0, -1), 'local_indices': (0, None)
        },
        (1, 2): { # Edge 1 (Local 1)
            'vertices': (1, 2), 'length': np.sqrt(0.5**2 + 0.5**2), 'normal': np.array([0.5, 0.5]) / np.sqrt(0.5), # Normal: (t2-t1, x1-x2)
            'elements': (0, -1), 'local_indices': (1, None)
        },
        (0, 2): { # Edge 2 (Local 2)
            'vertices': (2, 0), 'length': 0.5, 'normal': np.array([-1.0, 0.0]), # Normal: (t0-t2, x2-x0)
            'elements': (0, -1), 'local_indices': (2, None) # Corrected key order
        }
    }
    # --- End Dummy Mesh ---

    try:
        test_mesh = MeshData(dummy_vertices_data, dummy_elements_data, dummy_edge_map_data)
    except Exception as e:
        print(f"Fatal Error creating MeshData: {e}")
        exit()

    ADVECTION_SPEED = 1.0

    # --- Test Assembly for Element 0 ---
    test_element_index = 0
    try:
        test_element_verts = test_mesh.get_vertices_for_element(test_element_index)
        # Create random coefficients for testing
        test_element_coeffs = np.random.rand(N_DOF_PER_ELEMENT)
        # Dummy global U (only one element here)
        U_global_dummy = test_element_coeffs.reshape(1, -1)
    except Exception as e:
         print(f"Fatal Error getting data for element {test_element_index}: {e}")
         exit()

    print(f"\n--- Assembling for Element {test_element_index} ---")
    print(f"Vertices:\n{test_element_verts}")

    matrix_M = compute_element_matrix_M(test_element_verts, ADVECTION_SPEED, poly_order=P_ORDER)
    if matrix_M is not None:
        print(f"\nLocal Matrix M^K (Shape: {matrix_M.shape})")
        print(f"Top-Left 3x3:\n{matrix_M[:3,:3]}")

    rhs_R = compute_element_rhs_R(test_element_index, test_element_verts, test_element_coeffs,
                                 test_mesh, U_global_dummy, ADVECTION_SPEED, poly_order=P_ORDER)
    if rhs_R is not None:
        print(f"\nLocal RHS R^K (Shape: {rhs_R.shape})")
        print(f"First 3 elements:\n{rhs_R[:3]}")

    # --- Plotting Call ---
    if rhs_R is not None:
        print("\n--- Generating Plot (using random coefficients) ---")
        try:
            fig, ax = plot_st_element_solution(test_element_verts, test_element_coeffs, poly_order=P_ORDER)
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

    print("\n--- NOTE: This script only demonstrates local assembly. ---")
    print("--- Full solver requires global assembly, ICs, BCs, and linear solve. ---")
    print("--- Basis functions and quadrature are placeholders and need replacement! ---")
