# Space-Time Discontinuous Galerkin (STDG) - 1D Advection Components

This repository contains Python code implementing core components for a Space-Time Discontinuous Galerkin (STDG) solver aimed at solving the 1D linear advection equation (`u_t + a u_x = 0`). The method discretizes the problem directly on a 2D space-time domain using triangular elements.

This code was developed as part of a research project investigating the impact of space-time mesh orientation (e.g., T-axis aligned, X-axis aligned, Checkerboard) on the accuracy and stability of STDG methods for hyperbolic problems.

**Note:** This repository currently contains the building blocks and local assembly routines but **does not yet include the global assembly and linear solver** required for a full simulation. Key components like basis functions and quadrature rules are implemented as **placeholders** and need replacement for accurate results.

## Features & Components

The code includes functions and structures for:

1.  **Reference Element Setup (Placeholder):**
    *   Defines the standard reference triangle (vertices (0,0), (1,0), (0,1)).
    *   Includes placeholder functions (`basis_functions`, `basis_gradients`) for evaluating 2D polynomial basis functions and their gradients up to a specified order (`P_ORDER`, currently set to 3). **These use simple monomials and MUST be replaced with a suitable orthogonal basis (e.g., Jacobi-based) for actual use.**
2.  **Quadrature Rules (Placeholder & Basic):**
    *   `get_volume_quadrature`: Placeholder function intended to return 2D quadrature points and weights for triangles (e.g., from `quadpy` or established rules) accurate for a given polynomial degree. **Currently uses low-accuracy placeholder rules.**
    *   `get_edge_quadrature`: Returns 1D Gauss-Legendre quadrature points and weights for edge integrals using `scipy.special.roots_legendre`.
3.  **Affine Mapping:**
    *   `get_element_mapping`: Calculates the Jacobian, inverse Jacobian, and determinant for the mapping between the reference element and a physical triangular space-time element.
    *   `map_gradients`: Transforms gradients computed on the reference element to physical (x, t) coordinates.
4.  **Mesh Structure (Conceptual):**
    *   A conceptual `Mesh` class is defined to hold vertices, element definitions, and **crucially, detailed edge connectivity information** (normals, length, neighboring elements, local edge indices). **This structure needs to be populated by a mesh generator that provides this connectivity data.**
5.  **Numerical Flux:**
    *   `lax_friedrichs_flux`: Implements the Local Lax-Friedrichs numerical flux suitable for the space-time formulation of the 1D advection equation.
6.  **Local Matrix/Vector Assembly:**
    *   `assemble_local_matrix`: Computes the local "stiffness" matrix (`M^K`) for a single element, corresponding to the volume integral term in the STDG weak form.
    *   `assemble_local_rhs`: Computes the local right-hand side vector (`R^K`) for a single element, corresponding to the boundary integral terms involving the numerical flux. Requires mesh connectivity information and access to neighbor solution coefficients.
7.  **Plotting Utility:**
    *   `plot_element_solution`: Generates a contour plot visualizing the DG solution (represented by its coefficients) within a single physical space-time element.

## Current Status & Limitations

*   **Placeholders:** The basis functions/gradients and volume quadrature rules are **placeholders** and not suitable for production runs. They need to be replaced with robust, accurate implementations.
*   **Mesh Connectivity:** The code relies on a `Mesh` object with detailed edge connectivity. The example uses a hardcoded `dummy_edges` dictionary. A proper mesh generation routine that outputs this information is required.
*   **Boundary Conditions:** The `assemble_local_rhs` function has only a very basic placeholder for boundary condition handling via the flux. A correct implementation (e.g., for periodic BCs) is needed, likely involving modifications to how neighbor information is retrieved at boundaries.
*   **No Global Solver:** The code **stops** after demonstrating the assembly of *local* matrices and vectors for a single element. It **does not** perform global assembly into `MU=R` or solve the resulting linear system.
*   **Not a Solver:** This code provides components and demonstrates local assembly logic. It is **not** a complete, runnable STDG solver.

## Dependencies

*   `numpy`
*   `scipy` (for `roots_legendre`, potentially sparse solvers later)
*   `matplotlib` (for plotting)
*   *(Optional but Recommended)* `quadpy` (for accurate triangle quadrature rules)

## How to Run the Example

Executing the Python script (`python your_script_name.py`) will currently:
1.  Define the placeholder functions and parameters (P=3).
2.  Create a simple 2-element dummy mesh with hardcoded edge information.
3.  Generate random solution coefficients (`U_dummy`).
4.  Call `assemble_local_matrix` and `assemble_local_rhs` for the first element (`k_test = 0`), printing parts of the results.
5.  Call `plot_element_solution` to display a contour plot of the solution within the first element, based on the **random coefficients**.

The output demonstrates the assembly process for a single element but does not represent a physical solution.

## Future Work / TODO

To develop this into a functional STDG solver:

1.  Implement robust, orthogonal 2D basis functions and their gradients for triangles (e.g., using Jacobi polynomials).
2.  Implement accurate 2D volume quadrature rules (e.g., using `quadpy`).
3.  Integrate with a mesh generator that provides full element connectivity information in the format expected by the `Mesh` class.
4.  Implement the global assembly loop to build the sparse global matrices `M` and `R`.
5.  Implement correct boundary condition handling during global assembly or RHS calculation.
6.  Implement the projection of initial conditions onto the basis functions to get the starting state (relevant for elements intersecting t=0).
7.  Add a sparse linear solver (e.g., `scipy.sparse.linalg.spsolve`) to solve `M U = R`.
8.  Develop post-processing routines to extract solutions at specific times and calculate errors against analytical solutions.
9.  Run simulations for different mesh orientations and analyze results.
