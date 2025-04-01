![DG Solution vs Exact Solution at T=I OO (Baseline)](https://github.com/user-attachments/assets/38503498-e544-4703-9187-617b801099e9)
Discontinuous Galerkin Method Implementation
This repository contains an implementation of the Discontinuous Galerkin (DG) method, a numerical technique for solving differential equations. DG methods combine features of finite element and finite volume methods, making them suitable for a wide range of applications, particularly those involving complex geometries or problems with dominant first-order characteristics.

Features
High-Order Accuracy: DG methods allow for high-order polynomial approximations within each element, providing accurate solutions.

Flexibility: The use of discontinuous basis functions enables handling of complex geometries and allows for local mesh refinement without constraints on continuity between elements.

Parallelizability: DG methods are well-suited for parallel computing architectures due to their element-wise formulation.
Getting Started
To use this implementation:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/MohammadNawawra2003/Discontinuous-Galerkin-Method-.git
Navigate to the project directory:

bash
Copy
Edit
cd Discontinuous-Galerkin-Method-
Follow the instructions in the docs/ directory to set up and run the examples.

References
For a comprehensive understanding of the Discontinuous Galerkin method, consider the following resources:

Discontinuous Galerkin Methods: General Approach and Stability: Provides an overview of the general approach and stability considerations of DG methods.

A Tutorial on Discontinuous Galerkin Methods: Offers a tutorial introduction to DG methods, including historical context and foundational concepts.

For a visual explanation and simulation examples of Discontinuous Galerkin methods, you might find the following video helpful:

