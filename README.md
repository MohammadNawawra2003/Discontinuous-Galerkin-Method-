![DG Solution vs Exact Solution at T=I OO (Baseline)](https://github.com/user-attachments/assets/38503498-e544-4703-9187-617b801099e9)
Discontinuous Galerkin Method Implementation
This repository contains an implementation of the Discontinuous Galerkin (DG) method, a numerical technique for solving differential equations. DG methods combine features of finite element and finite volume methods, making them suitable for a wide range of applications, particularly those involving complex geometries or problems with dominant first-order characteristics.

Features
High-Order Accuracy: DG methods allow for high-order polynomial approximations within each element, providing accurate solutions.

Flexibility: The use of discontinuous basis functions enables handling of complex geometries and allows for local mesh refinement without constraints on continuity between elements.

Parallelizability: DG methods are well-suited for parallel computing architectures due to their element-wise formulation.
