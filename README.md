Lagrange code results:

Pl Lagrange Shape Functions on Reference Element [-1, 1]:
![Pl Lagrange Shape Functions on Reference Element  -1, 1](https://github.com/user-attachments/assets/1b75325c-def7-4cfd-8929-55aa7f301dc8)

DG P1 Lagrange vs Exact IC sine wave:
![DG P1 Lagrange vs Exact IC sine wave](https://github.com/user-attachments/assets/ab7f9189-f012-445c-a637-1fbb7295c2ed)

DG P1 Lagrange Convergence IC sine wave:
![DG P1 Lagrange Convergence IC sine wave](https://github.com/user-attachments/assets/fa9f0cf2-c0ec-412b-94eb-47e8bd0f2eaf)

Comparison of DG Lagrange P1 Solutions IC sine wave:
![Comparison of DG Lagrange P1 Solutions IC sine wave](https://github.com/user-attachments/assets/83c99dff-bc57-4e2a-889a-199dbc388b96)

Error comparison for different n:
  n    |    h       |   L2 Error   | Approx. Rate
-------|------------|--------------|--------------
     5 | 0.200000 | 1.814928e-01 |     -
    10 | 0.100000 | 4.002318e-02 |   2.181
    20 | 0.050000 | 8.714361e-03 |   2.199
    40 | 0.025000 | 1.992882e-03 |   2.129
    80 | 0.012500 | 4.742414e-04 |   2.071
---------------------------------

Average Observed Rate (where calculable): 2.123
(Expected rate for P1 elements is ~2.0 for smooth solutions)


Legendre code results:

Legendre Basis Functions p=0..1 on Ref Element [-1, 1]:
![Legendre Basis Functions p=0 1 on Ref Element  -1, 1](https://github.com/user-attachments/assets/eac464ce-5b22-43f5-bf19-c819f7a3b985)

DG P1 Legendre vs Exact IC sine wave:
![DG P1 Legendre vs Exact IC sine wave](https://github.com/user-attachments/assets/651de589-c2a6-44f5-be85-988433b37aad)

DG Legendre P1 Convergence IC ic_sine wave:
![DG Legendre P1 Convergence IC ic_sine wave](https://github.com/user-attachments/assets/7cfb464f-9270-40be-97d1-19cdf420df22)

Comparison of DG Legendre P1 Solutions for Different n:
![Comparison of DG Legendre P1 Solutions for Different n](https://github.com/user-attachments/assets/b7c40945-b23d-41ee-8159-8fdbbe3363d9)

Error comparison for different n:

  n    |    h       |   L2 Error   | Approx. Rate
-------|------------|--------------|--------------
     5 | 0.200000 | 1.152020e-01 |     -
    10 | 0.100000 | 2.170336e-02 |   2.408
    20 | 0.050000 | 4.599619e-03 |   2.238
    40 | 0.025000 | 1.085200e-03 |   2.084
    80 | 0.012500 | 2.669426e-04 |   2.023
---------------------------------
Average Observed Rate: 2.188
(Expected rate for P1 is ~2.0 for smooth solutions)


Discontinuous Galerkin Method
--------------------
This repository provides an implementation of the Discontinuous Galerkin (DG) method for solving differential equations with high-order accuracy and flexibility. The DG method is suitable for problems with complex geometries and those requiring local refinement.

Features
--------------------
High-order accuracy for solving differential equations

Flexible handling of complex geometries


Installation & Usage
--------------------
1.Clone the repository:

git clone https://github.com/MohammadNawawra2003/Discontinuous-Galerkin-Method-.git

2.Navigate to the project directory and follow the setup instructions provided in the docs/ directory.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Documentation & References
--------------------
DG Methods Overview: https://www3.nd.edu/~zxu2/acms60790S15/DG-general-approach.pdf

DG Tutorial: https://www.birs.ca/workshops/2011/11w5086/files/FengyanLi_TutorialOnDGM.pdf

