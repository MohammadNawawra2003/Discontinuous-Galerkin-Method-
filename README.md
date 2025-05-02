Discontinuous Galerkin Method (Space only)
--------------------
Space only Lagrange code results:

DG P1 Lagrange Solution uh(x, t) Surface Plot IC sine wave, n=40:
![DG P1 Lagrange Solution uh(x, t) Surface Plot IC sine wave, n=40](https://github.com/user-attachments/assets/c902536b-58f5-4b07-b30d-4650c31e206a)

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


DG P1 Lagrange vs Exact IC square wave:
![DG P1 Lagrange vs Exact IC square wave](https://github.com/user-attachments/assets/69399af9-6fb2-4156-9ca3-9097500bc53b)

Comparison of DG Lagrange P1 Solutions IC square wave:
![Comparison of DG Lagrange P1 Solutions IC square wave](https://github.com/user-attachments/assets/71c44ebf-539f-4b09-8795-f52d2cbcf4cc)

DG P1 Lagrange Convergence IC square wave:
![DG P1 Lagrange Convergence IC square wave](https://github.com/user-attachments/assets/fb6fd23d-98b0-4326-9442-ce8483bdb4f0)

--- Convergence Study Results (Lagrange P1) ---
  n    |    h       |   L2 Error   | Approx. Rate
-------|------------|--------------|--------------
     5 | 0.200000   | 5.674372e-01 |     -
    10 | 0.100000   | 4.820948e-01 | 0.235
    20 | 0.050000   | 3.503165e-01 | 0.461
    40 | 0.025000   | 2.596954e-01 | 0.432
    80 | 0.012500   | 1.944811e-01 | 0.417
---------------------------------
Asymptotic Observed Rate (finest grids): 0.406
(Note: Using discontinuous IC 'ic_square_wave'.
Expected rate is typically < 2. Often O(h^0.5) to O(h^1) in L2 norm.)

Using IC 'square' from normal simulation for n-comparison plot.



Space only Legendre code results:

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

DG P1 Legendre vs Exact IC square wave:
![DG P1 Legendre vs Exact IC square wave](https://github.com/user-attachments/assets/ae081811-2333-4466-ab51-b7beb475b26a)


DG Legendre P1 Convergence IC ic square wave
![DG Legendre P1 Convergence IC square wave](https://github.com/user-attachments/assets/7825a3b3-4b7d-49c2-ac11-e977ab81d487)


Comparison of DG Legendre P1 Solutions for Different n:
![Comparison of DG Legendre P1 Solutions for Different n](https://github.com/user-attachments/assets/a75ed45f-0beb-45c4-bb41-4925a8620cc8)

  n    |    h       |   L2 Error   | Approx. Rate
-------|------------|--------------|--------------
     5 | 0.200000 | 4.538447e-01 |     -
    10 | 0.100000 | 3.940828e-01 |   0.204
    20 | 0.050000 | 2.999868e-01 |   0.394
    40 | 0.025000 | 2.320481e-01 |   0.370
    80 | 0.012500 | 1.793366e-01 |   0.372
---------------------------------
Average Observed Rate: 0.335
(Expected rate for P1 is ~2.0 for smooth solutions)

Discontinuous Galerkin Method (Space-Time)
--------------------
Simulation of STDG Solution for P=1:
![advection_stdg_p1](https://github.com/user-attachments/assets/5b9beeae-9a86-4bd9-8772-9db4d4ef0a75)

Convergence Plot (L2 Error at T=1.0, P=1):
![Convergence Plot (L2 Error at T=1 0, P=1)](https://github.com/user-attachments/assets/6ac5f447-0d8d-4a35-af1a-9da55d72e26b)

Comparison of Numerical Solutions at t =1.0000 (P=1):
![Comparison of Numerical Solutions at t =1 with P=1](https://github.com/user-attachments/assets/20bdf51a-0da0-447c-8881-c50251114bfd)


Numerical Solution u(x,t) Surface (Space-Time DG, P=1):
![Numerical Solution u(x,t) Surface (Space-Time DG, P=1)](https://github.com/user-attachments/assets/ef5797a9-a791-4876-8f9b-4b42e25065cd)

  n    |    h       |   L2 Error   | Approx. Rate
-------|------------|--------------|--------------
   10  | 1.0000e-01 | 2.288621e-02 |     -
   20  | 5.0000e-02 | 4.698625e-03 |   2.2842
   40  | 2.5000e-02 | 1.092036e-03 |   2.1052
   80  | 1.2500e-02 | 2.673834e-04 |   2.0300
---------------------------------

--------------------
The full semi-descrete sysyem equation in matrix form:

![](https://latex.codecogs.com/png.image?\dpi{150}\fn_cm%20%7B%5Ccolor%7Bwhite%7D%20%5Cfrac%7B%5CDelta%20x%7D%7B6%7D%0A%5Cbegin%7Bpmatrix%7D%0A%5Cbegin%7Bpmatrix%7D%202%20%26%201%20%5C%5C%201%20%26%202%20%5Cend%7Bpmatrix%7D%20%26%200%20%26%20%5Cdots%20%26%200%20%5C%5C%0A0%20%26%20%5Cbegin%7Bpmatrix%7D%202%20%26%201%20%5C%5C%201%20%26%202%20%5Cend%7Bpmatrix%7D%20%26%20%5Cdots%20%26%200%20%5C%5C%0A%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%0A0%20%26%200%20%26%20%5Cdots%20%26%20%5Cbegin%7Bpmatrix%7D%202%20%26%201%20%5C%5C%201%20%26%202%20%5Cend%7Bpmatrix%7D%0A%5Cend%7Bpmatrix%7D%0A%5Cbegin%7Bpmatrix%7D%0Au%27_%7B0%2C1%7D%20%5C%5C%20u%27_%7B0%2C2%7D%20%5C%5C%0Au%27_%7B1%2C1%7D%20%5C%5C%20u%27_%7B1%2C2%7D%20%5C%5C%0A%5Cvdots%20%5C%5C%0Au%27_%7Bn-1%2C1%7D%20%5C%5C%20u%27_%7Bn-1%2C2%7D%0A%5Cend%7Bpmatrix%7D%0A%3D%0A%5Cbegin%7Bpmatrix%7D%0AR_0(U)%20%5C%5C%0AR_1(U)%20%5C%5C%0A%5Cvdots%20%5C%5C%0AR_%7Bn-1%7D(U)%0A%5Cend%7Bpmatrix%7D})


The Local RHS Vector for Element 
k (for c> 0, α=1.0):

![](https://latex.codecogs.com/png.image?\dpi{150}\fn_cm%20%7B%5Ccolor%7Bwhite%7D%20R_k(U)%20%3D%20c%20%5Cbegin%7Bpmatrix%7D%20-0.5(u_%7Bk%2C1%7D%20%2B%20u_%7Bk%2C2%7D)%20%2B%20u_%7Bk-1%2C2%7D%20%5C%5C%200.5(u_%7Bk%2C1%7D%20%2B%20u_%7Bk%2C2%7D)%20-%20u_%7Bk%2C2%7D%20%5Cend%7Bpmatrix%7D%2C%20%5Ctext%7B%20with%20%7D%20u_%7B-1%2C2%7D%20%3D%20u_%7Bn-1%2C2%7D%20%5Ctext%7B%20when%20%7D%20k%20%3D%200.%7D)

The Local Time Derivative Update:

![](https://latex.codecogs.com/png.image?\dpi{150}\fn_cm%20%7B%5Ccolor%7Bwhite%7D%20%5Cfrac%7BdU_k%7D%7Bdt%7D%20%3D%20M_k%5E%7B-1%7D%20R_k(U)%20%3D%20%5Cfrac%7B2%7D%7B%5CDelta%20x%7D%20%5Cbegin%7Bpmatrix%7D%202%20%26%20-1%20%5C%5C%20-1%20%26%202%20%5Cend%7Bpmatrix%7D%20R_k(U)%7D)

The large linear system used in STDG is expressed as:
AU = b

Vector U:
![](https://latex.codecogs.com/png.image?\dpi{150}\fn_cm%20{\color{white}U%20=%20\begin{pmatrix}%20U_{0,0}%20\\%20U_{0,1}%20\\%20\vdots%20\\%20U_{0,N_x-1}%20\\%20U_{1,0}%20\\%20\vdots%20\\%20U_{N_t-1,N_x-1}%20\end{pmatrix}})


Vector b:
![](https://latex.codecogs.com/png.image?\dpi{150}\fn_cm%20b%20%3D%20%5Cbegin%7Bpmatrix%7D%20b_%7B0%2C0%7D%20%5C%5C%20b_%7B0%2C1%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20b_%7B0%2CN_x-1%7D%20%5C%5C%200%20%5C%5C%20%5Cvdots%20%5C%5C%200%20%5Cend%7Bpmatrix%7D)


Matrix A:

![](https://latex.codecogs.com/png.image?\dpi{150}\fn_cm%20A%20%3D%20%5Cbegin%7Bpmatrix%7D%20A_%7B0%2C0%7D%20%26%200%20%26%200%20%26%20%5Ccdots%20%26%200%20%5C%5C%20A_%7B1%2C0%7D%20%26%20A_%7B1%2C1%7D%20%26%200%20%26%20%5Ccdots%20%26%200%20%5C%5C%200%20%26%20A_%7B2%2C1%7D%20%26%20A_%7B2%2C2%7D%20%26%20%5Ccdots%20%26%200%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%200%20%26%200%20%26%200%20%26%20%5Ccdots%20%26%20A_%7BN_t-1%2CN_t-1%7D%20%5Cend%7Bpmatrix%7D)



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

