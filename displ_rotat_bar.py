#########################################################################
## Elena Welch
## March 2022
## calculate displacement, curvature, shear and moment values for a fixed-hinged/fixed-fixed beam
## using the Finite Element Method and Hermite shape functions.
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
#########################################################################
'''IF THESE VALUES ARE NOT 1, THE STRONG FORMULATION DOES NOT STAND'''
L = 1
A_0 = 1
rho = 1
omega = 1
nodes = np.array([1, 2, 4, 8, 16])
E = 1
A_L = A_0*4/7
error_weak = []
error_dd = []
#########################################################################
## Area of a beam of length L
A = lambda x: A_0*(1 - x/L) + A_L*(x/L)
## centrifugal force for beam rotating at rate omega
f = lambda x: A(x)*rho*omega**2*x
#########################################################################
## Strong formulation is calculated by hand
C_1 = (-321-1372*np.log(2)+686*np.log(7))/(756*np.log(7/4))
C_2 = (376*np.log(7)-1715*np.log(2))/(486*np.log(7/4))
u = lambda x: 1/972*(-42*(54*C_1-49)*np.log(7-3*x)-(12*x+35)*(7-3*x)**2) + C_2
N = lambda x: A(x)*E*1/(7*(1-3/7*x))*(x**3-7/2*x**2) + C_1/(1-3/7*x)
N_root = N(0)  # reaction force at the root
u_half = u(L/2)  # displacement of bar at L/2
print("Displacement of bar at x=L/2: ", u_half)
print("Root reaction, strong formulation: ", N_root)
plt.figure(1)
X = np.linspace(0, L, 100)
plt.plot(X, u(X), label="Strong Solution")
#########################################################################
## Weak formulation: shape functions are quadratic
def psi(m, x1, x2, x):  # quadratic shape functions, m=1,2,3
    if m == 1:  # linear element
        return (x2-x)/(x2-x1)
    elif m == 2:  # linear element
        return (x-x1)/(x2-x1)
    elif m == 3:  # quadratic element
        return ((x2-x)*(x-x1))/(x2-x1)**2
def dpsi(m, x1, x2, x):  # derivative of psi function, m=1,2,3
    if m == 1:
        return -1/(x2-x1)
    elif m == 2:
        return 1/(x2-x1)
    elif m == 3:
        return (x2-2*x+x1)/(x2-x1)**2

for node in nodes:
    deltax = L/node
    x = np.linspace(0, L, node+1)
## MATRIX CREATION: K and F
    K_global = np.zeros((node+1, node+1))
    F_global = np.zeros((node+1, 1))
    for i in range(node):
        x1 = deltax*i
        x2 = deltax*(i+1)
        ## Gauss Legendre weights
        xmid = deltax*(i+0.5)
        w1 = deltax/2 * (5/9)
        w2 = deltax/2 * (8/9)
        w3 = deltax/2 * (5/9)
        ## Gauss Legendre points
        x_gl1 = xmid - deltax/2*(np.sqrt(3/5))
        x_gl2 = xmid
        x_gl3 = xmid + deltax/2*(np.sqrt(3/5))

        K = np.zeros((3, 3))
        F = np.zeros((3, 1))
        for m in range(3):
            F[m,0] = w1*f(x_gl1)*psi(m+1, x1, x2, x_gl1) + \
            w2*f(x_gl2)*psi(m+1, x1, x2, x_gl2) + \
            w1*f(x_gl3)*psi(m+1, x1, x2, x_gl3)
            for n in range(3):
                K[m,n] = w1*E*A(x_gl1)*dpsi(m+1, x1, x2, x_gl1)*dpsi(n+1, x1, x2, x_gl1) + \
                w2*E*A(x_gl2)*dpsi(m+1, x1, x2, x_gl2)*dpsi(n+1, x1, x2, x_gl2) + \
                w3*E*A(x_gl3)*dpsi(m+1, x1, x2, x_gl3)*dpsi(n+1, x1, x2, x_gl3)

## MATRIX CONDENSATION: K is 2x2, F is 2x1, these "bar" values are the simplified
## version of K and F, there is nothing else that's been changed
        if i == 0:  # storing values needed for alpha
            K_22_0 = K[2,2]
            K_21_0 = K[2,1]
            F_2_0 = F[2,0]
        K_bar = np.zeros((2,2,i+1))  # 3D array: 3x3xnode
        F_bar = np.zeros((2,1,i+1))  # 3D array: 3x3xnode
        for m in range(2):  # m=0,1,2
            F_bar[m,0,i] = F[m,0] - (K[m,2]/K[2,2])*F[2,0]
            for n in range(2):
                K_bar[m,n,i] = K[m,n] - (K[m,2]*K[2,n])/K[2,2]
        ## Global assembly of the matrix
        K_global[i,i] = K_global[i,i] + K_bar[0,0,i]
        K_global[i+1, i] = K_global[i+1,i] + K_bar[1,0,i]
        K_global[i,i+1] = K_global[i,i+1] + K_bar[1,0,i]
        K_global[i+1,i+1] = K_global[i+1,i+1] + K_bar[1,1,i]
        F_global[i,0] = F_global[i,0] + F_bar[0,0,i]
        F_global[i+1,0] = F_global[i+1,0] + F_bar[1,0,i]

        K_reduced = K_global[1:-1, 1:-1]  # removed the first and last elements
        F_reduced = F_global[1:-1,0]

    print("n=", str(node))
    print("K_bar=\n", tabulate(K_global))
    print("F_bar=\n", tabulate(F_global))
    U_sol = np.linalg.solve(K_reduced, F_reduced)  # solve for U reduced (excluding boundaries, U = 0)
    U = np.concatenate(([0], U_sol, [0]))  # finally including boundaries
    print("U=\n", tabulate(zip(U)))  # for appearances & readbility only
    alpha = (F_2_0-K_21_0*U[1])/K_22_0 # alpha found with simple algebra
    N_0_weak = F_global[0,0] - K_global[0,1]*U[1]  # weak root reaction
    N_0_dd = A_0*E*(U[1]+alpha)/deltax  # direct differentiation
    error_weak.append(N_root-N_0_weak)
    error_dd.append(N_root-N_0_dd)
    print("##############################")
    print("N (weak) = ", str(N_0_weak))
    print("N (direct diff) = ", str(N_0_dd))
    print("##############################")
    plt.figure(1)
    plt.plot(x/L, U, label='n='+str(node))
    #########################################################################
plt.title("Normalized Displacement")
plt.xlabel(r"$\frac{x}{L}$")
plt.ylabel(r"\bar{u}")
plt.legend()
plt.show()
#########################################################################
e_2_w = error_weak[1]
e_1_w = error_weak[0]
e_2_d = error_dd[1]
e_1_d = error_dd[0]
beta_weak = np.log2(e_1_w/e_2_w)  # rate of convergence found for n=2, n=4
beta_direct = np.log2(e_1_d/e_2_d)  # rate of convergence found for n=2, n=4
print("Rate of Convergence:")
print("beta weak = ", beta_weak)
print("beta dd = ", beta_direct)
# TODO: log plot of rate of convergence
