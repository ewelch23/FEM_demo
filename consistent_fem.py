import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

## parameters as defined by problem statement ##
E = 1
L = 1
rho = 1
b = 1
h_0 = 1
h_L = 3/7*h_0
g = 1
h = lambda x: h_0*(1-x/L) + h_L*(x/L)
I = lambda x: b*h(x)**3/12
A = lambda x: b*h(x)
p = lambda x: -rho*g*A(x)

def psi(m, x, x1, x2):  # hermite shape functions
    xbar = x-x1
    delx = x2-x1
    if m == 0:
        return 1-3*xbar**2/delx**2 + 2*xbar**3/delx**3
    elif m == 1:
        return xbar-2*xbar**2/delx + xbar**3/delx**2
    elif m == 2:
        return 3*xbar**2/delx**2-2*xbar**3/delx**3
    elif m == 3:
        return -xbar**2/delx+xbar**3/delx**2
    else:
        raise ValueError("Wrong degree")

def ddpsi(m, x, x1, x2):  # second derivative of hermite shape functions
    xbar = x-x1
    delx = x2-x1
    if m == 0:
        return -6/delx**2 + 12*xbar/delx**3
    elif m == 1:
        return -4/delx + 6*xbar/delx**2
    elif m == 2:
        return 6/delx**2-12*xbar/delx**3
    elif m == 3:
        return -2/delx+6*xbar/delx**2
    else:
        raise ValueError("Wrong degree")

def consistent_fem(n, L, E, psi, ddpsi,deltax):  # provide parameters, and shape functions
    '''This function will provide the stiffness and load matrix, when provided \
    the parameters of the beam, and the shape functions to be used. Gauss-Legendre integration \
    is 3-point integration.'''
    ## Gauss Legendre weights
    K_list = []
    f_list = []
    for i in range(n):
        x1 = i*deltax
        x2 = (i+1)*deltax
        xmid = deltax*(i+0.5)
        w1 = deltax/2 * (5/9)
        w2 = deltax/2 * (8/9)
        w3 = deltax/2 * (5/9)
        ## Gauss Legendre points
        x_gl1 = xmid - deltax/2*(np.sqrt(3/5))
        x_gl2 = xmid
        x_gl3 = xmid + deltax/2*(np.sqrt(3/5))
        ## matrix creation, simplified FEM ##
        K = np.zeros((4,4))
        f = np.zeros((4,1))
        for m in range(0,4):
            f[m,0] = w1*p(x_gl1)*psi(m,x_gl1,x1,x2) + w2*p(x_gl2)*psi(m,x_gl2,x1,x2) \
                        + w3*p(x_gl3)*psi(m,x_gl3,x1,x2)
            for j in range(0,4):
                K[m,j] = w1*E*I(x_gl1)*ddpsi(m,x_gl1,x1,x2)*ddpsi(j,x_gl1,x1,x2) \
                            + w2*E*I(x_gl2)*ddpsi(m,x_gl2,x1,x2)*ddpsi(j,x_gl2,x1,x2) \
                            + w3*E*I(x_gl3)*ddpsi(m,x_gl3,x1,x2)*ddpsi(j,x_gl3,x1,x2)

        K_padded = matrix_padding(K, 2*n+2, 2*n+2, 2*i, 2*i)
        f_padded = matrix_padding(f, 2*n+2, 1, 2*i, 0)
        K_list.append(K_padded)
        f_list.append(f_padded)
    K_global = sum(K_list)
    f_global = sum(f_list)
    return K_global, f_global

def matrix_padding(M, w, h, xoffset, yoffset):
    big_M = np.zeros((w, h))  # the size of the new matrix
    for i in range(np.size(M, 0)):
        for j in range(np.size(M,1)):
            big_M[i+xoffset, j+yoffset] = M[i,j]
    return big_M

def obtain_globals(K,f,n):
    K_list = []
    f_list = []
    for i in range(n):
        K_padded = matrix_padding(K, 2*n+2, 2*n+2, 2*i, 2*i)
        f_padded = matrix_padding(f, 2*n+2, 1, 2*i, 0)
        K_list.append(K_padded)
        f_list.append(f_padded)
    K_global = sum(K_list)
    f_global = sum(f_list)
    return K_global, f_global

def obtain_displ(K,f,case):
    K_reduced = K[2:-2,2:-2]
    f_reduced = f[2:-2]
    displ = np.linalg.solve(K_reduced,f_reduced)  # global displacements
    if case == 'fixed-fixed':
        displ = np.concatenate(([[0],[0]],displ,[[0],[0]]), axis=0)
        v = displ[0::2]  # index 0-> end, odds
        dv = displ[1::2]
        return v, dv
    elif case == 'fixed-hinged':
        displ = np.concatenate(([[0],[0]],displ), axis=0)
        v = displ[0::2]  # index 0-> end, odds
        v[0] = 0
        dv = displ[1::2]
        return v, dv
    else:
        raise ValueError("What are the boundary conditions?")

def obtain_shear(v,dv,K,f,n):
    shears = []
    for i in range(0,n-1):
        left_shear = v[i]*K[0,0] + dv[i]*K[0,1] \
        + v[i+1]*K[0,2] + K[0,3]*dv[i+1]- f[0]
        shears.append(left_shear)
    right_shear = -(v[-2]*K[-2,-4] + dv[-2]*K[-2,-3] + v[-1]*K[-2,-2] \
    + dv[-1]*K[-2,-1] - f[-2])
    shears.append(right_shear)
    return shears

def obtain_moments(v,dv,K,f,n,case):
    moments = []
    for i in range(0,n-1):
        left_moment = -(v[i]*K[1,0] + dv[i]*K[1,1] \
        + v[i+1]*K[1,2] + dv[i+1]*K[1,3]- f[1])
        moments.append(left_moment)
    right_moment = (v[-2]*K[-1,-4] + dv[-2]*K[-1,-3] + v[-1]*K[-1,-2] \
    + dv[-1]*K[-1,-3] - f[-1,0])
    if case == 'fixed-fixed':
        moments.append(right_moment)
    elif case == 'fixed-hinged':
        moments.append([0])
    return moments

def rateofconv(errors,deltax):
    betas = []
    for i in range(1,len(errors)):
        beta = np.log(errors[i]/errors[i-1])/np.log(deltax[i]/deltax[i-1])
        betas.append(beta)
    return betas

def richardson(Q1,Q2,Q3):
    value_of_int = (Q2**2-Q1*Q3)/(2*Q2-Q1-Q3)
    return value_of_int

##################################################################
number_nodes = [1]
deltax_list = []  # for plotting purposes
V_midpt, M_midpt, v_midpt, dv_midpt = [],[],[],[]  # storage for richardsons
for node in number_nodes:
    n = 2**node
    deltax = L/n
    deltax_list.append(deltax)
    K_simp, f_simp = consistent_fem(n,L,E,psi,ddpsi,deltax)
    print("K = \n", tabulate(K_simp,tablefmt='latex'))
    print("f = \n", tabulate(f_simp,tablefmt='latex'))
    v, dv = obtain_displ(K_simp,f_simp,"fixed-hinged")
    shear_simp = obtain_shear(v,dv,K_simp,f_simp,n)
    moment_simp = obtain_moments(v,dv,K_simp,f_simp,n,'fixed-hinged')
    print("###################################")
    print("n = " + str(n))
    print("deltax = " + str(deltax))
    print(tabulate(zip(*[v,dv,shear_simp,moment_simp]),headers=["v","v'",'Shear','Moment'],tablefmt='latex'))
    mid_pt = int((n+1)//2)  # for richardsons, after the loop
    V_midpt.append(shear_simp[mid_pt])
    M_midpt.append(moment_simp[mid_pt])
    v_midpt.append(v[mid_pt])
    dv_midpt.append(dv[mid_pt])

    x_plot = np.linspace(0,L,n)
    plt.figure(1)
    plt.plot(x_plot/L, v[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel("v")
    plt.title('Displacement for Fixed-Hinged Beam')
    plt.legend()
    # plt.savefig('displ_fixedhinge_exam2.png')
    plt.figure(2)
    plt.plot(x_plot/L, dv[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel("v'")
    plt.title('Curvature for Fixed-Hinged Beam')
    plt.legend()
    # plt.savefig('vprime_fixedhinge_exam2.png')
    plt.figure(3)
    plt.plot(x_plot/L, shear_simp[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel('V')
    plt.title('Shear Force for Fixed-Hinged Beam')
    plt.legend()
    # plt.savefig('shear_fixedhinge_exam2.png')
    plt.figure(4)
    plt.plot(x_plot/L, moment_simp[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel('M')
    plt.title('Moments for Fixed-Hinged Beam')
    # plt.savefig('mom_fixedhinge_exam2.png')
    plt.legend()

## exact_values ##
V_exact = richardson(V_midpt[-1],V_midpt[-2],V_midpt[-3])  # from the largest 3 meshes
M_exact = richardson(M_midpt[-1],M_midpt[-2],M_midpt[-3])
v_exact = richardson(v_midpt[-1],v_midpt[-2],v_midpt[-3])
dv_exact = richardson(dv_midpt[-1],dv_midpt[-2],dv_midpt[-3])
print(tabulate(zip(*[V_exact,M_exact,v_exact,dv_exact]),headers=['V(L/2)','M(L/2)','v(L/2)',"v'(L/2)"]))
error_V = abs(V_exact-V_midpt[:])/abs(V_exact)
error_M = abs(M_exact-M_midpt[:])/abs(M_exact)
error_v = abs(v_exact-v_midpt[:])/abs(v_exact)
error_dv = abs(dv_exact-dv_midpt[:])/abs(dv_exact)
print(tabulate(zip(*[deltax_list,error_V,error_M,error_v,error_dv]),headers=['deltax','Error in Shear','Error in Moments',"Error in v","Error in v'"],tablefmt='latex'))

beta_V = rateofconv(error_V,deltax_list)
beta_M = rateofconv(error_M,deltax_list)
beta_v = rateofconv(error_v,deltax_list)
beta_dv = rateofconv(error_dv,deltax_list)
print(tabulate(zip(*[beta_V,beta_M,beta_v,beta_dv]),headers=["beta_V","beta_M","beta_v","beta_v'"],tablefmt='latex'))

plt.figure(5)
plt.plot(-np.log(deltax_list[:]),np.log(error_V),label="V(L/2)")
plt.plot(-np.log(deltax_list[:]),np.log(error_M),label='M(L/2)')
plt.plot(-np.log(deltax_list[:]),np.log(error_v),label='v(L/2)')
plt.plot(-np.log(deltax_list[:]),np.log(error_dv),label="v'(L/2)")
plt.title("Rate of Convergence")
plt.xlabel(r"$-\log{(\Delta x)}$")
plt.ylabel(r"$\log{(Relative Error)}$")
plt.savefig('conv_fixedhinge_exam2.png')
plt.legend()
# plt.show()
