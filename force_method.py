#########################################################################
## AERO 306
## Final
## Elena Welch
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import integrate as integrate

## PERFORM INTEGRATIONS FOR EACH BEAM EQUATION ##
## TODO: signs need to be checked by hand
E = 1
L = 1
g = 1  # m/sec^2
rho = 1  # density
b = 1
h_0 = 1
h_L = 3/7*h_0
h = lambda x: h_0*(1-x/L) + h_L*(x/L)
I = lambda x: b*h(x)**3/12
A = lambda x: b*h(x)
p = lambda x: -rho*g*A(x)
P0 = 1

## case 1 ##
def force_eqns(xbar,E,I,deltax):
    func11 = lambda xbar: xbar**2/(E*I(xbar))
    func12 = lambda xbar: xbar/(E*I(xbar))
    func22 = lambda xbar: 1/(E*I(xbar))
    f_11 = integrate.quad(func11,0,deltax)[0]
    f_12 = integrate.quad(func12,0,deltax)[0]
    f_22 = integrate.quad(func22,0,deltax)[0]
    return f_11, f_12, f_22

def case1(f_11,f_12,f_22):
    x1 = f_22/(f_11*f_22-f_12**2)
    x2 = f_12/(f_11*f_22-f_12**2)
    return x1, x2

def case2(f_11,f_12,f_22):
    x1 = f_12/(f_11*f_22-f_12**2)
    x2 = -f_11/(f_11*f_22-f_12**2)
    return x1, x2

def case3(f_11,f_12,f_22):
    x1 = f_22/(f_11*f_22-f_12**2)
    x2 = -f_12/(f_11*f_22-f_12**2)
    return x1, x2

def case4(f_11,f_12,f_22):
    x1 = -f_12/(f_11*f_22-f_12**2)
    x2 = f_11/(f_11*f_22-f_12**2)
    return x1, x2

def case5(f_11,f_12,f_22,xbar,E,I,deltax):
    func1 = lambda xbar: (P0 * xbar**2/2 * (xbar/(E*I(xbar))))
    func2 = lambda xbar: (P0 * xbar**2/2 * (1/(E*I(xbar))))
    b1 = integrate.quad(func1, 0,deltax)[0]
    b2 = integrate.quad(func2, 0, deltax)[0]
    x1 = (b1*f_22-b2*f_12)/(f_11*f_22-f_12**2)
    x2 = (b2*f_11-b1*f_12)/(f_11*f_22-f_12**2)
    return x1, x2

def construct_K(case1,case2,case3,case4,deltax):  # input cases as lists [x1,x2]
    K = np.array([[case1[0],case2[0],-case3[0],-case4[0]],
                    [-case1[1], -case2[1], -case3[0]*deltax-case3[1], -case4[0]*deltax-case4[1]],
                    [-case1[0], -case2[0], case3[0], case4[0]],
                    [case1[0]*deltax+case1[1], case2[0]*deltax+case2[1], case3[1], case4[1]]])

    return K

def construct_f(case5,p,deltax,n):
    x = np.arange(0,n,deltax)
    for i in range(n):
        f = np.array([[-case5[0]],
                    [case5[1]],
                    [case5[0] + 2/3*deltax*(p(x[i-1]))+1/3*deltax*(p(x[i]))],
                    [-case5[0]*deltax+case5[1]*deltax**2*2/3*1/2*p(x[i-1])+1/6*p(x[i])*deltax**2]])
    return f

def force_method(K_global,f_global,n):  # not obtaining enough values
    K_reduced = K_global[2:-2,2:-2]
    f_reduced = f_global[2:-2]
    U = np.linalg.solve(K_reduced,f_reduced)  # for fixed-fixed boundary
    U = np.concatenate(([[0],[0]],U,[[0],[0]]), axis=0)
    v = U[0::2]  # index 0-> end, odds
    dv = U[1::2]
    return v, dv

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

def obtain_shear(v,dv,K,f,n):
    shears = []
    for i in range(0,n-1):
        try:
            left_shear = v[i]*K[0,0] + dv[i]*K[0,1] \
            + v[i+1]*K[0,2] + K[0,3]*dv[i+1]- f[0]
            shears.append(left_shear)
        except IndexError:
            shears.append(0)
    right_shear = -(v[-2]*K[-2,-4] + dv[-2]*K[-2,-3] + v[-1]*K[-2,-2] \
    + dv[-1]*K[-2,-1] - f[-2])
    shears.append(right_shear)
    return shears

def obtain_moments(v,dv,K,f,n):
    moments = []
    for i in range(0,n-1):
        try:
            left_moment = -(v[i]*K[1,0] + dv[i]*K[1,1] \
            + v[i+1]*K[1,2] + dv[i+1]*K[1,3]- f[1])
            moments.append(left_moment)
        except IndexError:
            moments.append(0)
    right_moment = (v[-2]*K[-1,-4] + dv[-2]*K[-1,-3] + v[-1]*K[-1,-2] \
    + dv[-1]*K[-1,-3] - f[-1,0])
    moments.append(right_moment)
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

##########################################################
number_nodes = [1,2,3]
deltax_list = []  # for plotting purposes
V_midpt, M_midpt, v_midpt, dv_midpt = [],[],[],[]
for node in number_nodes:
    n = 2**node
    deltax = L/n
    deltax_list.append(deltax)
    print("###################################")
    print("n = " + str(n))
    print("deltax = " + str(deltax))
    xbar = lambda x: x/L
    f_11,f_12,f_22 = force_eqns(xbar,E,I,deltax)
    x11,x12 = case1(f_11,f_12,f_22)
    x21,x22 = case2(f_11,f_12,f_22)
    x31,x32 = case3(f_11,f_12,f_22)
    x41,x42 = case4(f_11,f_12,f_22)
    x51,x52 = case5(f_11,f_12,f_22,xbar,E,I,deltax)
    print("case1: ", x11,x12)
    print("case2: ", x21,x22)
    print("case3: ", x31,x32)
    print("case4: ", x41,x42)
    print("case5: ", x51,x52)
    K_force = construct_K([x11,x12],[x21,x22],[x31,x32],[x41,x42],deltax)
    f_force = construct_f([x51,x52],p,deltax,n)
    K_global, f_global = obtain_globals(K_force,f_force,n)
    print("K = \n", tabulate(K_global,tablefmt='latex'))
    print("f = \n", tabulate(f_global,tablefmt='latex'))
    v, dv = force_method(K_global,f_global,n)
    shears = obtain_shear(v,dv,K_global,f_global,n)
    moments = obtain_moments(v,dv,K_force,f_force,n)
    print(tabulate(zip(*[v,dv,shears,moments]),headers=["v","v'",'Shear','Moment'],tablefmt='latex'))
    mid_pt = int((n-1)//2)  # for richardsons, after the loop
    V_midpt.append(shears[mid_pt])
    M_midpt.append(moments[mid_pt])
    v_midpt.append(v[mid_pt])
    dv_midpt.append(dv[mid_pt])

    x_plot1 = np.linspace(0,L,n+1)
    x_plot2 = np.linspace(0,L,n)
    plt.figure(1)
    plt.plot(x_plot1/L, v[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel("v")
    plt.title('Displacement')
    plt.legend()
    plt.savefig('displ_finalexam.png')
    plt.figure(2)
    plt.plot(x_plot1/L, dv[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel("v'")
    plt.title('Curvature')
    plt.legend()
    plt.savefig('vprime_finalexam.png')
    plt.figure(3)
    plt.plot(x_plot2/L, shears[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel('V')
    plt.title('Shear Force')
    plt.legend()
    plt.savefig('shear_finalexam.png')
    plt.figure(4)
    plt.plot(x_plot2/L, moments[:], label='n='+str(n))
    plt.xlabel('x')
    plt.ylabel('M')
    plt.title('Moments')
    plt.savefig('mom_finalexam.png')
    plt.legend()

## exact_values ##
V_exact = richardson(V_midpt[-1],V_midpt[-2],V_midpt[-3])  # from the largest 3 meshes
M_exact = richardson(M_midpt[-1],M_midpt[-2],M_midpt[-3])
v_exact = richardson(v_midpt[-1],v_midpt[-2],v_midpt[-3])
dv_exact = richardson(dv_midpt[-1],dv_midpt[-2],dv_midpt[-3])
print(tabulate(zip(*[V_exact,M_exact,v_exact,dv_exact]),headers=['V(L/2)','M(L/2)','v(L/2)',"v'(L/2)"],tablefmt='latex'))
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
plt.savefig('conv_finalexam.png')
plt.legend()
# plt.show()
