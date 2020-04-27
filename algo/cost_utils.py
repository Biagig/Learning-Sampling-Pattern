import numpy as np

def Rstar(u):
    return u+0j
def f1(v,p,y,fourier_op):
    return 0.5*np.linalg.norm(p*(fourier_op.op(v)-y))**2

# --- Funtion G: L2 norm + negative values penalisation
# INPUT: u=numpy 2d array
# OUTPUT: B(u) or G(u)
def b(u,zeta):
    return zeta*np.sum(np.where(u<=0,-u**3,0))
def g(u,zeta,epsilon):return np.real(b(u,zeta)+epsilon*np.linalg.norm(u)**2/2)


# --- Function F2: C2 approximation of L1 norm of a vector
# INPUT: w=vector representing a wavelet transform of an image
# OUTPUT: F2(u)
def rho(x,gamma):
    m = np.abs(x)
    return np.where(m<gamma,-m**3/3/gamma**2+m**2/gamma,m-gamma/3)

def J(u,pn1,gamma):
    return pn1*np.sum(rho(u,gamma))
def f2(w,pn1,gamma):
    return pn1*np.sum(rho(np.abs(w),gamma))


def energy_wavelet(u,p,y,pn1,gamma,zeta,epsilon,linear_op,fourier_op):
    return(f1(Rstar(u),p,y,fourier_op)+f2(linear_op.op(u),pn1,gamma)+g(u,zeta,epsilon))