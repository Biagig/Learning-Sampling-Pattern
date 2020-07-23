# ----------------------------------------------------------------
# -- DEFINITION OF GRADIENTS AND HESSIANS OF LOWER LEVEL ENERGY --
# ----------------------------------------------------------------
import numpy as np
from UpperLevel.L1norm import rho_vec,phix_vec,phi_vec,psix_vec,psi_vec


# -- Base functions --
# --------------------
def Eeps(x,p,**kwargs):
    epsilon = kwargs.get("eps",None)
    if epsilon is None: raise ValueError("A parameter eps is needed")

    return epsilon/2*np.linalg.norm(x)**2

def Edata(x,p,**kwargs):
    fourier_op = kwargs.get("fourier_op",None)
    y = kwargs.get("y",None)
    if y is None: raise ValueError("A parameter y is needed")
    if fourier_op is None: raise ValueError("An operator fourier_op is needed")

    return 0.5*np.linalg.norm(p[:-1]**2*(fourier_op.op(x)-y))**2

def Ereg(x,p,**kwargs):
    linear_op = kwargs.get("linear_op",None)
    gamma = kwargs.get("gamma",None)
    if linear_op is None: raise ValueError("An operator linear_op is needed")
    if gamma is None: raise ValueError("A parameter gamma is needed")

    return p[-1]*np.sum(rho_vec(np.abs(linear_op.op(x)),gamma))

def Etot(x,p,**kwargs):return Eeps(x,p,**kwargs)+Edata(x,p,**kwargs)+Ereg(x,p,**kwargs)


# -- First order derivatives --
# -----------------------------
def Du_Eeps(x,p,**kwargs):
    epsilon = kwargs.get("eps",None)
    if epsilon is None: raise ValueError("A parameter eps is needed")

    return epsilon*x

def Du_Edata(x,p,**kwargs):
    fourier_op = kwargs.get("fourier_op",None)
    y = kwargs.get("y",None)
    if fourier_op is None: raise ValueError("An operator fourier_op is needed")
    if y is None: raise ValueError("A parameter y is needed")

    return fourier_op.adj_op(p[:-1]**2*(fourier_op.op(x)-y))

def Du_Ereg(x,p,**kwargs):
    linear_op = kwargs.get("linear_op",None)
    gamma = kwargs.get("gamma",None)
    if linear_op is None: raise ValueError("An operator linear_op is needed")
    if gamma is None: raise ValueError("A parameter gamma is needed")
    
    return p[-1]*linear_op.adj_op(phix_vec(linear_op.op(x),gamma))

def Du_Etot(x,p,**kwargs):return Du_Eeps(x,p,**kwargs)+Du_Edata(x,p,**kwargs)+Du_Ereg(x,p,**kwargs)

def Dp_Edata( x, p, **kwargs ):
    fourier_op = kwargs.get("fourier_op",None)
    y = kwargs.get("y",None)
    if fourier_op is None: raise ValueError("An operator fourier_op is needed")
    if y is None: raise ValueError("A parameter y is needed")

    return p[ :-1 ] * np.abs( fourier_op.op( x ) - y )**2

def Dp_Ereg( x, p, **kwargs ):
    return 0


# -- Second order derivatives --
# ------------------------------
def Du2_Eeps(u,p,w,**kwargs):
    epsilon = kwargs.get("eps",None)
    if epsilon is None: raise ValueError("A parameter eps is needed")

    return epsilon*w

def Du2_Edata(u,p,w,**kwargs):
    fourier_op = kwargs.get("fourier_op",None)
    y = kwargs.get("y",None)
    if y is None: raise ValueError("A parameter y is needed")
    if fourier_op is None: raise ValueError("An operator fourier_op is needed")

    return fourier_op.adj_op(p[:-1]**2*fourier_op.op(w))

def Du2_J(u,w,**kwargs):
    gamma = kwargs.get("gamma",None)
    if gamma is None: raise ValueError("A parameter gamma is needed")

    phi_u = phi_vec(u,gamma)
    psi_u = psi_vec(u,gamma)

    return phi_u*w + psi_u*u*(np.real(u)*np.real(w)+np.imag(u)*np.imag(w))

def Du2_Ereg(u,p,w,**kwargs):
    linear_op = kwargs.get("linear_op",None)
    if linear_op is None: raise ValueError("An operator linear_op is needed")

    lin_u = linear_op.op(u)
    lin_w = linear_op.op(w)
    return p[-1]*(linear_op.adj_op(Du2_J(lin_u,lin_w,**kwargs)))

def Du2_Etot(u,p,w,**kwargs):return Du2_Eeps(u,p,w,**kwargs)+Du2_Edata(u,p,w,**kwargs)+Du2_Ereg(u,p,w,**kwargs)


# -- Cross derivatives --
# -----------------------
def Dpu_Edata(u,p,w,**kwargs):
    fourier_op = kwargs.get("fourier_op",None)
    y = kwargs.get("y",None)
    if y is None: raise ValueError("A parameter y is needed")
    if fourier_op is None: raise ValueError("An operator fourier_op is needed")

    Fu = fourier_op.op(u)-y
    Fw = fourier_op.op(w)

    return 2*p[:-1]*(np.real(Fu)*np.real(Fw)+np.imag(Fu)*np.imag(Fw))

def Dpu_Ereg(u,p,w,**kwargs):
    linear_op = kwargs.get("linear_op",None)
    gamma = kwargs.get("gamma",None)
    if linear_op is None: raise ValueError("An operator linear_op is needed")
    if gamma is None: raise ValueError("A parameter gamma is needed")
    exp = linear_op.adj_op(phi_vec(linear_op.op(u),gamma)*linear_op.op(u))
    return np.sum(np.real(w)*np.real( exp ) + np.imag( w ) * np.imag( exp ))

def Dpu_Etot(u,p,w,**kwargs):
    g = np.zeros((u.shape[0]**2+1,))
    g[:-1] = Dpu_Edata(u,p,w,**kwargs)
    #g[-1] = 0
    g[-1] = Dpu_Ereg(u,p,w,**kwargs)
    return g