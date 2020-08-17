# --------------------------------------
# -- DEFINITION OF PROXIMAL OPERATORS --
# --------------------------------------
import numpy as np
from scipy.sparse.linalg import cg,LinearOperator


def prox_G(x,c,epsilon):
    return x/(1+epsilon*c)

def prox_J(x,c,gamma,n):
    # k = len(x)//n
    # m = np.reshape(x,(k,n))
    # norms = np.tile(np.linalg.norm(m,axis=0),k)
    # return np.where(norms>gamma+c,x*(1-c/norms),
    #                 gamma*x/(c+0.5*gamma+np.sqrt((c+0.5*gamma)**2-c*norms)))
    return np.where(np.abs(x)>gamma+c,x*(1-c/np.abs(x)),
                    #gamma*x/(c+0.5*gamma+np.sqrt((c+0.5*gamma)**2-c*np.abs(x))))
                    x/np.abs(x)*(gamma+gamma**2/2/c-gamma/c*np.sqrt((c+0.5*gamma)**2-c*np.abs(x))))


def prox_F2_dual(z,c,gamma,pn1,n):
    return z-c*prox_J(z/c,pn1/c,gamma,n)


def prox_F1(u,c,p,y,fourier_op):
    return fourier_op.adj_op((fourier_op.op(u)+c*p*p*y)/(np.ones(len(y))+c*p*p))

def prox_F1_dual(u,c,p,y,fourier_op,mask_type):
    #Need to run CG if non cartesian
    if mask_type != "cartesian" and mask_type != "": return prox_F1_dual_NC(u,c,p,y,fourier_op)
    #Use the fact that F^-1=F^* when cartesian to simplify
    else:return u-c*fourier_op.adj_op((fourier_op.op(u)+p*p*y)/(c*np.ones(y.shape)+p*p))


def prox_F1_NC(u,c,p,y,fourier_op):
    n = u.shape[0]
    def mv(x):
        z = np.reshape(x[:n**2]+1j*x[n**2:],(n,n))
        fx = np.reshape(fourier_op.adj_op(c*p**2*fourier_op.op(z))+z,(n**2,))
        return np.concatenate([np.real(fx),np.imag(fx)])
    B = np.reshape(fourier_op.adj_op(c*p**2*y)+u,(n**2,))
    BR = np.concatenate([np.real(B),np.imag(B)])

    lin = LinearOperator((2*n**2,2*n**2),matvec=mv)
    xf,_ = cg(lin,BR,tol=1e-6,maxiter=1000)
    xf = np.reshape(xf[:n**2]+1j*xf[n**2:],(n,n))

    return xf

def prox_F1_dual_NC(u,c,p,y,fourier_op):
    return u-c*prox_F1_NC(u/c,1/c,p,y,fourier_op)
