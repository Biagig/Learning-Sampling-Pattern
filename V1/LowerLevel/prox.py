# --------------------------------------
# -- DEFINITION OF PROXIMAL OPERATORS --
# --------------------------------------
import numpy as np

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

def prox_F1_dual(u,c,p,y,fourier_op):
    return u-c*fourier_op.adj_op((fourier_op.op(u)+p*p*y)/(c*np.ones(y.shape)+p*p))
