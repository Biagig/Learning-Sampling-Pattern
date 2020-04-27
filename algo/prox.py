import numpy as np

def prox_G(x,c,epsilon,zeta):
    return np.where(x>=0,x/(1+epsilon*c),x/(0.5+np.sqrt(0.25-3*zeta*c*x/(1+epsilon*c)**2)))

def prox_J(x,c,gamma):
    return np.where(np.abs(x)>gamma+c,x*(1-c/np.abs(x)),
                    gamma*x/(c+0.5*gamma+np.sqrt((c+0.5*gamma)**2-c*np.abs(x))))
def prox_F2_dual(z,c,gamma,pn1):
    return z-c*prox_J(z/c,pn1/c,gamma)

def prox_F1_dual(u,c,p,y,fourier_op):
    return u-c*fourier_op.adj_op((fourier_op.op(u)+p*p*y)/(c*np.ones(y.shape)+p*p))