# ------------------------------
# -- PARAMETRISATION OF MASKS --
# ------------------------------
import numpy as np

# -- Cartesian mask --
def pcart(l):
    # Input: l, size n+1 representing coefficients of each line of a cartesian mask and l[-1]=alpha
    # Output: p the associated (n,n) cartesian mask
    n = len(l)-1
    p = np.concatenate([l[i]*np.ones(n) for i in range(n+1)])
    return p[:n**2+1]

def grad_pcart(l,x):
    # Gradient of pcart defined as a linear operator to save space in memory
    n = len(l)-1
    gp = np.zeros(n+1)
    for i in range(n):
        gp[i] = np.sum(x[n*i:n*(i+1)])
    gp[-1] = x[-1]
    return gp