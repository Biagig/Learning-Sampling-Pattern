# ------------------------------
# -- PARAMETRISATION OF MASKS --
# ------------------------------
import numpy as np

# -- Cartesian mask --
# --------------------
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


# -- Radial mask --
# -----------------
#Center out
def pradCO(l,n_rad):
    #Point (0,0) treated independently
    p = np.zeros((len(l)-2)*(n_rad-1)+2)

    #Center point
    p[0] = l[0]
    #Alpha
    p[-1] = l[-1]
    #Radial lines
    for i in range(1,len(l)-1):
        p[1+i*(n_rad-1):1+(i+1)*(n_rad)-1] = l[i]

    return p

def grad_pradCO(l,x):
    gp = np.zeros(len(l))

    n_rad = int(len(x)/(len(l)-2))
    gp[0] = x[0]
    gp[-1] = x[-1]
    for i in range(1,len(l)-1):
        gp[i] = np.sum(x[1+i*(n_rad-1):1+(i+1)*(n_rad-1)])
    
    return gp