import numpy as np
from algo.prox import prox_F1_dual,prox_F2_dual,prox_G
from algo.cost_utils import energy_wavelet
import time

def compute_constants(param,const,p):
    # --
    # -- Computes constants needed for the lower level algorithm
    # --
    # INPUT: param: dict of energy constants gamma, zeta, epsilon and pn1
    #        const: dict containing specific values for tau and sigma we want to use.
    #               OR empty dict; in that case we compute their values according to the article.
    # OUTPUT: dict of pdhg constants    
    L=1
    #From Sherry's code, don't know if there is a way to find something better
    eta = max(np.amax(p)**2,param["pn1"]*5/2/param["gamma"])
    mu = 2*np.sqrt(param["epsilon"]/(1+L**2)/eta)
    theta = 1/(1+mu)
    if not ("sigma" in const.keys()) and not ("tau" in const.keys()):
        tau = mu/2/param["epsilon"]
        sigma = mu*eta/2
    else:
        sigma=const["sigma"]
        tau=const["tau"]
    
    return {"L":1,"eta":eta,"mu":mu,"tau":tau,"sigma":sigma,"theta":theta}

def step(uk,vk,wk,uk_bar,const,p,y,param,linear_op,fourier_op):
    # --
    # -- Computes a step of the pdhg algorithm
    # --
    # INPUTS: - uk,vk,wk,uk_bar: values after k iterations of the pdhg algorithm
    #         - const,param: dicts of pdhg constants and energy parameters
    # OUTPUTS: - uk1,vk1,wk1,uk_bar1: calues after k+1 iterations of the algorithm
    #          - norm: value of the stopping criterion of the algorithm
    
    #Getting useful parameters
    sigma = const["sigma"]
    gamma = param["gamma"]
    pn1 = param["pn1"]
    tau=const["tau"]
    epsilon = param["epsilon"]
    zeta = param["zeta"]
    theta = const["theta"]
    
    vk1 = prox_F1_dual(vk+sigma*uk_bar,sigma,p,y,fourier_op)
    wk1 = prox_F2_dual(wk+sigma*linear_op.op(uk_bar),sigma,gamma,pn1)
    uk1 = prox_G(uk-tau*np.real(vk1)-tau*linear_op.adj_op(wk1),tau,epsilon,zeta)
    uk_bar1 = uk1+theta*(uk1-uk)

    norm = np.linalg.norm(uk1-uk)/np.linalg.norm(uk)
    norm += (np.linalg.norm(vk1-vk)+np.linalg.norm(wk1-wk))/(np.linalg.norm(vk)+np.linalg.norm(wk))
    return uk1,vk1,wk1,uk_bar1,norm


def pdhg(data,p,fourier_op,linear_op,param,const={},compute_energy=True,maxit=200,tol=1e-4):
    # --
    # -- MAIN LOWER LEVEL FUNCTION
    # --
    # INPUTS: - data: kspace measurements
    #         - p: subsampling mask. Same shape as the image.
    #         - param: lower level energy parameters
    #                  Must contain parameters keys "zeta","pn1","epsilon" and "gamma".
    #         - const: algorithm constants if we already know the values we want to use for tau and sigma
    #                  If not given, will compute them according to what is said in the article.
    #         - fourier_op: fourier operator from a full mask of same shape as the final image.
    #         - linear_op: linear operator used in regularisation functions
    #                      For the moment, only use waveletN.
    #         - compute_energy: bool, we compute et return energy over iterations if True
    #         - maxit,tol: We stop the algorithm when the norm of the difference between two steps 
    #                      is smaller than tol or after maxit iterations
    # OUTPUTS: - uk: final image
    #          - norms(, energy): evolution of stopping criterion (and energy if compute_energy is True)
    
    #Global parameters
    zeta=param["zeta"]
    pn1=param["pn1"]
    epsilon=param["epsilon"]
    gamma=param["gamma"]
    n_iter=0
    #Algorithm constants
    const = compute_constants(param,const,p)
    
    #Initializing
    uk = np.real(fourier_op.adj_op(p*data))
    vk = np.copy(uk)
    wk = linear_op.op(uk)
    uk_bar = np.copy(uk)
    norm = 2*tol
    #For plots
    if compute_energy:
        energy = []
    norms = []
    
    #Main loop
    t1 = time.time()
    while n_iter<maxit and norm>tol:
        uk,vk,wk,uk_bar,norm = step(uk,vk,wk,uk_bar,const,p,data,param,linear_op,fourier_op)
        n_iter += 1
        
        #Saving informations
        norms.append(norm)
        if compute_energy:
            energy.append(energy_wavelet(uk,p,data,pn1,gamma,zeta,epsilon,linear_op,fourier_op))
        
        #Printing
        if n_iter%10==0:
            if compute_energy:
                print(n_iter," iterations:\nCost:",energy[-1]
                      ,"\nNorm:",norm,"\n")
            else:
                print(n_iter," iterations:\nNorm:",norm,"\n")        
    print("Finished in",time.time()-t1,"seconds.")
    
    #Return
    if compute_energy:
        return uk,norms,energy
    else:
        return uk,norms

