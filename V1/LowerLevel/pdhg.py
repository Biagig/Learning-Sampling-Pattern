# -------------------------------------------------------------
# -- MAIN FUNCTION OPTIMIZING THE ENERGY WITH PDHG ALGORITHM --
# -------------------------------------------------------------
import numpy as np
from LowerLevel.prox import prox_F1_dual,prox_F2_dual,prox_G
from LowerLevel.cost_utils import energy_wavelet
import time
from modopt.math.metrics import ssim

def compute_constants(param,const,p):
    # --
    # -- Computes constants needed for the lower level algorithm
    # --
    # INPUT: param: dict of energy constants gamma,  epsilon and pn1
    #        const: dict containing specific values for tau and sigma we want to use.
    #               OR empty dict; in that case we compute their values according to the article.
    # OUTPUT: dict of pdhg constants    
    L=1
    #From Sherry's code, don't know if there is a way to find something better
    eta = max(np.amax(p)**2,param["pn1"]*4/param["gamma"])
    mu = 2*np.sqrt(param["epsilon"]/(1+L**2)/eta)
    theta = 1/(1+mu)

    const["L"]=L
    const["eta"]=eta
    const["mu"]=mu
    const["theta"]=theta
    if not ("sigma" in const.keys()) or not ("tau" in const.keys()):
        const["tau"] = mu/2/param["epsilon"]
        const["sigma"] = mu*eta/2
    
    return const


def step(uk,vk,wk,uk_bar,const,p,y,param,linear_op,fourier_op):
    # --
    # -- Computes a step of the pdhg algorithm
    # --
    # INPUTS: - uk,vk,wk,uk_bar: values after k iterations of the pdhg algorithm
    #         - const,param: dicts of pdhg constants and energy parameters
    # OUTPUTS: - uk1,vk1,wk1,uk_bar1: values after k+1 iterations of the algorithm
    #          - norm: value of the stopping criterion of the algorithm
    
    #Getting useful parameters
    sigma = const["sigma"]
    gamma = param["gamma"]
    pn1 = param["pn1"]
    tau=const["tau"]
    epsilon = param["epsilon"]
    theta = const["theta"]
    (n1,n2) = uk.shape
    
    vk1 = prox_F1_dual(vk+sigma*uk_bar,sigma,p,y,fourier_op)
    wk1 = prox_F2_dual(wk+sigma*linear_op.op(uk_bar),sigma,gamma,pn1,n1*n2)
    uk1 = prox_G(uk-tau*vk1-tau*linear_op.adj_op(wk1),tau,epsilon)
    uk_bar1 = uk1+theta*(uk1-uk)

    norm = np.linalg.norm(uk1-uk)/np.linalg.norm(uk)
    norm += (np.linalg.norm(vk1-vk)+np.linalg.norm(wk1-wk))/(np.linalg.norm(vk)+np.linalg.norm(wk))
    return uk1,vk1,wk1,uk_bar1,norm


def pdhg(data,p,fourier_op,linear_op,param,const={},compute_energy=True,ground_truth=None,maxit=200,tol=1e-4,verbose=1):
    # --
    # -- MAIN LOWER LEVEL FUNCTION
    # --
    # INPUTS: - data: kspace measurements
    #         - p: subsampling mask. Same shape as the image.
    #         - param: lower level energy parameters
    #                  Must contain parameters keys "pn1","epsilon" and "gamma".
    #         - const: algorithm constants if we already know the values we want to use for tau and sigma
    #                  If not given, will compute them according to what is said in the article.
    #         - fourier_op: fourier operator from a full mask of same shape as the final image.
    #         - linear_op: linear operator used in regularisation functions
    #                      For the moment, only use waveletN.
    #         - compute_energy: bool, we compute and return energy over iterations if True
    #         - ground_truth: matrix representing the true image the data come from. If not None, we compute the ssim over iterations.
    #         - maxit,tol: We stop the algorithm when the norm of the difference between two steps 
    #                      is smaller than tol or after maxit iterations
    # OUTPUTS: - uk: final image
    #          - norms(, energy): evolution of stopping criterion (and energy if compute_energy is True)
    
    #Global parameters
    pn1=param["pn1"]
    epsilon=param["epsilon"]
    gamma=param["gamma"]
    n_iter=0
    #Algorithm constants
    const = compute_constants(param,const,p)
    print("Sigma:",const["sigma"],"\nTau:",const["tau"])
    
    #Initializing
    uk = fourier_op.adj_op(p*data)
    vk = np.copy(uk)
    wk = linear_op.op(uk)
    uk_bar = np.copy(uk)
    norm = 2*tol

    #For plots
    if compute_energy:
        energy = []
    if ground_truth is not None:
        ssims=[]
    norms = []
    
    #Main loop
    t1 = time.time()
    while n_iter<maxit and norm>tol:
        uk,vk,wk,uk_bar,norm = step(uk,vk,wk,uk_bar,const,p,data,param,linear_op,fourier_op)
        n_iter += 1
        
        #Saving informations
        norms.append(norm)
        if compute_energy:
            energy.append(energy_wavelet(uk,p,data,pn1,gamma,epsilon,linear_op,fourier_op))
        if ground_truth is not None:
            ssims.append(ssim(uk,ground_truth))
        
        #Printing
        if n_iter%10==0 and verbose>0:
            if compute_energy:
                print(n_iter," iterations:\nCost:",energy[-1]
                      ,"\nNorm:",norm,"\n")
            else:
                print(n_iter," iterations:\nNorm:",norm,"\n")
    if verbose>=0:      
        print("Finished in",time.time()-t1,"seconds.")
    
    #Return
    if compute_energy and ground_truth is not None:
        return uk,norms,energy,ssims
    elif ground_truth is not None:
        return uk,norms,ssims
    elif compute_energy:
        return uk,norms,energy
    else:
        return uk,norms

