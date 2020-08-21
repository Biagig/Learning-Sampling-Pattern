# -------------------------------------------------------------
# -- MAIN FUNCTION OPTIMIZING THE ENERGY WITH PDHG ALGORITHM --
# -------------------------------------------------------------
import numpy as np
from LowerLevel.prox import prox_F1_dual,prox_F2_dual,prox_G
from LowerLevel.cost_utils import energy_wavelet
import time
from modopt.math.metrics import ssim
from mri.operators.utils import gridded_inverse_fourier_transform_nd

def compute_constants(param,const,p):
    # --
    # -- Computes constants needed for the lower level algorithm
    # --
    # INPUT: param: dict of energy constants gamma and  epsilon
    #        const: dict containing specific values for tau and sigma we want to use.
    #               OR empty dict; in that case we compute their values according to the article.
    # OUTPUT: dict of pdhg constants    
    L=1
    #From Sherry's code, don't know if there is a way to find something better
    eta = max(np.amax(p)**2,p[-1]*4/param["gamma"])
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


def step(uk,vk,wk,uk_bar,const,p,pn1,y,param,linear_op,fourier_op,mask_type):
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
    tau=const["tau"]
    epsilon = param["epsilon"]
    theta = const["theta"]
    (n1,n2) = uk.shape

    vk1 = prox_F1_dual(vk+sigma*uk_bar,sigma,p,y,fourier_op,mask_type)

    wk1 = prox_F2_dual(wk+sigma*linear_op.op(uk_bar),sigma,gamma,pn1,n1*n2)
    uk1 = prox_G(uk-tau*vk1-tau*linear_op.adj_op(wk1),tau,epsilon)
    uk_bar1 = uk1+theta*(uk1-uk)

    norm = np.linalg.norm(uk1-uk)/np.linalg.norm(uk)
    norm += (np.linalg.norm(vk1-vk)+np.linalg.norm(wk1-wk))/(np.linalg.norm(vk)+np.linalg.norm(wk))
    return uk1,vk1,wk1,uk_bar1,norm


def pdhg(data,p,**kwargs):
    # --
    # -- MAIN LOWER LEVEL FUNCTION
    # --
    # INPUTS: - data: kspace measurements
    #         - p: p[:-1]=subsampling mask S(p), p[-1]=regularisation parameter alpha(p)
    #                   So len(p)=len(data)+1
    #         - fourier_op: fourier operator from a full mask of same shape as the final image.
    #         - linear_op: linear operator used in regularisation functions
    #                      For the moment, only use waveletN.
    #         - param: lower level energy parameters
    #                  Must contain parameters keys "epsilon" and "gamma".
    #           mask_type (optional): type of mask used ("cartesian", "radial"). Assume a cartesian mask if not given.
    # --
    # OPTIONAL INPUTS:
    #         - const: algorithm constants if we already know the values we want to use for tau and sigma
    #                  If not given, will compute them according to what is said in the article.
    #         - compute_energy: bool, we compute and return energy over iterations if True (default: False)
    #         - ground_truth: matrix representing the true image the data come from (default: None). If not None, we compute the ssim over iterations.
    #         - maxit,tol: We stop the algorithm when the norm of the difference between two steps 
    #                      is smaller than tol or after maxit iterations (default: 200, 1e-4)
    # --
    # OUTPUTS: - uk: final image
    #          - norms(, energy, ssims): evolution of stopping criterion (and energy if compute_energy is True / ssims if ground_truth not None)
    
    fourier_op = kwargs.get("fourier_op",None)
    linear_op = kwargs.get("linear_op",None)
    param = kwargs.get("param",None)
    if fourier_op is None: raise ValueError("A fourier operator fourier_op must be given")
    if linear_op is None: raise ValueError("A linear operator linear_op must be given")
    if param is None: raise ValueError("Lower level parameters must be given")
    mask_type = kwargs.get("mask_type","")


    const = kwargs.get("const",{})
    compute_energy = kwargs.get("compute_energy",False)
    ground_truth = kwargs.get("ground_truth",None)
    maxit = kwargs.get("maxit",200)
    tol = kwargs.get("tol",1e-6)
    verbose = kwargs.get("verbose",1)



    #Global parameters
    p,pn1 = p[:-1],p[-1]
    epsilon = param["epsilon"]
    gamma = param["gamma"]
    n_iter = 0
    #Algorithm constants
    const = compute_constants(param,const,p)
    if verbose >= 0:print("Sigma:",const["sigma"],"\nTau:",const["tau"])
    
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
        uk,vk,wk,uk_bar,norm = step(uk,vk,wk,uk_bar,const,p,pn1,data,param,linear_op,fourier_op,mask_type)
        n_iter += 1
        
        #Saving informations
        norms.append(norm)
        if compute_energy:
            energy.append(energy_wavelet(uk,p,pn1,data,gamma,epsilon,linear_op,fourier_op))
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

