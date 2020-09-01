# ---------------------------------------------
# -- DEFINITION OF UPPER LEVEL COST FUNCTION --
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg,LinearOperator
import time
from LowerLevel.pdhg import *
from UpperLevel.parametrisation import *
from UpperLevel.hessians import Du2_Etot,Dpu_Etot


# -- Base functions --
# --------------------
# -- Efficiency of the reconstruction --
def L(u,u2,c):return c/2 * np.linalg.norm(u.flatten()-u2.flatten())**2
def Du_L(u,u2,c):return c*(u-u2)

# -- Penalisation --
c=5
def P(p,beta):return beta*np.sum(p[:-1]*(1+c*(1-p[:-1])))
def grad_P(p,beta):
    Dp = np.zeros(p.shape)
    Dp[:-1] = beta*(1+(c-2*c*p[:-1]))
    return Dp


# -- Energy function --
# ---------------------
def E(**kwargs):
    # -- 
    # -- Computes the cost of a given mask or parametrisation of mask --
    # -- 
    # INPUTS:   images: list of all images used to evaluate the reconstruction
    #           kspace_data: list of noised kspace data associated to these images
    #           param: list of lower and upper level parameters. Must contain:
    #                   - epsilon: weight of L2 norm in lower level reconstruction
    #                   - gamma: parameter for approximation of L1 norm
    #                   - c: weight of L for upper level cost function
    #                   - beta: weight of P for upper level cost function
    #               Remark: You should choose these parameters to have E(pk)~1 at the beginning (otherwire, L-BFGS-B may stop too early)
    #           fourier_op: Fourier operator for lower level reconstruction
    #           linear_op: Linear operator for lower level reconstruction
    #           mask_type (optional): learn cartesian mask if mask_type="cartesian", otherwise each point is independant.
    #                                   Other parametrisations may be implemented later.
    #           pk: initial mask. Mandatory if mask_type!="cartesian"
    #           lk: initial mask parametrisation. Mandatory if mask_type="cartesian"
    #           verbose (optional)

    # Getting parameters
    images = kwargs.get("images",None)       
    kspace_data = kwargs.get("kspace_data",None)
    param = kwargs.get("param",None)

    mask_type = kwargs.get("mask_type","")
    lk = kwargs.get("lk",None)
    pk = kwargs.get("pk",None)

    verbose = kwargs.get("verbose",0)
    if verbose>=0:print("\n\nEVALUATING E(p)")
    

    # Checking inputs errors
    if images is None or len(images)<1:raise ValueError("At least one image is needed")
    if param is None:raise ValueError("Lower level parameters must be given")
    if len(images)!=len(kspace_data):raise ValueError("Need as many images and kspace data")


    #Compute P(pk/lk)
    Ep = 0
    if mask_type == "cartesian":
        pk = pcart(lk)
        Ep = P(lk,param["beta"])
    elif mask_type == "radial_CO":
        n_rad = kwargs.get("n_rad")
        pk = pradCO(lk,n_rad)
        Ep = P(lk,param["beta"])
    else:Ep = P(pk,param["beta"])

    #Compute L(pk/lk)
    Nimages = len(images)
    for i in range(Nimages):
        if verbose>=0:print(f"\nImage {i+1}:")
        u0_mat,y = images[i],kspace_data[i]

        if verbose>0:print("\nStarting PDHG")
        uk,_ = pdhg(y , pk , maxit = 50 , **kwargs)
        
        Ep += L(uk,u0_mat,param["c"])/Nimages
    
    return Ep



# -- Gradient of cost function --
# -------------------------------
def grad_L(**kwargs):
    # -- 
    # -- Compute gradient of L with respect to p
    # --
    # INPUTS:   pk: Point where we want to compute the gradient         
    #           u0_mat: Ground_truth image
    #           y: kspace data associated to u0_mat
    #           param: list of lower and upper level parameters. Must contain:
    #                   - epsilon: weight of L2 norm in lower level reconstruction
    #                   - gamma: parameter for approximation of L1 norm
    #                   - c: weight of L for upper level cost function
    #                   - beta: weight of P for upper level cost function
    #               Remark: You should choose these parameters to have E(pk)~1 at the beginning (otherwire, L-BFGS-B may stop too early)
    #           fourier_op: Fourier operator for lower level reconstruction
    #           linear_op: Linear operator for lower level reconstruction
    #
    #           max_cgiter (optional): maximum number of Conjugate Gradient iterations (default: 3000)
    #           cgtol (optional): tolerance of Conjugate Gradient iterations (default: 1e-6)
    #           compute_conv (optional): plot convergence if True (default: False)
    #           verbose (optional)

    # -- Getting parameters
    max_cgiter = kwargs.get("max_cgiter",4000)
    cgtol = kwargs.get("cgtol",1e-6)
    compute_conv = kwargs.get("compute_conv",False)
    mask_type = kwargs.get("mask_type","")
    learn_mask = kwargs.get("learn_mask",True)
    learn_alpha = kwargs.get("learn_alpha",True)

    u0_mat = kwargs.get("u0_mat",None)
    param = kwargs.get("param",None)
    y = kwargs.get("y",None)
    pk = kwargs.get("pk",None)
    fourier_op = kwargs.get("fourier_op",None)
    linear_op = kwargs.get("linear_op",None)
    const = kwargs.get("const",{})
    

    verbose = kwargs.get("verbose",0)
    cg_conv = []
    
    if u0_mat is None:raise ValueError("A ground truth image u0_mat is needed")
    if y is None:raise ValueError("kspace data y are needed")
    n = len(u0_mat)
    
    # -- Compute uk from pk with lower level solver if not given
    if verbose>=0:print("\nStarting PDHG")
    uk,_ = pdhg(y , pk , mask_type = mask_type ,
                fourier_op = fourier_op , linear_op = linear_op , param = param,
                maxit = 50 , verbose = verbose , const = const)
    
    # -- Defining linear operator from pk and uk
    def mv(w):
        w_complex = np.reshape( w[:n**2]+1j*w[n**2:] , (n,n) )
        fx = np.reshape(Du2_Etot( uk , pk , w_complex ,
                                   eps=param["epsilon"],
                                   fourier_op=fourier_op,
                                   y=y,
                                   linear_op=linear_op,
                                   gamma=param["gamma"]),(n**2,))
        return np.concatenate([np.real(fx),np.imag(fx)])
    lin = LinearOperator((2*n**2,2*n**2),matvec=mv)


    if verbose>=0:print("\nStarting Conjugate Gradient method")
    t1=time.time()
    B = np.reshape(Du_L(uk,u0_mat,param["c"]),(n**2,))
    B_real = np.concatenate([np.real(B),np.imag(B)])
    def cgcall(x):
        #CG callback function to plot convergence
        if compute_conv:cg_conv.append(np.linalg.norm(lin(x)-B_real)/np.linalg.norm(B_real))
    
    x_inter,_ = cg(lin,B_real,tol=cgtol,maxiter=max_cgiter,callback=cgcall)
    if verbose>=0:print(f"Finished in {time.time()-t1}s - ||Ax-b||/||b||: {np.linalg.norm(lin(x_inter)-B_real)/np.linalg.norm(B_real)}")

    
    # -- Plotting
    if compute_conv:
        plt.plot(cg_conv)
        plt.yscale("log")
        plt.title("Convergence of the conjugate gradient")
        plt.xlabel("Number of iterations")
        plt.ylabel("||Ax-b||/||b||")
        #plt.savefig("Upper Level/CG_conv.png")
    
    
    if np.linalg.norm(lin(x_inter)-B_real)/np.linalg.norm(B_real)>1e-3: return np.zeros(pk.shape)
    else: return -Dpu_Etot(uk,pk,np.reshape( x_inter[:n**2]+1j*x_inter[n**2:] , (n,n) ),
                               eps=param["epsilon"],
                               fourier_op=fourier_op,
                               y=y,
                               linear_op=linear_op,
                               gamma=param["gamma"],
                               learn_mask = learn_mask,
                               learn_alpha = learn_alpha)


# -- Definition of gradient of E --
# ---------------------------------
def grad_E(**kwargs):
    # -- 
    # -- Compute gradient of E with respect to p or l
    # --
    # INPUTS:   mask_type: "cartesian" to learn cartesian line; learn points otherwise
    #           pk: Point where we want to compute the gradient. Mandatory if mask_type != "cartesian"         
    #           lk: Mask parametrisation where we want to compute gradient. Mandatory if mask_type == "cartesian"
    #           images: list of all images used to evaluate the reconstruction
    #           kspace_data: list of noised kspace data associated to these images
    #           param: list of lower and upper level parameters. Must contain:
    #                   - epsilon: weight of L2 norm in lower level reconstruction
    #                   - gamma: parameter for approximation of L1 norm
    #                   - c: weight of L for upper level cost function
    #                   - beta: weight of P for upper level cost function
    #               Remark: You should choose these parameters to have E(pk)~1 at the beginning (otherwire, L-BFGS-B may stop too early)
    #           fourier_op: Fourier operator for lower level reconstruction
    #           linear_op: Linear operator for lower level reconstruction
    #
    #           max_cgiter (optional): maximum number of Conjugate Gradient iterations (default: 3000)
    #           cgtol (optional): tolerance of Conjugate Gradient iterations (default: 1e-6)
    #           compute_conv (optional): plot convergence if True (default: False)
    #           verbose (optional)
    # Getting parameters
    images = kwargs.get("images",None)       
    kspace_data = kwargs.get("kspace_data",None)
    param = kwargs.get("param",None)
    
    mask_type = kwargs.get("mask_type","")
    lk = kwargs.get("lk",None)
    pk = kwargs.get("pk",None)
    n_rad = kwargs.get("n_rad",0)

    if mask_type == "cartesian":kwargs["pk"] = pcart(lk)
    if mask_type == "radial_CO":kwargs["pk"] = pradCO(lk,n_rad)
    
    verbose = kwargs.get("verbose",0)
    
    if images is None or len(images)<1:raise ValueError("At least one image is needed")
    if len(images)!=len(kspace_data):raise ValueError("Need as many images and kspace data")


    #Computing gradient
    Nimages = len(images)

    gEp = np.zeros(len(kspace_data[0])+1)
    if verbose>=0:print("\n\nEVALUATING GRAD_E(p)")
    
    #Gradient with respect to p in pk
    for i in range(Nimages):
        if verbose>=0:print(f"\nImage {i+1}:")
        gEp += grad_L(u0_mat=images[i],y=kspace_data[i],**kwargs)/Nimages


    #Last operation if parametrisation
    if mask_type == "cartesian":
        gEp = grad_pcart(lk,gEp)
        return gEp+grad_P(lk,param["beta"])
    if mask_type == "radial_CO":
        gEp = grad_pradCO(lk,gEp)
        return gEp+grad_P(lk,param["beta"])

    else:return gEp+grad_P(pk,param["beta"])