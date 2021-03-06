# --------------------------------------------
# -- MAIN UPPER LEVEL OPTIMIZATION FUNCTION --
# --------------------------------------------
import time
import numpy as np
from UpperLevel.cost_utils import E,grad_E
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from mri.operators import NonCartesianFFT, WaveletUD2, WaveletN


# -- Implements the upper level algorithm --
# ------------------------------------------
# --
# -- INPUTS FOR ANY INSTANCE:
#               fourier_op, linear_op, param, const (optional) for PDHG
#               images: list of all images used to evaluate the reconstruction
#               kspace_data: list of noised kspace data associated to these images
#               maxfun, maxiter (optional): parameteres for L-BFGS-B (see scipy doc if needed)
#               verbose (optional)
#               n_rad: number of radii (for radial mask only)
# --
# -- METHODS:
#               optimize: main function. Returns a mask learned and two lists energy_upper and alphas showing the evolution
#                         of these parameters after each iteration
#               fcall: callback function for L-BFGS-B. Stores informations in energy_upper and alphas and prints it

class Mask_Learner(object):
    def __init__(self,**kwargs):
        #For plots and returns
        self.energy_upper = []
        self.alphas = []
        self.niter = 0

        #Lower level parameters
        self.param = kwargs.get("param",None)
        self.const = kwargs.get("const",{})

        #Upper level parameters
        self.images = kwargs.get("images",None)
        self.kspace_data = kwargs.get("kspace_data",None)
        self.samples = kwargs.get("samples",[])
        self.wavelet_name = kwargs.get("wavelet_name","")
        self.wavelet_scale = kwargs.get("wavelet_scale",1)

        self.n_rad = kwargs.get("n_rad",0)
        self.verbose = kwargs.get("verbose",-1)
        self.maxfun = kwargs.get("maxfun",20)
        self.maxiter = kwargs.get("maxiter",20)
        self.pgtol = kwargs.get("pgtol",1e-6)


    # -- Callback function --
    # -----------------------
    def fcall(self,x,mask_type):
        self.niter += 1
        if mask_type in ["radial_CO","cartesian"]:Ep = E(lk=x , mask_type=mask_type  , images=self.images , kspace_data=self.kspace_data ,
                                                    samples= self.samples , wavelet_name = self.wavelet_name , wavelet_scale = self.wavelet_scale , param=self.param , 
                                                    verbose=self.verbose , const=self.const , n_rad = self.n_rad)
        else:Ep = E(pk=x , mask_type=mask_type  , images=self.images , kspace_data=self.kspace_data ,
                        samples= self.samples , wavelet_name = self.wavelet_name , wavelet_scale = self.wavelet_scale , param=self.param , 
                        verbose=self.verbose , const=self.const)
        self.energy_upper.append(Ep)
        self.alphas.append(x[-1])
        print("\033[1m" + f"\n{self.niter} iterations: E(p)={Ep}, alpha={x[-1]}\n\n" + "\033[0m")
    

    # -- Main function --
    # -------------------
    def optimize(self, parallel = False , parallel_verbose = 0 , **kwargs):
        # -- Getting parameters --
        # ------------------------
        #Upper level inputs
        mask_type = kwargs.get("mask_type","")
        learn_mask = kwargs.get("learn_mask",True)
        learn_alpha = kwargs.get("learn_alpha",True)
        l0 = kwargs.get("l0",None)
        p0 = kwargs.get("p0",None)
        shots = False


        # -- Checking inputs --
        if mask_type in ["cartesian","radial_CO"]:
            if l0 is None: raise ValueError("an initial mask parametrisation l0 must be given")
            shots = True
        else:
            if p0 is None: raise ValueError("an initial mask p0 must be given")


        t1 = time.time()
        self.niter=0


        # -- Initializing --
        # ------------------
        print("Multithread:",parallel)

        if shots:
            n = len(l0)-1
            self.alphas = [l0[-1]]
        else:
            n = len(p0)-1
            self.alphas = [p0[-1]]
            
        self.energy_upper = [E(lk=l0 , pk=p0 , mask_type=mask_type  , images=self.images , kspace_data=self.kspace_data ,
                                                samples= self.samples , wavelet_name = self.wavelet_name , wavelet_scale = self.wavelet_scale , param=self.param , 
                                                verbose=self.verbose , const=self.const , n_rad = self.n_rad , parallel = parallel)]
        

        # -- Using L-BFGS-B --
        # --------------------
        if shots:
            #Optimize l
            lf,_,_ = fmin_l_bfgs_b(lambda x:E(lk=x , mask_type=mask_type , images=self.images , kspace_data=self.kspace_data ,
                                                    samples= self.samples , wavelet_name = self.wavelet_name , wavelet_scale = self.wavelet_scale , param=self.param , 
                                                    verbose=self.verbose , const=self.const , n_rad = self.n_rad , parallel = parallel , parallel_verbose = parallel_verbose) , 
                                        l0,
                                        lambda x:grad_E(lk=x , mask_type=mask_type ,images=self.images,kspace_data=self.kspace_data,
                                                        samples= self.samples , wavelet_name = self.wavelet_name , wavelet_scale = self.wavelet_scale , param=self.param , 
                                                        verbose=self.verbose , const=self.const , n_rad = self.n_rad ,
                                                        learn_mask = learn_mask , learn_alpha = learn_alpha , parallel = parallel , parallel_verbose = parallel_verbose),
                                        bounds=[(0,1)]*n+[(1e-10,np.inf)],pgtol=self.pgtol,
                                        maxfun=self.maxfun , maxiter=self.maxiter , maxls=2 ,
                                        callback = lambda x:self.fcall(x,mask_type))

        else:
            #Optimize p directly
            pf,_,_ = fmin_l_bfgs_b(lambda x:E(pk=x , mask_type=mask_type , images=self.images , kspace_data=self.kspace_data ,
                                                    samples= self.samples , wavelet_name = self.wavelet_name , wavelet_scale = self.wavelet_scale , param=self.param , 
                                                    verbose=self.verbose , const=self.const , n_rad = self.n_rad , parallel = parallel , parallel_verbose = parallel_verbose) , 
                                        p0,
                                        lambda x:grad_E(pk=x , mask_type=mask_type ,images=self.images,kspace_data=self.kspace_data,
                                                        samples= self.samples , wavelet_name = self.wavelet_name , wavelet_scale = self.wavelet_scale , param=self.param , 
                                                        verbose=self.verbose , const=self.const , n_rad = self.n_rad ,
                                                        learn_mask = learn_mask , learn_alpha = learn_alpha , parallel = parallel , parallel_verbose = parallel_verbose),
                                        bounds=[(0,1)]*n+[(1e-10,np.inf)],pgtol=self.pgtol,
                                        maxfun=self.maxfun , maxiter=self.maxiter , maxls=2 ,
                                        callback = lambda x:self.fcall(x,mask_type))


        # -- Returning output --
        # ----------------------
        print("\033[1m" + f"\nFINISHED IN {time.time()-t1} SECONDS\n" + "\033[0m")
        if shots:return lf,self.energy_upper,self.alphas
        else:return pf,self.energy_upper,self.alphas