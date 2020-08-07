# --------------------------------------------
# -- MAIN UPPER LEVEL OPTIMIZATION FUNCTION --
# --------------------------------------------
import time
import numpy as np
from UpperLevel.cost_utils import E,grad_E
from UpperLevel.parametrisation import pcart,grad_pcart
from scipy.optimize.lbfgsb import fmin_l_bfgs_b


class Mask_Learner(object):
    def __init__(self,**kwargs):
        self.energy_upper = []
        self.alphas = []
        self.niter = 0

        #Lower level parameters
        self.fourier_op = kwargs.get("fourier_op",None)
        self.linear_op = kwargs.get("linear_op",None)
        self.param = kwargs.get("param",None)
        self.const = kwargs.get("const",{})

        #Upper level parameters
        self.images = kwargs.get("images",None)
        self.kspace_data = kwargs.get("kspace_data",None)
        self.verbose = kwargs.get("verbose",-1)
        self.maxfun = kwargs.get("maxfun",20)
        self.maxiter = kwargs.get("maxiter",20)


    # -- Callback function --
    # -----------------------
    def fcall(self,x,mask_type):
        self.niter += 1
        if mask_type == "cartesian":Ep = E(lk=x , mask_type="cartesian"  , images=self.images , kspace_data=self.kspace_data ,
                                                    fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                                                    verbose=self.verbose , const=self.const)
        else:Ep = E(pk=x , mask_type=mask_type  , images=self.images , kspace_data=self.kspace_data ,
                        fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                        verbose=self.verbose , const=self.const)
        self.energy_upper.append(Ep)
        self.alphas.append(x[-1])
        print("\033[1m" + f"\n{self.niter} iterations: E(p)={Ep}, alpha={x[-1]}\n\n" + "\033[0m")
    

    # -- Main function --
    # -------------------
    def optimize(self,**kwargs):
        # -- Getting parameters --
        # ------------------------
        #Upper level inputs
        mask_type = kwargs.get("mask_type","")
        l0 = kwargs.get("l0",None)
        p0 = kwargs.get("p0",None)


        # -- Checking inputs --
        if mask_type == "cartesian" and l0 is None: raise ValueError("an initial mask parametrisation l0 must be given")
        if mask_type != "cartesian" and p0 is None: raise ValueError("an initial mask p0 must be given")


        t1 = time.time()
        self.niter=0

        #Case 1: Learning a cartesian mask
        if mask_type == "cartesian":
            #Initializing
            n = len(l0)-1
            self.energy_upper = [E(lk=l0 , mask_type="cartesian"  , images=self.images , kspace_data=self.kspace_data ,
                                                    fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                                                    verbose=self.verbose , const=self.const)]
            self.alphas = [l0[-1]]

            #Using L-BFGS-B
            lf,_,_ = fmin_l_bfgs_b(lambda x:E(lk=x , mask_type="cartesian" , images=self.images , kspace_data=self.kspace_data ,
                                                    fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                                                    verbose=self.verbose , const=self.const) , 
                                        l0,
                                        lambda x:grad_E(lk=x , mask_type="cartesian" ,images=self.images,kspace_data=self.kspace_data,
                                                        fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                                                        verbose=self.verbose , const=self.const),
                                        bounds=[(0,1)]*n+[(1e-10,np.inf)],pgtol=1e-6,
                                        maxfun=self.maxfun , maxiter=self.maxiter , maxls=2 ,
                                        callback = lambda x:self.fcall(x,mask_type))

            #Returning output
            print("\033[1m" + f"\nFINISHED IN {time.time()-t1} SECONDS\n" + "\033[0m")
            return lf,self.energy_upper,self.alphas
        


        #Case 2: Learning points
        else:
            #Initializing
            n = int(np.sqrt(len(p0)))
            self.energy_upper = [E(pk=p0 , mask_type="" , images=self.images , kspace_data=self.kspace_data ,
                                                    fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                                                    verbose=self.verbose , const=self.const)]
            self.alphas = [p0[-1]]

            #Using L-BFGS-B
            pf,_,_ = fmin_l_bfgs_b(lambda x:E(pk=x , mask_type=""  , images=self.images , kspace_data=self.kspace_data ,
                                                    fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                                                    verbose=self.verbose , const=self.const) , 
                                        p0,
                                        lambda x:grad_E(pk=x , mask_type="" ,images=self.images,kspace_data=self.kspace_data,
                                                        fourier_op=self.fourier_op , linear_op=self.linear_op , param=self.param , 
                                                        verbose=self.verbose , const=self.const),
                                        bounds=[(0,1)]*n**2+[(1e-10,np.inf)],pgtol=1e-6,
                                        maxfun=self.maxfun , maxiter=self.maxiter , maxls=2,
                                        callback = lambda x:self.fcall(x,""))

            #Returning output
            print("\033[1m" + f"\nFINISHED IN {time.time()-t1} SECONDS\n" + "\033[0m")
            return pf,self.energy_upper,self.alphas
