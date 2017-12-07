# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:53:32 2017

@author: harry.matthews
"""
import pandas as pa
import numpy as np
from helpers import mynormpdf, configure_input_matrices
import copy



class Linear_regression(object):
    """Abstract base class for linear regressions """
    def __init__(self):
        super(Linear_regression,self).__init__()
    
    def predict(self, X, original_space=False):
        """Predict block of Y-variables, given X 
        INPUTS:
            X - an n (observations) x p(variables) matrix of predictor variables
            original_space - if True it will assume that X has not been centered and will center it
                            and will add the mean of Y back onto the predicted Y-block, so that input from the original space of X variables results in a prediction in the original space of Y variables
        OUTPUTS:
            Y - an n (observations) x o(variables matrix of outcome variables
        

        """
        if isinstance(X,float):
            X = np.array([[X]])
            
        if X.ndim == 1:
            X = np.atleast_2d(X).T
        
        if original_space==True:
            X = X-self.Xmean
            
       
        
        if any([item>1 for item in self.coefs.shape]): # if any axis has only one element
            
            Y = np.dot(X, self.coefs)
        else:
            Y = X*self.coefs.flatten()
        
        
        if original_space==True:
            Y = Y+self.Ymean
        
        
        return Y
    




class PLSR(Linear_regression):
    """
    Implements a partial least-squares regression by SIMPLS, with observation weights added
        REFERENCES:  De Jong, S. (1993). SIMPLS: an alternative approach to partial least squares regression. Chemometrics and intelligent laboratory systems, 18(3), 251-263.
    """
    
    def __init__(self):

        super(PLSR,self).__init__()
        self.X0 = None
        self.Y0 = None
    
    def fit(self,X, Y,ncomps='full_rank',indices='All',obs_weights=None,override_mean_centering=False): 
        """
        Fits the regression using the SIMPLS algorithm - columns are mean-centred internally but not scaled to have unit variance
        
        INPUTS:
            X - a pandas DataFrame where columns correspond to predictor variables and rows to observations
            Y - a pandas DataFrame where columns correspond to outocome variables and rows to observations
            ncomps - the number of latent components to include in the regression if ncomps = 'full_rank' (default) this will include all components, whihc is equal to equal to np.min([len(X.index), len(X.columns)])
            indices - the indices denoting which observations (rows) to include in the regression
            obs_weights - weights to apply to each observation in the regression - if None then all observations will be weighted equally
            override_mean_centering - if True columns of the data matrices will not be mean_centered
        NOTES:
            X and Y must have corresponding indices in their Index, and contain only numeric values the regression will treat all variables as though they were continuous 
            
        """
        
        # Trim X and Y to include only required cases and ensure rows of x and Y correspond
        X,Y,Xmat,Ymat = configure_input_matrices(X,Y,indices)
        
        
        if ncomps=='full_rank': # fit with all components
            ncomps = np.min([len(X.index)-1, len(X.columns)])
        elif isinstance(ncomps, int)==False:
            raise TypeError('PLSR:  ncomps must either be an integer or \'full_rank\' but value ' + str(ncomps) + 'of type '+ str(type(ncomps))+'was specified')
    
        
        if ncomps>np.min([len(X.index)-1, len(X.columns)]):
            raise ValueError('Number of components in a PLS-regression should be less than or equal to the smaller of (number of observations in x-1, number of variables in x) i.e. should be less than '+ str(np.min([len(X.index)-1, len(X.columns)]))+' but value '+str(ncomps)+'was specified') 
        
                
        
        if obs_weights is not None: 
            #ensure weights array is 2d
            if obs_weights.ndim==1:
                obs_weights = np.atleast_2d(obs_weights).T
        else:

            obs_weights = np.ones([Xmat.shape[0],1])
        
        # compute weighted mean
        if override_mean_centering==True:
            Xmean = np.zeros(Xmat.shape[1])
            Ymean = np.zeros(Ymat.shape[1])
        else:
        
            Xmean = np.atleast_2d(np.sum(Xmat*obs_weights,axis=0)/np.sum(obs_weights))
            Ymean = np.atleast_2d(np.sum(Ymat*obs_weights,axis=0)/np.sum(obs_weights))
        
        
        #mean center
        X0=Xmat-Xmean
        Y0=Ymat-Ymean
        
        # apply weightings
        X0 = X0*obs_weights
        Y0 = Y0*obs_weights

        # keep info for later
        self.X0 = pa.DataFrame(columns=X.columns,index = X.index,data=X0)
        self.Y0 = pa.DataFrame(columns=Y.columns,index = Y.index,data=Y0)
        
        self.Xmean = Xmean
        self.Ymean = Ymean
        self.ncomps = ncomps
        
        # The following implementaion of simpls is copied from MATLAB PLSREGRESS 
        n,dx=X0.shape
        dy = Y0.shape[1]

        # initialise arrays
        Xloadings = np.zeros([dx,ncomps])
        Yloadings = np.zeros([dy,ncomps])

        Xscores = np.zeros([n,ncomps])
        Yscores = np.zeros([n,ncomps])

        Weights = np.zeros([dx,ncomps])
        
        V = np.zeros([dx,ncomps])

        Cov = np.dot(np.transpose(X0),Y0)

        for i in range(ncomps):
            u,s,v = np.linalg.svd(Cov,full_matrices = 0)

            ri = u[:,0]
            ci = v[0,:]
            si = s[0]
            


            ti = np.dot(X0,ri) # projection onto ri
            normti = np.linalg.norm(ti)
            ti = ti/normti
            Xloadings[:,i] = np.dot(np.transpose(X0),ti)

            qi = si*ci/normti
            Yloadings[:,i] = qi

            Xscores[:,i] = ti
            Yscores[:,i]= np.dot(Y0,qi)

            Weights[:,i] = ri/normti

            vi = Xloadings[:,i]

            for repeat in xrange(2):
                for j in range(i):
                    vj = V[:,j]
                    vi = vi - np.dot(vj,vi)*vj


            vi = vi/np.linalg.norm(vi)
            V[:,i] = vi



            Cov = Cov-np.outer(vi,np.dot(vi,Cov))
            Vi = V[:,0:(i+1)]
            if i==0: 
                Cov = Cov-np.outer(Vi,np.dot(np.transpose(Vi),Cov))# Vi will be a single column, numpy will only do this operation using np.outer
            else:
                Cov = Cov-np.dot(Vi,np.dot(np.transpose(Vi),Cov))



        # Orthogonalise Y-scores
        for i in range(ncomps):
            ui = Yscores[:,i]
            for repeat in xrange(2):
                for j in range(i):
                    tj = Xscores[:,j]
                    ui = ui - np.dot(tj,ui)*tj

            Yscores[:,i] = ui

        self.Xloadings = Xloadings
        self.Yloadings = Yloadings
        self.Xscores = Xscores
        
        self.Yscores = Yscores
        self.Weights = Weights
        self.Yresiduals = Y0 - np.dot(self.Xscores,np.transpose(self.Yloadings))
        self.Xresiduals = X0 - np.dot(self.Xscores,np.transpose(self.Xloadings))
        self.predicted_values = np.dot(self.Xscores,np.transpose(self.Yloadings))

        self.coefs = np.atleast_2d(np.dot(self.Weights,np.transpose(self.Yloadings)))
  
       




#################Kernel regression##################################

class Kernel(object):
    """This class represents a single kernel in a kernel regression 
    implements the evaluation of a kernel regression at a given point
    
    NOTES:
        fit method computes two estimates of location, one is a weighted mean, (self.mean_location) the other is the location predicted from a weighted linear regression (self.regression_predicted_location) (See Hastie 2001, Ch 6 pg 169)
        this linear regression also provides an estimate of the tangent direction of the overall non-linear regression at the point (self.direction) and the rate of change at the given point (self.magnitude)
    
    REFERENCES:
        Hastie, T., Tibshirani, R., & Friedman, J. (2001). The elements of statistical learning. New York.
    
    """
    def __init__(self):
        super(Kernel,self).__init__()
    
    def __getattribute__(self,name):
        # Will raise more useful error messages if attempt to access unavailable attribute
        try:
            return super(Kernel,self).__getattribute__(name)
        except:
            raise AttributeError('Tried to access unavailable '+str(name)+' attribute of Kernel instance. You may have not yet fit the kernel by calling the \'fit\' method. You may have typed it wrong. You are probably after one of \'mean_location\', \'regression_predicted_location\', \'regression_coefs\'' )
   
    def fit(self,x,Y,kernel_center,**kwargs):
        """Evaluates the kernel regression of a univariate or multivariate Y onto univariate x
        INPUTS:
            x - a n(observations) x 1(predictor variable) pandas DataFrame cointaining the predictor variable
            Y - a n(observations) x o(outcome variables) pandas DataFrame cointaining the outcome variables
            kernel_center - the value of x at which to evaluate the function

            kwargs - keyword arguments
                            weighting_function - available functions are: 'gaussian' (default), 'triangular', 'uniform',  'epanechnikov' and 'tricube'
                            linear_regression_model - a class definition of the linear regression model to use to evaluate the kernel regression (default = PLSR)
                            bandwidth - (no default) the bandwidth to use
                            minimum_n - the minimum number of cases that must be within +/-2*bandwidth of the kernel center, if fewer cases then an error will be raised (default = 10)
                            
                            
        """
        kwargs = copy.deepcopy(kwargs) # explicitly copy kwargs to check there are no redundant keyword arguments by 'popping' them out as they are used and then check if there are any left, as dictionary will be reused for many kernesl we must do this in a copy otherwise all keyword arguments will be removed when evaluation the first kernel
        

        self.kernel_center = kernel_center
        self.x = x
        self.weighting_function = kwargs.pop('weighting_function', 'gaussian')
        self.linear_regression_model = kwargs.pop('linear_regression_model',PLSR)
        self.bandwidth = kwargs.pop('bandwidth')

        
        
        self.minimum_n = kwargs.pop('minimum_n',10)
        self.weighting_function = kwargs.pop('weighting_function', 'gaussian')
        self.linear_regression_model = kwargs.pop('linear_regression_model',PLSR)
            
        if any([isinstance(self.minimum_n,int)==False,self.minimum_n<1]):
                raise TypeError('minimum_n must be a non-zero integer but value ' + str(self.minimum_n)+ ' of type '+str(type(self.minimum_n)))
            
        # Check number of cases within given +/-2*bandwidth
        lower_lim = self.kernel_center-2*self.bandwidth
        upper_lim = self.kernel_center+2*self.bandwidth
        xmat = x.as_matrix().astype(float) # is also handy to have x as a matrix
        n = np.sum((xmat>lower_lim)*(xmat<upper_lim))
        if n < self.minimum_n:
                    raise ValueError('Only ' + str(n) + ' cases (which is less than the specified minimum allowable) within kernel boundary and adaptation has not been allowed so evaluation is impossible!!!')
        self.n = n 

        
        weights = compute_weights(xmat,self.bandwidth,self.kernel_center,weighting_function = self.weighting_function,norm=True)
        model = self.linear_regression_model()
        model.fit(x,Y,obs_weights=weights)
        
        # calculate expected locations and directions
        self.mean_location = model.Ymean
        self.regression_predicted_location = model.predict(self.kernel_center-model.Xmean)+model.Ymean
        self.regression_coefs = model.coefs
        
        if len(kwargs.keys())>0:
            raise ValueError('Keyword arguments '+ str(kwargs.keys())+ 'are not valid')

class Kernel_regression(object):
    """
    Implements a Nadaraya-Watson kernel regression of a univariate or multivariate block of Y-variables onto a single x-variable
    NOTES:
        computes two estimates of the expectation, when evaluated: one is a weighted mean, the other is the the expectation predicted from a local weighted linear regression (See Hastie 2001, Ch 6 pg 169)
        this linear regression also provides an estimate of the tangent direction of the overall non-linear regression at the point
    
    REFERENCES:
        Hastie, T., Tibshirani, R., & Friedman, J. (2001). The elements of statistical learning. New York.
    """
    def __init__(self):
        super(Kernel_regression,self).__init__()
    
    def fit(self,x,Y,preferred_expectation_estimate,**kwargs):
        
        """
        This method just sets parameters important to evaluate the model and checks inputs.
        INPUTS:
            x - a n(observations) x 1(predictor variable) pandas DataFrame cointaining the predictor variable
            Y - a n(observations) x o(outcome variables) pandas DataFrame cointaining the outcome variables
           
            preferred_expectation_estimate - this implementation computes two estimates of the expectation this argument defines which one to return by default from the 'predict' method: options are 'mean_location' or 'regression_predicted_location'
            
            kwargs - keyword arguments
                            weighting_function - available functions are: 'gaussian' (default), 'triangular', 'uniform',  'epanechnikov' and 'tricube'
                            linear_regression_model - a class definition of the linear regression model to use to evaluate the kernel regression (default = PLSR)
                            bandwidth - (no default) the bandwidth to use
                            minimum_n - the minimum number of cases that must be within +/-2*bandwidth of the kernel center, if fewer cases then an error will be raised (default = 10)

        """
        indices = kwargs.pop('indices','All')
        
        # trim x and Y to include only specified indices and ensure rows correspond across x and Y
        x,Y,xmat,ymat=configure_input_matrices(x,Y,indices)        
        
        # check preferres expectation estimate is valid
        if preferred_expectation_estimate not in ['regression_predicted_location', 'mean_location']:
            raise ValueError('preferred_expectation_estimate must be one of\'regression_predicted_location\', \'mean_location\'')
            
        self.preferred_expectation_estimate = preferred_expectation_estimate
        
        self.kernel_settings=kwargs
        self.x = x
        self.Y = Y
    
    
    
    
    
    
    
    def predict(self,x,parameters=None,original_space=None):
        """for each given value of x this evaluates the kernel regression at x and returns the value predicted by the model
        INPUTS: 
            x - a numpy array of values at which to evaluate the function
            paramaters - the parameter/s to return. Strictly, this could be any attribute of a multivariate_statistics.Kernel instance
                    however, the idea is to use this to return some parameter of interest about the model. By default this will be the estimate of the expectation specified
                    by the 'preferred_expectation_estimate' given to the fit method of this class. Other parameters of interest would be 'regression_coefs'
        
            original_space - this does nothing, but is for consistency with linear regression classes
        
        OUTPUTS:
            predicted_values - if paramaeters is a single string this will return a numpy array, where rows correspond to values in x. The number and contents of columns will depend to the parameter
                                if parameters is a list or tuple of strings specifying many attributes, this will return a list of arrays
                                
        """
        if parameters is None:
            parameters=self.preferred_expectation_estimate
        
        if isinstance(x, int):
            x = float(x)
        
        if isinstance(x,float):
            x = np.array([x])
        
        
        for i, value in enumerate(x):
            kernel_obj = Kernel()
            kernel_obj.fit(self.x, self.Y, value,**self.kernel_settings)
            
            if isinstance(parameters,str): # if just single string
                pred_value = getattr(kernel_obj,parameters)
                if isinstance(pred_value,(float,np.ndarray)):
                    pred_value = np.array(pred_value)
                    if i==0:
                        # initialise numpy array
                        pred_values = np.zeros([x.flatten().size,pred_value.flatten().size])
                    pred_values[i,:] = pred_value
                
                else:
                    
                    if i==0:
                        # initialise emptuy list
                        pred_values = []
                    pred_values.append(pred_value)
            
            elif hasattr(parameters,'__iter__'): # if iterable and not a string 
                if i==0:
                    pred_values = []
                
                for paramnum,paramname in enumerate(parameters):
                    
                    if isinstance(paramname,str):
                        pred_value = getattr(kernel_obj, paramname)
                        if isinstance(pred_value,(float,np.ndarray)):
                            pred_value = np.array(pred_value)
                            if i == 0:
                                pred_values.append(np.zeros([x.flatten().size,pred_value.flatten().size])) # initialise array
                            pred_values[paramnum][i,:] = pred_value
                
                        else:
                    
                            if i==0:
                                # initialise emptuy list
                                pred_values.append([])
                            pred_values[paramnum].append(pred_value)
                    else:
                        raise TypeError('List of parameters must contain names of attributes as string, have you not typed them in inverted commas?')
                    
            else:
                raise TypeError('parameters must be None a string or an iterable (e.g. list) of strings')
                
        return pred_values
    
    def get_kernel_at(self,x):
        """Returns Kernel instance for value of x"""
        kernel_obj = Kernel()
        kernel_obj.fit(self.x, self.Y, x,self.method,**self.kernel_settings)
        return kernel_obj
    
    def get_resampled_parameters(self,x,parameters=None,num_resamples=100):
        """Resample with replacement and calculate output parameters
        INPUTS:
            x - values of the predictor at which to evaluate the models
            parameters - parameters to extract from the model - must be either a string or a list of strings
            num_resamples - number of times to resample
        """
        
        
        #TODO allow parallel computing here
        for n in xrange(num_resamples):
            # resample indices with replacement
            inds = np.random.randint(len(self.x.index),size=len(self.x.index))
            # resample ids
            IDs = self.x.index[inds]
            
            # fit model to resampled data
           
            resampled_x = pa.DataFrame(data=self.x.loc[IDs,:].as_matrix().astype(float))
            resampled_Y = pa.DataFrame(data=self.Y.loc[IDs,:].as_matrix().astype(float))
            
            
            kernel_regression_obj = Kernel_regression()
            kernel_regression_obj.fit(resampled_x,resampled_Y,self.preferred_expectation_estimate,**self.kernel_settings)
            
            
            # return predicted values based on resample datasets
            pred_values = kernel_regression_obj.predict(x,parameters=parameters)
            
            if hasattr(parameters,'__iter__'): # if trying to return multiple parameters put each into separate array. all arrays will be contained in a single list
                    if n==0:
                        resampled_pred_values = []
                        for array in pred_values:
                            resampled_pred_values.append(np.zeros([array.shape[0],array.shape[1], num_resamples]))# initialise arrays
                        
                    for paramnum, array in enumerate(pred_values):
                        resampled_pred_values[paramnum][:,:,n] = array
                        
                            
            else: # just assign to a single array
                if n==0:
                    resampled_pred_values = np.zeros([pred_values.shape[0],pred_values.shape[1], num_resamples])
                resampled_pred_values[:,:,n] = pred_values
                                     
        return resampled_pred_values                






def compute_phenotype_score(Y, x,model1,model2):
        """Computes scores representing an observation's position along the vector between the expectations at value x on model 1 and at value x on model 2
            Y - an n (observations) x p (outcome variables) matrix of observatiomns to calculate scores for
            x - the values at which to calculate the expectations - must have n elements
            model1 - Kernel regression model of one class
            model2 - Kernel regression model of second class
        """
        # ensure Y is 2D
        Y = np.atleast_2d(Y)
        
        # calculate expectations
        p1 = model1.predict(x)
        p2 = model2.predict(x)
        

        # get vector from p1 to p2
        vec = p1 - p2
        

        
        # get vector length
        length = np.linalg.norm(vec,axis=1)
        
        
        #normalise vector to unit length
        vec = vec/np.atleast_2d(length).T
        
        # calculate reference point as halfway between expectations
        refpoint=(p1+p2)/2.
        
        # calculate projections relative to refpoint
        projections = np.sum(((Y-refpoint))*vec,axis=1)
        
       
        # normalise scale 1= p1 -1 =p2
        scores = projections/(length/2.)
        
        
        return scores
    


def compute_weights(x,bandwidth,center,weighting_function='gaussian',norm=True):
    """computes weights for each entry in x according to some weighting functions 
    INPUTS:
        x - a vector of values of some continuous variable
        bandwidth - the 'scale' of the weighting function, for functions with 'hard' boundaries (all bar gaussian) weights will be non-zero between +/- 2*bandwidth of the centre of the kernel
                    for a gaussian kernel the corresponds to sigma of the gaussian
        center - the center of the weighting function (i.e. the value of x for whihc the weights are maximum)
        weighting_function - available functions are: 'gaussian' (default), 'triangular', 'uniform',  'epanechnikov' and 'tricube'
        
        norm - if True the weights are normalised to sum to one
        
    
    OUTPUTS:
        w - vector of weights corresponding to the values in x
    """    
    x = x.flatten()
    bandwidth = float(bandwidth) 
    if weighting_function=='triangular':
        
        
          w=1.- np.abs(x-center)/(bandwidth*2) #assumes symmetric kernel bounds
          w[w<0.] = 0. # values outside kernel bounds equals zero 
        
    elif weighting_function=='gaussian':
        # kernel "bounds refer to +/- 1.96SD in Gaussian"
        w = mynormpdf(x,mu=center,sigma=bandwidth)
       
    elif weighting_function=='uniform':
        mini, maxi = [center-2*bandwidth, center+2*bandwidth]
        mask = (x>=mini)*(x<=maxi)
        w = mask.astype(float)
        
    
    elif weighting_function == 'epanechnikov':
        u=(x-center)/(bandwidth*2)
        
        w =0.75*(1-u**2)
        w[np.abs(u)>1.]=0.
    
    elif weighting_function == 'tricube':
        u=np.abs(x-center)/(bandwidth*2)
        w = (70./81.)*(1-np.abs(u)**3)**3
        w[np.abs(u)>1.]=0.
    else:
        raise ValueError('Invalid weighting function = ' + weighting_function)
    if norm==True:
        w=w/np.sum(w)
    return w

