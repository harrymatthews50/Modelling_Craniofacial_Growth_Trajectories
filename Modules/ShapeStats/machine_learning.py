# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 17:26:46 2017

@author: harry.matthews
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
import scipy
import copy as co
from joblib import Parallel, delayed
import pandas as pa
import sys
import matplotlib.pyplot as plt
sys.path.append(__file__)
import helpers



#####Loss Functions
def sum_of_squared_errors(prval, trval, axis=None):
    diffs = prval-trval
    return np.sum(diffs**2, axis=axis)

def sum_of_absolute_errors(prval, trval, axis=None):
    diffs = prval-trval
    return np.sum(np.abs(diffs), axis=axis)




class Grid_Search1D(object):
    """
    Implements a repeated one-dimensional grid search, following Algorithm 1 in Krstajic et al.(2014)
    The nesting of loops is a bit different to save on time but is ultimately  the same
    
    REFERENCES:
        Krstajic, D., Buturovic, L. J., Leahy, D. E., & Thomas, S. (2014). Cross-validation pitfalls when selecting and assessing regression and classification models. Journal of Cheminformatics, 6(1), 10. 
    
    """
    def __init__(self):
        super(Grid_Search1D,self).__init__()
    
    def fit(self,model_definition, model_args,model_kwargs,tunable_parameter_name,tunable_parameter_values,indices='All',V=10,n_reps=10,loss_func=sum_of_squared_errors,use_parallel_computing=False,strata=None):
            """INPUTS:
                model_definition - a class definition that will be called internally to create each model
                model_args - the arguments to be passed to the models 'fit' function
                model_kwargs - a dictionary of keyword arguments to be passed to the models.fit function
                tunable_parameter_name - the name of the keyword argument whose variants are the elements on the grid
                tunable_parameter_values - a vector of values to iterate through, in order of increasing model complexity
                indices - a list of indices to the X and Y DataFrames specifying the cases to be used is the grid_search (default = 'All')
                V - the number of folds to split the data into
                n_reps - the number of times to repeat the grid search - the final result will be the mean of these searches
                strata - the strata to use to stratify resampling for grid search

                    
            """

            loss_test =  [grid_search(model_definition, model_args,model_kwargs,tunable_parameter_name,tunable_parameter_values,indices=indices,V=V,loss_func=loss_func,strata=strata) for rep in xrange(n_reps)]
            
            loss_test = [np.atleast_2d(item).T for item in loss_test]
            loss_test = np.concatenate(loss_test,axis=1)
            
            
            self.loss = pa.DataFrame(index=['Loss'],columns=tunable_parameter_values)
            self.loss.loc['Loss',:]= np.mean(loss_test,axis=1)

                        
            minimum_ind= np.argmin(np.ma.masked_invalid(self.loss.loc['Loss'].as_matrix().astype(float)))
            minimum_parameter_value = self.loss.columns[minimum_ind]

            
            self.optimal_value = minimum_parameter_value


def grid_search(model_definition, model_args,model_kwargs,tunable_parameter_name,tunable_parameter_values,indices='All',V=10,loss_func=sum_of_squared_errors,strata=None):
        
        """
        Implements one repetition of a  one-dimesnsional grid search, following Algorithm 1 in Krstajic et al.(2014)
        The nesting of loops is a bit different to save on time but is ultimate;y  the same
        
        REFERENCES:
            Krstajic, D., Buturovic, L. J., Leahy, D. E., & Thomas, S. (2014). Cross-validation pitfalls when selecting and assessing regression and classification models. Journal of Cheminformatics, 6(1), 10. 
        
        INPUTS:
                    model_definition - a class definition that will be called internally to create each model, msut be from multivariate_statistics package
                    model_args - the arguments to be passed to the models 'fit' function
                    model_kwargs - a dictionary of keyword arguments to be passed to the models.fit function
                    tunable_parameter_name - the name of the keyword argument whose variants are the elements on the grid
                    tunable_parameter_values - a vector of values to iterate through
                    indices - a list of indices to the X and Y DataFrames specifying the cases to be used is the grid_search (default = 'All')
                    V - the number of folds to split the data into
                    strata - group labels for stratified V-Folds, if None VFold is not stratified 
        
        OUTPUTS:
            
                
                
        """
        X,Y=model_args[:2]
        X, Y, Xmat, Ymat = helpers.configure_input_matrices(X,Y,indices)
        
        
            # repeat yvalues along 3rd axis for input into loss func later
        repeat_Ymat  = np.repeat(Ymat[:,:,np.newaxis],len(tunable_parameter_values),axis=2)
            
        model_kwargs = co.deepcopy(model_kwargs) # copy kwargs otherwise modifications made in function will result in modification outside of function
        
        IDs = X.index 
        
        fold_iterator = StratifiedKFold(n_splits=V,shuffle=True)
        if strata is None:
            strata = np.ones(len(IDs))
        predicted_Y_values_test = np.zeros([Ymat.shape[0], Y.values.shape[1],len(tunable_parameter_values)])
        predicted_Y_values_test[:] = np.nan

        
        
        # for each train/test fold
        for train, test in fold_iterator.split(np.arange(len(IDs)),strata):
            
            for parameter_number, parameter_value in enumerate(tunable_parameter_values):
                    # for each parameter value fit the model with the training set and get predicted values
                    
                    model_kwargs[tunable_parameter_name] = parameter_value
                    model_kwargs['indices'] = IDs[train]
                    
                    model_obj = model_definition()
                    try:
                        model_obj.fit(*model_args,**model_kwargs)
                        predicted_Y_values_test[test,:,parameter_number] = model_obj.predict(Xmat[test],original_space=True)
                    except Exception,e:
                        print('GridSearch model fitting for value '+ str(parameter_value)+' failed:\n'+str(e))
                                
        return loss_func(predicted_Y_values_test,repeat_Ymat,axis=(0,1))




class GD_Grid_Search1D(object):
    """Implements a grid search by gradient descent"""
    def __init__(self):
        super(GD_Grid_Search1D,self).__init__()
    
    def fit(self,model_definition, model_args, model_kwargs, tunable_parameter_name, startalpha,theta1,resolution,num_iterations=10.,indices='All',V=10,loss_func=sum_of_squared_errors ):
        
        
        X,Y=model_args[:2]
        X, Y, Xmat, Ymat = helpers.configure_input_matrices(X,Y,indices)
        
        
            # repeat yvalues along 3rd axis for input into loss func later
        repeat_Ymat  = np.repeat(Ymat[:,:,np.newaxis],3,axis=2)
            
        model_kwargs = co.deepcopy(model_kwargs) # copy kwargs otherwise it will be modified outside of function which may cause problems
        updates = []
        IDs = X.index 
        it = 0
        previous_loss = None
        alpha = startalpha
        while True:
        # evaluate gradient with respect to loss at theta1 by evaluating the loss at two adjacent values
            m_val = theta1-resolution
            p_val = theta1+resolution
            predicted_Y_values_test = np.zeros([Ymat.shape[0],Ymat.shape[1],3])
            for parameter_number, parameter_value in enumerate([m_val,theta1,p_val]):
                fold_iterator = KFold(n_splits=V,shuffle=True)
                model_kwargs[tunable_parameter_name] = parameter_value
                
                
                for train, test in fold_iterator.split(np.arange(len(IDs))):
                    
                    model_kwargs['indices'] = IDs[train]
                    
                    model_obj = model_definition()
                    model_obj.fit(*model_args,**model_kwargs)
                    predicted_Y_values_test[test,:,parameter_number] = model_obj.predict(Xmat[test],original_space=True)
             
            
            loss = loss_func(predicted_Y_values_test,repeat_Ymat,axis=(0,1))
            gradient = (loss[2]-loss[0])/(2*resolution)
            theta2 = theta1-alpha*gradient
            
            if previous_loss is not None: # adjust learning rate
                if previous_loss>loss[1]: # we are converging, try increasing learning rate
                    alpha = alpha*1.1
                    theta1 = theta2
                elif previous_loss<loss[1]:# we have missed minumum, reduce learning rate and leave theta as is
                    alpha = alpha*0.5
            
            previous_loss=loss[1]
                
            updates.append((theta1,alpha,loss[1]))
            it+=1
            if it==num_iterations:
                break
            
            self.updates=updates
            
            
        
        
        
        
        
        
        
        
        
#==============================================================================
#     
#     class tune_parameter_grid_search():
# 
#     def __init__(self,regression_obj,parameter_name,test_parameter_values,kernel_settings, num_its = 10, n_folds=10,random_state_seed=None,stratified=True,stratification_groups = 'default',loss_func=press, loss_func_kwargs={},estimates=['mean','regression_based_centre']):
#         
#     
#         
#         Xfull = regression_obj.var.as_matrix()
#         Yfull = regression_obj.PCA.Tcoeff.T
#         print(Yfull.shape)
#         N = Xfull.shape[0]
#         
#         
#         
#         
#         if all([stratification_groups=='default',stratified==True]):
#             # define stratification groups to bin data into age brackets
#             stratification_groups = nm.zeros(Xfull.shape,dtype=int)
#             bounds = nm.linspace(nm.min(Xfull),nm.max(Xfull)+0.000001,num=9)
#             for i, lowbound in enumerate(bounds[:-1]):
#                 highbound = bounds[i+1]
#                 mask = (Xfull>=lowbound)*(Xfull<highbound)
#                 stratification_groups[mask] = i
#         if random_state_seed is None:
#             random_state_seed = nm.random.randint(5000)
#             
#         
#         # keep loss estimates for each estimate in a dictionary
#         #initialise dictionary entries
#         test_error = {}
#         train_error = {}
#         for estimate in estimates:
#             test_error[estimate]=nm.zeros([len(test_parameter_values),num_its])
#             train_error[estimate] = nm.zeros(len(test_parameter_values))
#         
#     
#             
#             
#             
#         
#         kernel_settings = co.deepcopy(kernel_settings)
#         
#         
#         ###Estimate test error 
#         
#         
#         for it in range(num_its):   #"1. Repeat the following process Nexp times"
#         
#             if stratified is False:
#                 CViterator = KFold(N,n_folds, shuffle=True,random_state=random_state_seed+it) # random seed varies for each iteration, but will provide consistent results if random_state_seed is set
#             else:
#                 
#                 CViterator = StratifiedKFold(stratification_groups,n_folds,shuffle=True,random_state=random_state_seed+it)
#                 # a. divide the dataset D pseudo randomly into V folds
#             
# 
#             
#             predYvalues = {}
#             for estimate in estimates:
#                 predYvalues[estimate] = nm.zeros([N,len(test_parameter_values),Yfull.shape[1]])
#             for train_inds, test_inds in CViterator: #b. for I from 1 to V   i Define set L as the dataset D without the Ith fold, ii. Define set T as as the ith fold of the datase
#                 kernel_settings['mask'] = train_inds
#                 
#                 for alpha_i, alpha in enumerate(test_parameter_values):
#                     kernel_settings[parameter_name] = alpha
#                     regression_obj.fit(**kernel_settings)
#                     for i in test_inds:
#                         value = Xfull[i]
#                         for estimate in estimates:
#                             face, dump, dump = regression_obj.evaluate_at_value(value,return_value=estimate)
#                             predYvalues[estimate][i,alpha_i,:] = face
# 
#==============================================================================






    
   
##Demo 1
## Grid search for the optimal number of components in a PLSR
#import sys
#sys.path.append(__file__)
#from multivariate_statistics import PLSR, Kernel_regression
#import pandas as pa
#import time
#if __name__=='__main__':
#   
#    
#    df = pa.DataFrame.from_csv('C:\\Users\\harry.matthews\\Documents\\PythonShapeStatsv1\\test_data\\pandas_test_data.csv')
#    X= df 
#    Y = df
#    
#    tunable_parameter_name = 'ncomps'
#    tunable_parameter_values = np.array([1,2,3,4])
#    model_args = [X,Y]
#    model_kwargs = {}
#    model_definition = PLSR
#    t1 = time.time()
#    GridSearchobj = Grid_Search1D()
#    GridSearchobj.fit(PLSR,model_args,model_kwargs,tunable_parameter_name,tunable_parameter_values,V=10,n_reps=100,loss_func=press,use_parallel_computing=False)
#    t2 = time.time()
#    GridSearchobj2 = Grid_Search1D()
#    GridSearchobj2.fit(PLSR,model_args,model_kwargs,tunable_parameter_name,tunable_parameter_values,V=10,n_reps=100,loss_func=press,use_parallel_computing=True)
#    t3 = time.time()
#    
#    # find optimal number of components - those with minimum loss
#    optimal_n_comps = tunable_parameter_values[np.argmin(GridSearchobj.loss.loc['Loss',:].as_matrix().astype(float))]
#    
#    PLSRmod = PLSR()
#    PLSRmod.fit(X,Y,ncomps=optimal_n_comps)



# Demo 2
# Grid search optimal bandwidth for kernel regression















