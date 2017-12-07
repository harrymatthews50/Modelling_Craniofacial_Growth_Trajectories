# -*- coding: utf-8 -*-
"""

"""
#####Load ShapeStats Modules

modulepath = 'C:\Users\harry.matthews\Documents\Projects\Modelling_3D_Craniofacial_Growth_Curves_Supp_Material\Modules' #TODO Rememeber to update location on your machine

import sys
sys.path.append(modulepath)
from ShapeStats import multivariate_statistics, machine_learning 

import numpy as np
import pandas as pa
import os
import pickle
from collections import Counter

part_datapath = os.path.join(os.path.split(modulepath)[0],'Participant_Data')
gen_datapath = os.path.join(os.path.split(modulepath)[0],'Generated_Data')

# load participant data
metadata = pa.read_excel(os.path.join(part_datapath, 'Participant_Metadata.xlsx'))
size = pa.read_excel(os.path.join(part_datapath, 'Mean_Distance_To_Centroid.xlsx'))
shape = pa.read_excel(os.path.join(part_datapath, 'PC_projections.xlsx'))

# create X-block
X = metadata.loc[:,['Age']]

# get IDs for boys and girls
boys= [item for item in X.index if metadata.loc[item,'Sex']=='M']
girls = [item for item in X.index if metadata.loc[item,'Sex']=='F']


for label, Y in zip(('Size','Shape'), (size, shape )): # repeat for both size and shape


    # contstant model parameters
    model_args= [X,Y,'regression_predicted_location']
    model_kwargs = {}
    model_kwargs['weighting_function']='gaussian'
    model_kwargs['linear_regression_model'] = multivariate_statistics.PLSR
    model_kwargs['minimum_n'] = 3
    
    #parameter to tune
    tunable_parameter_name = 'bandwidth'
    
    #values of 'bandwidth' to try
    tunable_parameter_values = np.linspace(0.5,4.,15) # range needs to be chosen by a bit of trial and error - if you find optimal value is either the minimum or maximum of the range of values being tested, then extend the range
    
    
    
    # repeated grid-search settings
    
    V=10 # number of train-test splits for each parameter value in grid search
    n_reps = 50 # number of times to repeat the grid search

    # determine age bin each participant belongs to to stratify train-test splits for grid search 
    bin_limits = list(np.arange(0,18,1))+[20000000000000000]
    boys_strata = np.digitize(X.loc[boys,:].as_matrix().flatten(), bin_limits)
    girls_strata = np.digitize(X.loc[girls,:].as_matrix().flatten(), bin_limits)
    
    
    #check at least V cases in every bin 
    count_boys = Counter(boys_strata)
    count_girls = Counter(girls_strata)
    assert all([item>=V for item in count_boys.itervalues()])
    assert all([item>=V for item in count_girls.itervalues()])

    
    #seed random number generator
    np.random.seed(12)
    
    
    
    boys_grid_search = machine_learning.Grid_Search1D()
    boys_grid_search.fit(multivariate_statistics.Kernel_regression,
                                                      model_args,
                                                      model_kwargs,
                                                      tunable_parameter_name,
                                                      tunable_parameter_values,
                                                      indices=boys,
                                                      V=V,
                                                      n_reps=n_reps,
                                                      loss_func=machine_learning.sum_of_absolute_errors,
                                                      strata = boys_strata,
                                                      )
    
    
    
    girls_grid_search = machine_learning.Grid_Search1D()
    girls_grid_search.fit(multivariate_statistics.Kernel_regression,
                                                      model_args,
                                                      model_kwargs,
                                                      tunable_parameter_name,
                                                      tunable_parameter_values,
                                                      indices=girls,
                                                      V=V,
                                                      loss_func=machine_learning.sum_of_absolute_errors,
                                                      n_reps=n_reps,
                                                      strata = girls_strata,
                                                      )

    
    # get best bandwidth estimated for boys and for girls
    
    boys_best = boys_grid_search.optimal_value
    girls_best = girls_grid_search.optimal_value
    
    # pick the biggest (most conservative)
    bandwidth = np.max([girls_best, boys_best])
    
    
    
    # fit kernel regressions 
    boys_regression = multivariate_statistics.Kernel_regression()
    boys_regression.fit(*model_args,bandwidth=bandwidth,indices=boys,**model_kwargs)
    
    girls_regression = multivariate_statistics.Kernel_regression()
    girls_regression.fit(*model_args,bandwidth=bandwidth,indices=girls,**model_kwargs)


    #save to disk
    with open(os.path.join(gen_datapath,'Regression_objects', label+'boys_regression.p'),'wb') as p:
        pickle.dump( boys_regression,p, protocol=2)
    
    with open(os.path.join(gen_datapath,'Regression_objects', label+'girls_regression.p'),'wb') as p:
        pickle.dump(girls_regression,p, protocol=2)
    


    

