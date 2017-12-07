# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:35:14 2017

@author: harry.matthews
"""
modulepath = 'C:\Users\harry.matthews\Documents\Projects\Modelling_3D_Craniofacial_Growth_Curves_Supp_Material\Modules' #TODO Rememeber to update location on your machine

import sys
sys.path.append(modulepath)
from ShapeStats import multivariate_statistics, helpers 
import pickle
import os
import numpy as np
import pandas as pa
# define directory for generated data relative to modulepath
part_datapath = os.path.join(os.path.split(modulepath)[0],'Participant_Data')
gen_datapath = os.path.join(os.path.split(modulepath)[0],'Generated_Data')

# ages at which to evaluate the kernel regression models
evaluation_ages = np.array([1.12, 1.5,2., 2.5, 3., 3.5, 4., 4.5, 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14., 14.5, 15., 15.5, 16.])

numits = 10000 # number of permutations/ resamplings to run to generate confidence intervals and p-values
#%% Resample within each population to get confidence intervals of expected size

# load regression objects
with open(os.path.join(gen_datapath, 'Regression_objects','Sizeboys_regression.p'),'rb') as p:
        boys_size_regression = pickle.load(p)
    
with open(os.path.join(gen_datapath,'Regression_objects', 'Sizegirls_regression.p'),'rb') as p:
        girls_size_regression = pickle.load(p)

boys_expected_size = boys_size_regression.predict(evaluation_ages)
girls_expected_size = girls_size_regression.predict(evaluation_ages)


# get expected size for each resampling
np.random.seed(87)
resample_boys_size = boys_size_regression.get_resampled_parameters(evaluation_ages, parameters = 'regression_predicted_location',num_resamples=numits) 
resample_girls_size = girls_size_regression.get_resampled_parameters(evaluation_ages, parameters = 'regression_predicted_location',num_resamples=numits) 



for label,centre,resampled in zip(('Boys','Girls'),(boys_expected_size, girls_expected_size),(resample_boys_size,resample_girls_size)): 
    #confidence intervals from resampling
    lower,upper= np.percentile(resampled,[2.5,97.5],axis=2)
    
    #write to dataframe
    
    df =pa.DataFrame(index= ['Centre','Upper','Lower'], columns = evaluation_ages)
    df.loc['Centre',:] = centre.flatten()
    df.loc['Upper',:] = upper.flatten()
    df.loc['Lower',:] = lower.flatten()
    df.to_excel(os.path.join(gen_datapath,'Overall_size_and_shape_differences_results',label+'Expected_sizeCI.xlsx'))
#%% Resample within each population and calculate growth rate n times to generate confidence intervals

#load regression objects
with open(os.path.join(gen_datapath, 'Regression_objects','Shapeboys_regression.p'),'rb') as p:
        boys_shape_regression = pickle.load(p)
    
with open(os.path.join(gen_datapath,'Regression_objects', 'Shapegirls_regression.p'),'rb') as p:
        girls_shape_regression = pickle.load(p)

# get regression coefs through PC space at each age
girlcoefs = girls_shape_regression.predict(evaluation_ages,'regression_coefs')
boycoefs = boys_shape_regression.predict(evaluation_ages,'regression_coefs')


#Operation requires back-projection - see README
#==============================================================================
# 
# # get vectors through PC space for each resampling
# 
# np.random.seed(53)
# resample_girlcoefs = girls_shape_regression.get_resampled_parameters(evaluation_ages,parameters='regression_coefs',num_resamples = numits)
# resample_boycoefs = boys_shape_regression.get_resampled_parameters(evaluation_ages,parameters='regression_coefs',num_resamples = numits)
# 
# 
# # needged for backprojection
# eigvecs = np.load(os.path.join(part_datapath,'eigenvectors.npy'))
# 
# 
# for label, coefs, resample_coefs in zip(('Boys','Girls'), (boycoefs,girlcoefs),(resample_boycoefs,resample_girlcoefs)):
#     # back project regression vectors for all ages
#     growthvecs = helpers.PC_space_to_landmark_space(coefs, eigvecs)
#     
#     # calculate_mean_lengths
#     lengths = np.linalg.norm(growthvecs,axis=0)
#     mean_lengths = np.mean(lengths,axis=0)
# 
#     ########### calculate confidence intervals
#     # calculate mean lengths of growth vectors for each age for each resampling
#     
#     resample_mean_lengths = np.zeros([len(evaluation_ages), numits])
#     for n in range(numits): #TODO vectorise
#                 
#         resample_growthvecs = helpers.PC_space_to_landmark_space(resample_coefs[:,:,n],eigvecs)
#             
#         resample_lengths = np.linalg.norm(resample_growthvecs,axis=0)
#         resample_mean_lengths[:,n] = np.mean(resample_lengths,axis=0)
#     
#     
#    # confidence intervals
#     lower,upper= np.percentile(resample_mean_lengths,[2.5,97.5],axis=1)
# 
#     # put in dataframe and write to file
#     
#     df =pa.DataFrame(index= ['Centre','Upper','Lower'], columns = evaluation_ages)
#     df.loc['Centre',:] = mean_lengths
#     df.loc['Upper',:] = upper
#     df.loc['Lower',:] = lower
#     df.to_excel(os.path.join(gen_datapath,'Overall_size_and_shape_differences_results',label+'Growth_rateCI.xlsx'))
#     
#==============================================================================
#%% Resample within each population to get expected shapes and calculate Procrustes distance between them to get confidence intervals of Procrustes distance

#load regression objects
with open(os.path.join(gen_datapath, 'Regression_objects','Shapeboys_regression.p'),'rb') as p:
        boys_shape_regression = pickle.load(p)
    
with open(os.path.join(gen_datapath,'Regression_objects', 'Shapegirls_regression.p'),'rb') as p:
        girls_shape_regression = pickle.load(p)

# get expected heads in PC space
exp_girl = girls_shape_regression.predict(evaluation_ages,'regression_predicted_location')
exp_boy = boys_shape_regression.predict(evaluation_ages,'regression_predicted_location')

# get expected heads in PC space for each resampling

np.random.seed(25)
resample_exp_girl = girls_shape_regression.get_resampled_parameters(evaluation_ages,parameters='regression_predicted_location',num_resamples = numits)
resample_exp_boy = boys_shape_regression.get_resampled_parameters(evaluation_ages,parameters='regression_predicted_location',num_resamples = numits)

# calculate expected Procrustes distance (this is the same in PC space or in the space of point-coordinates)

proc_dist = np.linalg.norm(exp_girl-exp_boy,axis=1)

# resampled_procrustes distance
resample_proc_dist = np.linalg.norm(resample_exp_girl-resample_exp_boy,axis=1) 

#confidence intervals from resampling
lower,upper= np.percentile(resample_proc_dist,[2.5,97.5],axis=1)
    
#write to file

df =pa.DataFrame(index= ['Centre','Upper','Lower'], columns = evaluation_ages)
df.loc['Centre',:] = proc_dist
df.loc['Upper',:] = upper
df.loc['Lower',:] = lower
df.to_excel(os.path.join(gen_datapath,'Overall_size_and_shape_differences_results','Procrustes_DistanceCI.xlsx'))

#%% Permute group labels and refit regression models with same settings to generate null distributions of Procrustes distance for p-value calculation

#load regression objects  
with open(os.path.join(gen_datapath, 'Regression_objects','Shapeboys_regression.p'),'rb') as p:
        boys_shape_regression = pickle.load(p)
    
with open(os.path.join(gen_datapath,'Regression_objects', 'Shapegirls_regression.p'),'rb') as p:
        girls_shape_regression = pickle.load(p)


#get settings - settings for both regressions are the same
settings = boys_shape_regression.kernel_settings

# load participant data
metadata = pa.read_excel(os.path.join(part_datapath, 'Participant_Metadata.xlsx'))
shape = pa.read_excel(os.path.join(part_datapath, 'PC_projections.xlsx'))

#get sex labels
sex = metadata.loc[:,'Sex'].as_matrix()

#initialise array
null_proc_dists = np.zeros([len(evaluation_ages),numits])

np.random.seed(79)
for n in range(numits):
    #permute labels
    perm_labels = sex[np.random.permutation(len(sex))]
    perm_boys = metadata.index[perm_labels=='M']
    perm_girls = metadata.index[perm_labels=='F']
    
    #fit regressions
    perm_boys_regression = multivariate_statistics.Kernel_regression()
    perm_boys_regression.fit(metadata.loc[:,'Age'],shape,'regression_predicted_location',indices=perm_boys,**settings)
    
    perm_girls_regression = multivariate_statistics.Kernel_regression()
    perm_girls_regression.fit(metadata.loc[:,'Age'],shape,'regression_predicted_location',indices=perm_girls,**settings)
    
    # get expectation
    perm_exp_boys = perm_boys_regression.predict(evaluation_ages,parameters='regression_predicted_location')
    perm_exp_girls = perm_girls_regression.predict(evaluation_ages, parameters= 'regression_predicted_location')
    
    #add Procrustes distance to null values
    null_proc_dists[:,n] = np.linalg.norm((perm_exp_girls - perm_exp_boys),axis=1)


# get actual procrustes distances
exp_girl = girls_shape_regression.predict(evaluation_ages,'regression_predicted_location')
exp_boy = boys_shape_regression.predict(evaluation_ages,'regression_predicted_location')

proc_dist = np.linalg.norm(exp_girl-exp_boy,axis=1)

# calculate one-tailed p-values as the proportion of null procrustes distances that are larger than the actual procrustes distance 
pvals = np.sum(np.greater(null_proc_dists,np.atleast_2d(proc_dist).T),axis=1)/float(numits)



# convert to string and format p-value of zero to <1/numits
pstring = []
for p in pvals:
    if p==0.:
        pstring.append('<'+str(1/float(numits)))
    else:
        pstring.append(str(p))
        
    
# write to file 
df = pa.DataFrame(columns=['p'], index = evaluation_ages)
df.loc[:,'p'] = pstring      
df.to_excel(os.path.join(gen_datapath,'Overall_size_and_shape_differences_results','Procrustes_Distance_pvals.xlsx'))


    







