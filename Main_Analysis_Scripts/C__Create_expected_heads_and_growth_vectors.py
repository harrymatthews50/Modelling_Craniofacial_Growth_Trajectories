# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:17:23 2017

@author: harry.matthews
"""
modulepath = 'C:\Users\harry.matthews\Documents\Projects\Modelling_3D_Craniofacial_Growth_Curves_Supp_Material\Modules' #TODO Rememeber to update location on your machine

import sys
sys.path.append(modulepath)
from ShapeStats import multivariate_statistics, helpers 

import numpy as np
import os
import pickle
import pandas as pa


gen_datapath = os.path.join(os.path.split(modulepath)[0],'Generated_Data')
part_datapath = os.path.join(os.path.split(modulepath)[0],'Participant_Data')
# ages at which to evaluate the kernel regression models
evaluation_ages = np.array([1.12, 1.5,2., 2.5, 3., 3.5, 4., 4.5, 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14., 14.5, 15., 15.5, 16.])

#Operation requires back-projection - see README
#==============================================================================
# 
# #load regression objects
# with open(os.path.join(gen_datapath, 'Regression_objects','Shapeboys_regression.p'),'rb') as p:
#         boys_shape_regression = pickle.load(p)
#     
# with open(os.path.join(gen_datapath,'Regression_objects', 'Shapegirls_regression.p'),'rb') as p:
#         girls_shape_regression = pickle.load(p)
# 
# 
# # get expected heads and growth vectors in PC space
# expected_boys_heads, boys_growth_vectors = boys_shape_regression.predict(evaluation_ages,parameters = ['regression_predicted_location','regression_coefs'])
# expected_girls_heads, girls_growth_vectors = girls_shape_regression.predict(evaluation_ages,parameters = ['regression_predicted_location','regression_coefs'] )
# 
# # back project to landmark space
# eigvecs = np.load(os.path.join(part_datapath,'eigenvectors.npy'))
# meanvec = np.load(os.path.join(part_datapath,'meanvec.npy'))
# 
# # load mesh connectivity which is needed to write meshes to file
# con = helpers.obj2array(os.path.join(os.path.split(modulepath)[0],'Generic_Template.obj'))[1]
# 
# expected_boys_heads = helpers.PC_space_to_landmark_space(expected_boys_heads,eigvecs,meanvec)
# expected_girls_heads = helpers.PC_space_to_landmark_space(expected_girls_heads,eigvecs,meanvec)
# 
# boys_growth_vectors = helpers.PC_space_to_landmark_space(boys_growth_vectors, eigvecs)
# girls_growth_vectors = helpers.PC_space_to_landmark_space(girls_growth_vectors, eigvecs)
# 
# #create morphs
# 
# # get differenec between expected boys and expected girls
# diff = expected_boys_heads - expected_girls_heads
# 
# # define exaggerated heads
# exaggerated_boys_heads = expected_boys_heads + diff*2
# exaggerated_girls_heads = expected_girls_heads - diff*2
# 
# for i, age in enumerate(evaluation_ages):
# 
#     helpers.writewobj(expected_boys_heads[:,:,i],con, os.path.join(gen_datapath,'Expected_Heads_and_morphs','ExpectedBoy_'+('%.2f'%age).replace('.','_')+'.obj'))
#     helpers.writewobj(exaggerated_boys_heads[:,:,i],con, os.path.join(gen_datapath,'Expected_Heads_and_morphs','ExaggeratedBoy_'+('%.2f'%age).replace('.','_')+'.obj'))
#     np.savetxt(os.path.join(gen_datapath,'Growth_vectors','GrowthVectorsBoy_'+('%.2f'%age).replace('.','_')+'.txt'),boys_growth_vectors[:,:,i], delimiter = ',')
#    
#     helpers.writewobj(expected_girls_heads[:,:,i],con, os.path.join(gen_datapath,'Expected_Heads_and_morphs','ExpectedGirl_'+('%.2f'%age).replace('.','_')+'.obj'))
#     helpers.writewobj(exaggerated_girls_heads[:,:,i],con, os.path.join(gen_datapath,'Expected_Heads_and_morphs','ExaggeratedGirl_'+('%.2f'%age).replace('.','_')+'.obj'))
#     np.savetxt(os.path.join(gen_datapath,'Growth_vectors','GrowthVectorsGirl_'+('%.2f'%age).replace('.','_')+'.txt'),girls_growth_vectors[:,:,i], delimiter = ',')
#     
#     
#==============================================================================


















