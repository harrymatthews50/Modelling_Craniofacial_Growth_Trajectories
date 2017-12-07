# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:30:12 2017

@author: harry.matthews
"""

modulepath = 'C:\Users\harry.matthews\Documents\Projects\Modelling_3D_Craniofacial_Growth_Curves_Supp_Material\Modules' #TODO Rememeber to update location on your machine

import sys
sys.path.append(modulepath)
from ShapeStats import multivariate_statistics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import pandas as pa
import os
import pickle


part_datapath = os.path.join(os.path.split(modulepath)[0],'Participant_Data')
gen_datapath = os.path.join(os.path.split(modulepath)[0],'Generated_Data')

# load regression object for settings - all models have the same settings
with open(os.path.join(gen_datapath, 'Regression_objects','Shapeboys_regression.p'),'rb') as p:
        boys_shape_regression = pickle.load(p)

settings = boys_shape_regression.kernel_settings

nreps = 100
nfolds = 10

#load participant data
metadata = pa.read_excel(os.path.join(part_datapath, 'Participant_Metadata.xlsx'))
shape = pa.read_excel(os.path.join(part_datapath, 'PC_projections.xlsx'))



sex = metadata.loc[:,'Sex'].as_matrix()
age = metadata.loc[:,'Age'].as_matrix()

# define strata for train test splits, assign each case an integer which is unique to their age and sex
strata = np.zeros(len(shape.index))
bin_limits = list(np.arange(0,18,1))+[20000000000000000]   
boys_age_codes = np.digitize(age[sex=='M'],bin_limits)
girls_age_codes = np.digitize(age[sex=='F'],bin_limits)

strata[sex=='M'] = boys_age_codes
strata[sex=='F'] = girls_age_codes+np.max(boys_age_codes)

      
      
################### calculate scores repeatedly in a K-fold cross validation      


results = []


np.random.seed(59) 
for r in range(nreps):
    
    # data frame containing columns for each fold and rows for each case this keeps the results of each fold separate, as later we calculate ROCs for each fold 
     scores = pa.DataFrame(index=metadata.index,columns=np.arange(nfolds))

     for f,(train_inds, test_inds) in enumerate(StratifiedKFold(n_splits=nfolds,shuffle=True).split(metadata.index,strata)):
                     
                     train_IDs = metadata.index[train_inds]
                     test_IDs = metadata.index[test_inds]
                     
                     # separate training_IDs into males and female
                     boys_train = train_IDs[metadata.loc[train_IDs,'Sex']=='M']
                     girls_train = train_IDs[metadata.loc[train_IDs,'Sex']=='F']
                     
                     
                     #fit regresion models based on the training data, data to use is determined by the 'indices' keyword argument
                     boys_regression_model = multivariate_statistics.Kernel_regression()
                     boys_regression_model.fit(metadata.loc[:,'Age'], shape,'regression_predicted_location',indices=boys_train,**settings)
                     
                     girls_regression_model = multivariate_statistics.Kernel_regression()
                     girls_regression_model.fit(metadata.loc[:,'Age'], shape,'regression_predicted_location',indices=girls_train,**settings)
                     
                     #get test cases shape and age info
                     
                     test_age = metadata.loc[test_IDs,'Age'].as_matrix()
                     test_shape = shape.loc[test_IDs,:].as_matrix()
                     
                     scores.loc[test_IDs,f] = multivariate_statistics.compute_phenotype_score(test_shape, test_age, boys_regression_model,girls_regression_model)
                     
                     
     results.append(scores)
#%% Calculate score for each individual as their mean score over all repetitions

#### collapse across folds to get participants score from each repetions

scores_by_rep = np.zeros([len(metadata.index),nreps])
for rep, scores in enumerate(results):
    # get indices to scores as a flattened array
    
    dim1, dim2 = np.nonzero(pa.isnull(scores).as_matrix()==False)
    inds = np.ravel_multi_index((dim1,dim2),scores.shape,order='C')     
    scores_by_rep[dim1,rep] = np.ravel(scores.as_matrix(),order='C')[inds]
    
participant_scores = pa.DataFrame(index=metadata.index,columns = ['Score'])
participant_scores.loc[:,'Score'] = np.mean(scores_by_rep,axis=1)
participant_scores.to_excel(os.path.join(gen_datapath,'Classification','Scores.xlsx'))

    
#%% Calculate percentage correct and AUC, binning data from each age bin and compile it all into a table

bin_limits = [0.,5.,10.,15.,20.]
bin_titles = ['<5','5-10','10-15','>15']

table = pa.DataFrame(index = bin_titles, columns = ['AUC','Boys % Correct','Girls % Correct'])
 
# identify wchich bin each case belongs to
bin_inds = np.digitize(metadata.loc[:,'Age'],bin_limits)-1 # minus one to use as zero based python inex

for row in np.unique(bin_inds):
    count = 0
    
    
    #IDs of cases in the age bin 
    IDs_in_bin = metadata.index[bin_inds==row]
    AUCs = np.zeros(nreps*nfolds)
    pct_correct_boys = np.zeros_like(AUCs)
    pct_correct_girls = np.zeros_like(AUCs)                      

    for rep in range(nreps):
        for fold in range(nfolds):
            IDs_in_fold = metadata.index[np.nonzero(pa.isnull(results[rep].loc[:,fold]).as_matrix()==False)[0]]
            
            #IDs in both the age bin and the current fold
            IDs_to_use = [item for item in IDs_in_bin if item in IDs_in_fold]
            
            # get true values of sex and alspo their predicted scores
            true = metadata.loc[IDs_to_use,'Sex_numeric'].as_matrix()
            pred = results[rep].loc[IDs_to_use,fold].as_matrix()
            
            #calulate area uner roc curve
            fp,tp,thresholds = roc_curve(true,pred,pos_label=1)
            AUCs[count] = auc(fp,tp)
            
            # calulate percentage correct boys and girls, using threshold of zero
            confmat = confusion_matrix(true==1,pred>0.)  
            pct_correct_boys[count] = confmat[1,1]/float(np.sum(true==1))*100
            pct_correct_girls[count] = confmat[0,0]/float(np.sum(true==2))*100

            count+=1
    
    #write to table
    for col, statistic in enumerate([AUCs,pct_correct_boys,pct_correct_girls]):
        mean = np.mean(statistic)
        lower, upper = np.percentile(statistic,[2.5,97.5])
        table.iloc[row,col] = '%.2f'%mean + ' ('+'%.2f'%lower+','+'%.2f'%upper+')'
    table.to_excel(os.path.join(gen_datapath,'Classification','Results_Table.xlsx'))             
                  
                     

