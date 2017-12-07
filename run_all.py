# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:45:38 2017

@author: harry.matthews
"""
import os

scriptdir = 'C:\Users\harry.matthews\Documents\Projects\Modelling_3D_Craniofacial_Growth_Curves_Supp_Material\Main_Analysis_Scripts'
scripts = os.listdir(scriptdir)

#sort alphabetically
scripts.sort()

for script in scripts:
    execfile(os.path.join(scriptdir,script))

