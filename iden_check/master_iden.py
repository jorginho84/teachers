#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:25:47 2021

@author: jorge-home


This code compares simulated with data moments across sets of parameters

"""


import numpy as np
import pandas as pd
import pickle
import itertools
import sys, os
from scipy import stats
#from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from joblib import Parallel, delayed
from scipy import interpolate
import matplotlib.pyplot as plt
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
#sys.path.append("D:\Git\WageError")
#import gridemax
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
#import pybobyqa
#import xlsxwriter
from openpyxl import load_workbook


np.random.seed(123)


exec(open('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/iden_check/load_param.py').read())

#data = pd.read_stata('D:\Git\ExpSIMCE/data_pythonpast.dta')
data = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/DATA/data_pythonpast_v2023.dta')


# TREATMENT #
treatment = np.array(data['d_trat'])

# EXPERIENCE #
years = np.array(data['experience'])

# SCORE PORTFOLIO #
p1_0 = np.array(data['score_port_past'])
p1 = np.array(data['score_port'])

# SCORE TEST #
p2_0 = np.array(data['score_port_past'])
p2 = np.array(data['score_test'])

# CATEGORY PORTFOLIO #
catPort = np.array(data['cat_port'])

# CATEGORY TEST #
catPrueba = np.array(data['cat_test'])

# TRAME #
#Recover initial placement from data (2016) 
TrameI = np.array(data['trame'])

#TrameInitial = data[['tramo_a2016']]
#TrameI = data['tramo_a2016'].to_numpy()

# TYPE SCHOOL #
typeSchool = np.array(data['typeschool'])

# Priority #
priotity = np.array(data['por_priority'])

AEP_priority = np.array(data['priority_aep'])

rural_rbd = np.array(data['rural_rbd'])

locality = np.array(data['AsignacionZona'])


N = np.size(p1_0)

HOURS = np.array([44]*N)

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,priotity,rural_rbd,locality, AEP_priority)

modelSD = sd.SimData(N,model)


#ses_opt = np.load("D:\Git\ExpSIMCE/ses_model.npy")
ses_opt = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/ses_model_v2023.npy")
w_matrix = np.zeros((ses_opt.shape[0],ses_opt.shape[0]))

for j in range(ses_opt.shape[0]):
    w_matrix[j,j] = ses_opt[j]**(-2)

output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, priotity,rural_rbd,locality, AEP_priority, w_matrix,moments_vector)
    
    
#----------------------------------------------------------------------------#
    
    #Generating figures#
    
#----------------------------------------------------------------------------#    
font_size = 12

######Utility Function#####

    
#gamma0:utility cost of effort type 1
exec(open('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/iden_check/gamma0.py').read())


#gamma1:utility cost of effort type 2
#exec(open('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/iden_check/gamma1.py').read())