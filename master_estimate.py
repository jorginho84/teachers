# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:28:07 2020

@author: pjac2
"""

#from __future__ import division #omit for python 3.x
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
#sys.path.append("C:\\Users\\Jorge\\Dropbox\\Chicago\\Research\\Human capital and the household\]codes\\model")
sys.path.append("D:\Git\ExpSIMCE")
#import gridemax
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
#import pybobyqa
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook

np.random.seed(123)

betas_nelder = np.load("D:\Git\ExpSIMCE/betasopt_model_RA2.npy")

moments_vector = np.load("D:\Git\ExpSIMCE/moments.npy")

data = pd.read_stata('D:\Git\ExpSIMCE/data_pythonpast.dta')



#count_nan = data['zpjeport'].isnull().sum()
#print('Count of nan: ' +str(count_nan))
#count_nan_1 = data['zpjeprue'].isnull().sum()
#print('Count of nan: ' +str(count_nan_1))

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

# TYPE SCHOOL #
typeSchool = np.array(data['typeschool'])

#### PARAMETERS MODEL ####
N = np.size(p1_0)
HOURS = np.array([44]*N)

alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
      betas_nelder[3], betas_nelder[4]],
     [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
      betas_nelder[8], betas_nelder[9]]]

betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12] ,betas_nelder[13],betas_nelder[14]]

gammas = [betas_nelder[15],betas_nelder[16],betas_nelder[17]]

# basic rent by hour in dollar (average mayo 2020, until 13/05/2020) *
# value hour (pesos)= 14403 *
# value hour (pesos)= 15155 *

dolar= 600

value = [14403, 15155]

hw = [value[0]/dolar,value[1]/dolar]

porc = [0.0338, 0.0333]

# *** This is withouth teaching career ***
# * value professional qualification (pesos)= 72100 *
# * value professional mention (pesos)= 24034 *
# *** This is with teaching career ***
# * value professional qualification (pesos)= 253076 *
# * value professional mention (pesos)= 84360 *

#inflation adjustment: 2012Jan-2019Dec: 1.266
qualiPesos = [72100*1.266, 24034*1.266, 253076, 84360] 
 

pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]

#inflation adjustemtn: 2012Jan-2020Jan: 1.111
Asig = [150000*1.111,100000*1.111,50000*1.111]
AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar]

#* Progression component by tranche *
#* value progression initial (pesos)= 14515
#* value progression early (pesos)= 47831
#* value progression advanced (pesos)= 96266
#* value progression advanced (pesos)= 99914 Fixed component
#* value progression expert 1 (pesos)= 360892
#* value progression expert 1 (pesos)= 138769 Fixed component
#* value progression expert 2 (pesos)= 776654
#* value progression expert 2 (pesos)= 210929 Fixed component

progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]

pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,  
           progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]


param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP)


ses_opt = np.load("D:\Git\ExpSIMCE/ses_model.npy")
w_matrix = np.zeros((ses_opt.shape[0],ses_opt.shape[0]))

for j in range(ses_opt.shape[0]):
    w_matrix[j,j] = ses_opt[j]**(-2)



output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, w_matrix,moments_vector)


start_time = time.time()

#here we go
output_me = output_ins.optimizer()

time_opt=time.time() - start_time
print ('Done in')
print("--- %s seconds ---" % (time_opt))


#the list of estimated parameters
beta_1 = output_me.x[0]
beta_2 = output_me.x[1]
beta_3 = output_me.x[2]
beta_4 = output_me.x[3]
beta_5 = output_me.x[4]
beta_6 = output_me.x[5]
beta_7 = output_me.x[6]
beta_8 = output_me.x[7]
beta_9 = output_me.x[8]
beta_10 = output_me.x[9]
beta_11 = output_me.x[10]
beta_12 = output_me.x[11]
beta_13 = output_me.x[12]
beta_14 = output_me.x[13]
beta_15 = output_me.x[14]
beta_16 = output_me.x[15]
beta_17 = output_me.x[16]
beta_18 = output_me.x[17]




betas_opt_me = np.array([beta_1, beta_2,
	beta_3,
	beta_4,beta_5,beta_6,beta_7,beta_8,
	beta_9,beta_10,beta_11,beta_12,
	beta_13,beta_14,beta_15,
	beta_16,beta_17,beta_18])
                        


np.save('D:\Git\ExpSIMCE/betasopt_model_RA3.npy',betas_opt_me)

