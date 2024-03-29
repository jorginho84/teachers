# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:28:07 2020

@author: pjac2

exec(open("/home/jrodriguezo/teachers/codes/master_estimate.py").read())

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
#sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
#sys.path.append("D:\Git\ExpSIMCE")
#sys.path.append("C:\\Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13")
sys.path.append("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy")
import time
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est


np.random.seed(123)

#betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/betasopt_model_v29.npy")
#betas_nelder  = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/betasopt_model_v24.npy")
betas_nelder  = np.load(r"C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/estimates/betasopt_model_v53.npy")


#moments_vector = np.load("D:\Git\ExpSIMCE/moments.npy")
#moments_vector = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/moments_v2023.npy")
#moments_vector = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/moments_v2023.npy")
moments_vector = np.load(r"C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/estimates/moments_new.npy")

#data = pd.read_stata('D:\Git\ExpSIMCE/data_pythonpast.dta')
data = pd.read_stata(r'C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/data_pythonpast_v2023.dta')
#data = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/data_pythonpast_v2023.dta')

# Reading the data in pkl extension.
#data= pd.read_pickle("data_pythonv.pkl")



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
TrameI = np.array(data['tramo_a2016'])

# TYPE SCHOOL #
typeSchool = np.array(data['typeschool'])

# Priority #
priotity = np.array(data['por_priority'])

rural_rbd = np.array(data['rural_rbd'])

locality = np.array(data['AsignacionZona'])

AEP_priority = np.array(data['priority_aep'])

#### PARAMETERS MODEL ####
N = np.size(p1_0)
HOURS = np.array([44]*N)

alphas = [[0, betas_nelder[0],0,betas_nelder[1],
             betas_nelder[2], betas_nelder[3]],
            [0, 0,betas_nelder[4],betas_nelder[5],
            betas_nelder[6], betas_nelder[7]]]
            
betas = [-0.85, betas_nelder[9], betas_nelder[10],betas_nelder[11],betas_nelder[12],betas_nelder[13]]
gammas = [betas_nelder[14],betas_nelder[15],betas_nelder[16]]
    

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

#inflation adjustemtn: 2012Jan-2020Jan: 1.111***
qualiPesos = [72100*1.111, 24034*1.111, 253076, 84360] 
pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]

#inflation adjustemtn: 2012Jan-2020Jan: 1.111
Asig = [50000*1.111, 100000*1.111, 150000*1.111]
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

pri = [48542,66609,115151]
priori = [pri[0]/dolar, pri[1]/dolar, pri[2]/dolar]


param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)


#ses_opt = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/ses_model.npy")
#ses_opt = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/ses_model_v2023.npy")
ses_opt = np.load(r"C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/estimates/ses_model_new.npy")

var_cov = np.load(r"C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/estimates/var_cov_new.npy")

w_matrix = np.zeros((ses_opt.shape[0],ses_opt.shape[0]))
for j in range(ses_opt.shape[0]):
    w_matrix[j,j] = ses_opt[j]**(-2)


output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,priotity,rural_rbd,locality, AEP_priority, \
                 w_matrix,moments_vector)


start_time = time.time()

#here we go
output_me = output_ins.optimizer()

time_opt=time.time() - start_time
print ('Done in')
print("--- %s seconds ---" % (time_opt))


#the list of estimated parameters
beta_0 = output_me.x[0]
beta_1 = output_me.x[1]
beta_2 = np.exp(output_me.x[2])
beta_3 = output_me.x[3]
beta_4 = output_me.x[4]
beta_5 = output_me.x[5]
beta_6 = np.exp(output_me.x[6])
beta_7 = output_me.x[7]
beta_8 = output_me.x[8]
beta_9 = output_me.x[9]
beta_10 = output_me.x[10]
beta_11 = np.exp(output_me.x[11])
beta_12 = output_me.x[12]
beta_13 = output_me.x[13]
beta_14 = output_me.x[14]
beta_15 = output_me.x[15]
beta_16 = output_me.x[16]





betas_opt_me = np.array([beta_0,beta_1, beta_2,
	beta_3,
	beta_4,beta_5,beta_6,beta_7,beta_8,
	beta_9,beta_10,beta_11,beta_12,
	beta_13,beta_14,beta_15,
	beta_16])
                        


np.save(r'C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/estimates/betasopt_model_v56.npy',betas_opt_me)



"""
##this is to check if value function coincides. It does.##

qw = output_ins.objfunction(output_me.x)


betas_nelder_2  = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v39.npy")

alphas = [[betas_nelder_2[0], betas_nelder_2[1],0,betas_nelder_2[2],
             betas_nelder_2[3], betas_nelder_2[4]],
            [betas_nelder_2[5], 0,betas_nelder_2[6],betas_nelder_2[7],
            betas_nelder_2[8], betas_nelder_2[9]]]


            
betas = [betas_nelder_2[10], betas_nelder_2[11], betas_nelder_2[12],betas_nelder_2[13],betas_nelder_2[14],betas_nelder_2[15]]
gammas = [betas_nelder_2[16],betas_nelder_2[17],betas_nelder_2[18]]
    
alphas_control = [[betas_nelder_2[19],betas_nelder_2[20]],[betas_nelder_2[21],betas_nelder_2[22]]]
betas_control = [betas_nelder_2[23],betas_nelder_2[24]]


param0 = parameters.Parameters(alphas,betas,gammas,alphas_control,betas_control,hw,porc,pro,pol,AEP,priori)

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,
                     TrameI,priotity,rural_rbd,locality, AEP_priority)

modelSD = sd.SimData(N,model)

   

output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,priotity,rural_rbd,locality, AEP_priority, \
                 w_matrix,moments_vector)
    
       
beta0 = np.array([param0.alphas[0][0],
                          param0.alphas[0][1],
                          param0.alphas[0][3],  
                          np.log(param0.alphas[0][4]),
                          param0.alphas[0][5],
                          param0.alphas[1][0],
                          param0.alphas[1][2],
                          param0.alphas[1][3],
                          np.log(param0.alphas[1][4]),
                          param0.alphas[1][5],
                          param0.betas[0],
                          param0.betas[1],
                          param0.betas[2],
                          param0.betas[3],
                          param0.betas[4],
                          param0.betas[5],
                          param0.gammas[0],
                          param0.gammas[1],
                          param0.gammas[2],
                          param0.alphas_control[0][0],
                          np.log(param0.alphas_control[0][1]),
                          param0.alphas_control[1][0],
                          np.log(param0.alphas_control[1][1]),
                          param0.betas_control[0],
                          param0.betas_control[1]])



q_w2 = output_ins.objfunction(beta0)





"""