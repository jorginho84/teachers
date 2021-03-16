# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:59:57 2021

@author: pjac2
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
#sys.path.append("C:\\Users\\Jorge\\Dropbox\\Chicago\\Research\\Human capital and the household\]codes\\model")
sys.path.append("D:\Git\TeachersMaster")
#import gridemax
import time
#import int_linear
import between
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
#import pybobyqa
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
from scipy import interpolate
import time
import openpyxl
sys.path.append("D:\Git\TeachersMaster")

#Betas and var-cov matrix

betas_nelder  = np.load("D:\\Git\\TeachersMaster\\betasopt_model.npy")

data = pd.read_stata('D:\Git\TeachersMaster\data_python.dta')

data =data[data['d_trat']==1]

#With this we can give tratment=0 to the vector
#data['d_trat'] = data['d_trat'].replace(1,0)

# TREATMENT #
#treatmentOne = data[['d_trat']]
treatment = data['d_trat'].to_numpy()

# EXPERIENCE #
#yearsOne = data[['experience']]
years = data['experience'].to_numpy()

# SCORE PORTFOLIO #
#p1_0_1 = data[['score_port']]
p1_0 = data['score_port'].to_numpy()
p1 = data['score_port'].to_numpy()

#p1_0_1 = data[['ptj_portafolio_a2016']]
#p1_0 = data['ptj_portafolio_a2016'].to_numpy()
#p1 = data['ptj_portafolio_a2016'].to_numpy()

# SCORE TEST #
#p2_0_1 = data[['score_test']]
p2_0 = data['score_test'].to_numpy()
p2 = data['score_test'].to_numpy()

#p2_0_1 = data[['ptj_prueba_a2016']]
#p2_0 = data['ptj_prueba_a2016'].to_numpy()
#p2 = data['ptj_prueba_a2016'].to_numpy()

# CATEGORY PORTFOLIO #
#categPortfolio = data[['cat_port']]
catPort = data['cat_port'].to_numpy()

#categPortfolio = data[['cat_portafolio_a2016']]
#catPort = data['cat_portafolio_a2016'].to_numpy()

# CATEGORY TEST #
#categPrueba = data[['cat_test']]
catPrueba = data['cat_test'].to_numpy()

#categPrueba = data[['cat_prueba_a2016']]
#catPrueba = data['cat_prueba_a2016'].to_numpy()


# TRAME #
#Recover initial placement from data (2016) 
#TrameInitial = data[['trame']]
TrameI = data['trame'].to_numpy()

#TrameInitial = data[['tramo_a2016']]
#TrameI = data['tramo_a2016'].to_numpy()

# TYPE SCHOOL #
#typeSchoolOne = data[['typeschool']]
typeSchool = data['typeschool'].to_numpy()

#### PARAMETERS MODEL ####

N = np.size(p1_0)

HOURS = np.array([44]*N)


#gamma_0 = betas_nelder[14]
#gamma_1 = betas_nelder[15]
#gamma_2 = betas_nelder[16]
#betas_opt_t = np.array([betas_nelder[11],betas_nelder[12],
#	betas_nelder[13]]).reshape((3,1))
#alphas_port = np.array([betas_nelder[0],betas_nelder[1],betas_nelder[2],
#               betas_nelder[3],betas_nelder[4]]).reshape((5,1))
#alphas_test = np.array([betas_nelder[5],betas_nelder[6],betas_nelder[7],
#               betas_nelder[8],betas_nelder[9]]).reshape((5,1))


alphas = [[betas_nelder[0],betas_nelder[1],betas_nelder[2],betas_nelder[3],
           betas_nelder[4]],
		[betas_nelder[5],betas_nelder[6],betas_nelder[7],betas_nelder[8],
           betas_nelder[9]]]

#betas = [100,0.9,0.9,-0.05,-0.05,20]
#Parámetros más importantes
#betas = [100,10,33,20]

betas = [betas_nelder[10],betas_nelder[11],betas_nelder[12],betas_nelder[13]]

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

qualiPesos = [72100, 24034, 253076, 84360] 

pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]

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


param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol)

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI)

initial_p = model.initial()
print(initial_p)


between.betweenOne()

print("Random Effort")

misj = len(initial_p)
effort = np.random.randint(2, size=misj)
print(effort)

between.betweenOne()

print("Effort Teachers")

tscores = model.t_test(effort)
print(tscores)

between.betweenOne()

print("Placement")

placement = model.placement(tscores)
print(placement)

between.betweenOne()

print("Income")
    
income = model.income(placement)
print(income)

between.betweenOne()

print("Distance")

nextT, distancetrame = model.distance(placement)
print(nextT)
print(distancetrame)


between.betweenOne()

print("Effort Student")

h_student = model.student_h(effort)
print(h_student)

between.betweenOne()

print("Direct utility")

utilityTeacher = model.utility(income, effort, h_student)
print(utilityTeacher)

# SIMDATA

between.betweenOne()

print("Utility simdata")

modelSD = sd.SimData(N,model,treatment)

utilitySD = modelSD.util(effort)
print(utilitySD)


between.betweenOne()

# SIMULACIÓN SIMDATA


print("SIMDATA")

opt = modelSD.choice(treatment)
print(opt)

prueba_simce = opt['Opt Simce']
print(prueba_simce)


if treatment[0] == 1:
    data_t = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1], 'Treatment': opt['Treatment']}
    data_st = pd.DataFrame(data_t, columns=['SIMCE','PORTFOLIO','TEST', 'Treatment'])
    simce_t = data_st['SIMCE']
    simce_tt = np.mean(data_st['SIMCE'].to_numpy())
    print(simce_tt)
else:
    data_wt = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1], 'Treatment': opt['Treatment']}
    data_swt = pd.DataFrame(data_wt, columns=['SIMCE','PORTFOLIO','TEST', 'Treatment'])
    simce_wt = data_swt['SIMCE']
    simce_wtt = np.mean(data_swt['SIMCE'].to_numpy()) 
    print(simce_wtt)
    
    
 
conterfactual_late = np.array([simce_tt, simce_wtt])

    
print(conterfactual_late)
#np.save('D:\Git\TeachersMaster\conterfactual_late.npy',conterfactual_late)











