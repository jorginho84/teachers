# -*- coding: utf-8 -*-
"""
Simulates data

"""

import numpy as np
import pandas as pd
import pickle
import tracemalloc
import itertools
import sys, os
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
sys.path.append("C:\\Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13")
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
import estimate_2 as est_2
#import estimate_3 as est_3
import between
import random
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
import time

# DATA 2018
betas_nelder  = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/betasopt_model_v23.npy")
#betas_nelder  = np.load("D:\Git\Wagenewcomponent/betasopt_model_v23.npy")
#betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/betasopt_model_v23.npy")

#moments_vector = np.load("D:\Git\ExpSIMCE/moments.npy")
#moments_vector = np.load("D:\Git\Wagenewcomponent/moments.npy")

#data = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\Local_teacherGIT/data_pythonpast_v2023.dta')
data= pd.read_pickle("data_pythonv.pkl")
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

#TrameInitial = data[['tramo_a2016']]
#TrameI = data['tramo_a2016'].to_numpy()

# TYPE SCHOOL #
typeSchool = np.array(data['typeschool'])

# Priority #
priotity = np.array(data['por_priority'])

rural_rbd = np.array(data['rural_rbd'])

locality = np.array(data['AsignacionZona'])

priority_aep = np.array(data['priority_aep'])


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


#inflation adjustemtn: 2012Jan-2020Jan: 1.111
Asig = [150000*1.111,100000*1.111,50000*1.111]
AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar]

# *** This is withouth teaching career ***
# * value professional qualification (pesos)= 72100 *
# * value professional mention (pesos)= 24034 *
# *** This is with teaching career ***
# * value professional qualification (pesos)= 253076 *
# * value professional mention (pesos)= 84360 *

#inflation adjustment: 2012Jan-2019Dec: 1.266
qualiPesos = [72100*1.266, 24034*1.266, 253076, 84360] 

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

pri = [47872,113561]
priori = [pri[0]/dolar, pri[1]/dolar]
    

param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,priotity,rural_rbd,locality,priority_aep)

print("Random Effort")

initial_p = model.initial()
print(initial_p)
misj = len(initial_p)
effort = np.random.randint(2, size=misj)
print(effort)

print("Effort Teachers")

tscores = model.t_test(effort)
print(tscores)

between.betweenOne()

print("Placement")

placement = model.placement(tscores[0], initial_p)
print(placement)

income = model.income(placement)
print(income)

#data.describe(include='all')  

modelSD = sd.SimData(N,model)

data_sim = modelSD.choice()

np.mean(data_sim['Opt Simce'])

#wages with minium score and low experience
tscores_min = [np.zeros(N),np.zeros(N)]
placement_min = model.placement(tscores_min)
income_min = model.income(placement_min)

income_min[0][years ==2 ]


