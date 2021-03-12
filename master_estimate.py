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
sys.path.append("D:\Git\TeacherPrincipal")
sys.path.append("D:\Git\result")
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




moments_vector = pd.read_excel("D:\Git\TeacherPrincipal\Outcomes.xlsx", header=3, usecols='C:F').values
#moments_vector_excel = pd.read_excel("D:\Git\TeacherPrincipal\Outcomes.xlsx", header=3, usecols='D')
#moments_vector_zero = moments_vector_excel[['simulation']]
#moments_vector = moments_vector_zero['simulation'].to_numpy()

#ajhdsajk = moments_vector[0,1]

data = pd.read_stata('D:\Git\TeacherPrincipal\data_python.dta')



#count_nan = data['zpjeport'].isnull().sum()
#print('Count of nan: ' +str(count_nan))
#count_nan_1 = data['zpjeprue'].isnull().sum()
#print('Count of nan: ' +str(count_nan_1))

# TREATMENT #
treatmentOne = data[['d_trat']]
treatment = data['d_trat'].to_numpy()

# EXPERIENCE #
yearsOne = data[['experience']]
years = data['experience'].to_numpy()

# SCORE PORTFOLIO #
p1_0_1 = data[['score_port']]
p1_0 = data['score_port'].to_numpy()
p1 = data['score_port'].to_numpy()

#p1_0_1 = data[['ptj_portafolio_a2016']]
#p1_0 = data['ptj_portafolio_a2016'].to_numpy()
#p1 = data['ptj_portafolio_a2016'].to_numpy()

# SCORE TEST #
p2_0_1 = data[['score_test']]
p2_0 = data['score_test'].to_numpy()
p2 = data['score_test'].to_numpy()

#p2_0_1 = data[['ptj_prueba_a2016']]
#p2_0 = data['ptj_prueba_a2016'].to_numpy()
#p2 = data['ptj_prueba_a2016'].to_numpy()

# CATEGORY PORTFOLIO #
categPortfolio = data[['cat_port']]
catPort = data['cat_port'].to_numpy()

#categPortfolio = data[['cat_portafolio_a2016']]
#catPort = data['cat_portafolio_a2016'].to_numpy()

# CATEGORY TEST #
categPrueba = data[['cat_test']]
catPrueba = data['cat_test'].to_numpy()

#categPrueba = data[['cat_prueba_a2016']]
#catPrueba = data['cat_prueba_a2016'].to_numpy()


# TRAME #
#Recover initial placement from data (2016) 
TrameInitial = data[['trame']]
TrameI = data['trame'].to_numpy()

#TrameInitial = data[['tramo_a2016']]
#TrameI = data['tramo_a2016'].to_numpy()

# TYPE SCHOOL #
typeSchoolOne = data[['typeschool']]
typeSchool = data['typeschool'].to_numpy()

#### PARAMETERS MODEL ####

N = np.size(p1_0)

HOURS = np.array([44]*N)

alphas = [[0.5,0.1,0.2,-0.01,0.1],
		[0.5,0.1,0.2,-0.01,0.1]]

#betas = [100,0.9,0.9,-0.05,-0.05,20]
#Parámetros más importantes
#betas = [100,10,33,20]

betas = [-0.4,0.3,0.9,1]

gammas = [-0.1,-0.2,0.8]

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

w_matrix = np.identity(15)

output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, w_matrix,moments_vector)


start_time = time.time()

#here we go
output = output_ins.optimizer()

time_opt=time.time() - start_time
print ('Done in')
print("--- %s seconds ---" % (time_opt))


#the list of estimated parameters
beta_1 = output.x[0]
beta_2 = output.x[1]
beta_3 = output.x[2]
beta_4 = output.x[3]
beta_5 = output.x[4]
beta_6 = output.x[5]
beta_7 = np.exp(output.x[6])
beta_8 = output.x[7]
beta_9 = output.x[8]
beta_10 = output.x[9]
beta_11 = np.exp(output.x[10])
beta_12 = output.x[11]
beta_13 = output.x[12]
beta_14 = output.x[13]
beta_15 = output.x[14]
beta_16 = output.x[15]
beta_17 = output.x[16]


betas_opt = np.array([beta_1, beta_2,
	beta_3,
	beta_4,beta_5,beta_6,beta_7,beta_8,
	beta_9,beta_10,beta_11,beta_12,
	beta_13,beta_14,beta_15,
	beta_16,beta_17])

print(betas_opt)


np.save('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/model/betas_v1.npy',betas_opt)

wb = load_workbook('D:\Git\TeacherPrincipal\Outcomes.xlsx')

sheet = wb.active

sheet['G4'] = 'Betas Opt'
sheet['G5'] = beta_1
sheet['G6'] = beta_2
sheet['G7'] = beta_3
sheet['G8'] = beta_4
sheet['G9'] = beta_5
sheet['G10'] = beta_6
sheet['G11'] = beta_7
sheet['G12'] = beta_8
sheet['G13'] = beta_9
sheet['G14'] = beta_10
sheet['G15'] = beta_11
sheet['G16'] = beta_12
sheet['G17'] = beta_13
sheet['G18'] = beta_14
sheet['G19'] = beta_15
sheet['G20'] = beta_16
sheet['G21'] = beta_17


wb.save('D:\Git\TeacherPrincipal\Outcomes.xlsx')


#np.save('D:\Git\TeacherPrincipal\beta_opt.npy',betas_opt)