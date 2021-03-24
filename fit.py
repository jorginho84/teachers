#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:33:39 2021

@author: jorge-home

This code computes fit anaylisis

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
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
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

betas_nelder = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/betasopt_model_v2.npy")

moments_vector = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/moments.npy")

#ajhdsajk = moments_vector[0,1]

data = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/data_pythonpast.dta')



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

alphas = [[betas_nelder[0], betas_nelder[1],betas_nelder[2],betas_nelder[3],
          betas_nelder[4], betas_nelder[5]],
         [betas_nelder[6], betas_nelder[7],betas_nelder[8],betas_nelder[9],
         betas_nelder[10], betas_nelder[11]]]

betas = [betas_nelder[12], betas_nelder[13], betas_nelder[14] ,betas_nelder[15]]

gammas = [betas_nelder[16],betas_nelder[17],betas_nelder[18]]

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

modelSD = sd.SimData(N,model,treatment)

w_matrix = w_matrix = np.zeros((19,19))
ses_opt = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/ses_model.npy")

for j in range(19):
    w_matrix[j,j] = ses_opt[j]**(-2)

output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, w_matrix,moments_vector)


corr_data = output_ins.simulation(50,modelSD)
print(corr_data)


beta0 = np.array([param0.alphas[0][0],
                          param0.alphas[0][1],
                          param0.alphas[0][2],
                          param0.alphas[0][3],
                          param0.alphas[0][4],
                          param0.alphas[0][5],
                          param0.alphas[1][0],
                          param0.alphas[1][1],
                          param0.alphas[1][2],
                          param0.alphas[1][3],
                          param0.alphas[1][4],
                          param0.alphas[1][5],
                          param0.betas[0],
                          param0.betas[1],
                          param0.betas[2],
                          param0.betas[3],
                          param0.gammas[0],
                          param0.gammas[1],
                          param0.gammas[2]])

qw = output_ins.objfunction(beta0)

##### PYTHON TO EXCEL #####

#workbook = xlsxwriter.Workbook('D:\Git\TeacherPrincipal\Outcomes.xlsx')
#worksheet = workbook.add_worksheet()

#book = Workbook()
#sheet = book.active

wb = load_workbook('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/Outcomes.xlsx')
sheet = wb.active

sheet['C5'] = 'Mean Portfolio'
sheet['C6'] = 'Variance Portfolio'
sheet['C7'] = 'Mean SIMCE'
sheet['C8'] = 'Variance SIMCE'
sheet['C9'] = 'Mean Test'
sheet['C10'] = 'Variance Test'
sheet['C11'] = 'Mean Portfolio-Test'
sheet['C12'] = '\% Initial'
sheet['C13'] = '\% Intermediate'
sheet['C14'] = '\% Advanced'
sheet['C15'] = '\% Expert'
sheet['C16'] = 'corr(Port,Simce)'
sheet['C17'] = 'corr(Test,Simce)'
sheet['C18'] = 'corr(exp,Port)'
sheet['C19'] = 'corr(exp,Test)'
sheet['C20'] = '\% Intermediate control'
sheet['C21'] = '\% adva/expert control'
sheet['C23'] = 'Corr(Port,p)'
sheet['C22'] = 'Corr(Test,p)'
sheet['D4'] = 'simulation'
sheet['E4'] = 'data'
sheet['F4'] = 'se'

sheet['D5'] = corr_data['Mean Portfolio']
sheet['D6'] = corr_data['Var Port']
sheet['D7'] = corr_data['Mean SIMCE']
sheet['D8'] = corr_data['Var SIMCE']
sheet['D9'] = corr_data['Mean Test']
sheet['D10'] = corr_data['Var Test']
sheet['D11'] = corr_data['Mean PortTest']
sheet['D12'] = corr_data['perc init']
sheet['D13'] = corr_data['perc inter']
sheet['D14'] = corr_data['perc advanced']
sheet['D15'] = corr_data['perc expert']
sheet['D16'] = corr_data['Estimation SIMCE vs Portfolio']
sheet['D17'] = corr_data['Estimation SIMCE vs Prueba']
sheet['D18'] = corr_data['Estimation EXP vs Portfolio']
sheet['D19'] = corr_data['Estimation EXP vs Prueba']
sheet['D20'] = corr_data['perc inter control']
sheet['D21'] = corr_data['perc adv/exp control']
sheet['D22'] = corr_data['Estimation Test vs p']
sheet['D23'] = corr_data['Estimation Portfolio vs p']



sim = np.array([corr_data['Mean Portfolio'],
corr_data['Var Port'],
corr_data['Mean SIMCE'],
corr_data['Var SIMCE'],
corr_data['Mean Test'],
corr_data['Var Test'],
corr_data['Mean PortTest'],
corr_data['perc init'],
corr_data['perc inter'],
corr_data['perc advanced'],
corr_data['perc expert'],
corr_data['Estimation SIMCE vs Portfolio'],
corr_data['Estimation SIMCE vs Prueba'],
corr_data['Estimation EXP vs Portfolio'],
corr_data['Estimation EXP vs Prueba'],
corr_data['perc inter control'],
corr_data['perc adv/exp control'],
corr_data['Estimation Test vs p'],
corr_data['Estimation Portfolio vs p']])

x_vector = sim - moments_vector

q_w = np.dot(np.dot(np.transpose(x_vector),w_matrix),x_vector)

weight = x_vector**2/ses_opt**2


wb.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/Outcomes.xlsx')


