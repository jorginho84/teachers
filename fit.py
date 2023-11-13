#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:33:39 2021

@author: jorge-home

This code computes fit analysis

exec(open("/home/jrodriguezo/teachers/codes/fit.py").read())

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
#sys.path.append("C:\\Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13")
#sys.path.append("D:\Git\WageError")
sys.path.append("/home/jrodriguezo/teachers/codes")
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

#betas_nelder  = np.load("D:\Git\ExpSIMCE/betasopt_model_RA3.npy")
#betas_nelder  = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/betasopt_model_v25.npy")
#betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/betasopt_model_v40.npy")
betas_nelder = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v44.npy")

#moments_vector = np.load("D:\Git\ExpSIMCE/moments.npy")
#moments_vector = np.load("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/moments_v2023.npy")
#moments_vector = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/moments_new.npy")
moments_vector = np.load("/home/jrodriguezo/teachers/codes/moments_new.npy")

#data = pd.read_stata('D:\Git\ExpSIMCE/data_pythonpast.dta')
#data = pd.read_stata('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/data_pythonpast_v2023.dta')
#data= pd.read_pickle("data_pythonv.pkl")
#data = pd.read_stata('/Users/jorge-home/Library/CloudStorage/Dropbox/Research/teachers-reform/teachers/DATA/data_pythonpast_v2023.dta')
data = pd.read_stata('/home/jrodriguezo/teachers/data/data_pythonpast_v2023.dta')


#ses_opt = np.load("C:\\Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/ses_model_v2023.npy")
#ses_opt = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/ses_model_new.npy")
ses_opt = np.load("/home/jrodriguezo/teachers/codes/ses_model_new.npy")

# TREATMENT #
treatment = np.array(data['d_trat'])

# EXPERIENCE #
years = np.array(data['experience'])

# SCORE PORTFOLIO #
p1_0 = np.array(data['score_port_past'])
p1 = np.array(data['score_port'])

# SCORE TEST #
p2_0 = np.array(data['score_test_past'])
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

AEP_priority = np.array(data['priority_aep'])

rural_rbd = np.array(data['rural_rbd'])

locality = np.array(data['AsignacionZona'])

#### PARAMETERS MODEL ####

N = np.size(p1_0)

HOURS = np.array([44]*N)

alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
             betas_nelder[3], betas_nelder[4]],
            [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
            betas_nelder[8], betas_nelder[9]]]
            
betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12],betas_nelder[13],betas_nelder[14],betas_nelder[15]]
gammas = [betas_nelder[16],betas_nelder[17],betas_nelder[18]]
    
alphas_control = [[betas_nelder[19],betas_nelder[20]],[betas_nelder[21],betas_nelder[22]]]
betas_control = [betas_nelder[23],betas_nelder[24]]


# basic rent by hour in dollar (average mayo 2020, until 13/05/2020) *
# value hour (pesos)= 14403 *
# value hour (pesos)= 15155 *

dolar= 600

value = [14403, 15155]

hw = [value[0]/dolar,value[1]/dolar]

porc = [0.0338, 0.0333]


#inflation adjustemtn: 2012Jan-2020Jan: 1.111
Asig = [50000*1.111, 100000*1.111, 150000*1.111]
AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar]

# *** This is withouth teaching career ***
# * value professional qualification (pesos)= 72100 *
# * value professional mention (pesos)= 24034 *
# *** This is with teaching career ***
# * value professional qualification (pesos)= 253076 *
# * value professional mention (pesos)= 84360 *

#inflation adjustemtn: 2012Jan-2020Jan: 1.111***
qualiPesos = [72100*1.111, 24034*1.111, 253076, 84360] 

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

pri = [48542,66609,115151]
priori = [pri[0]/dolar, pri[1]/dolar, pri[2]/dolar]
    

param0 = parameters.Parameters(alphas,betas,gammas,alphas_control,betas_control,hw,porc,pro,pol,AEP,priori)

model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,
                     TrameI,priotity,rural_rbd,locality, AEP_priority)

modelSD = sd.SimData(N,model)


"""
opt = modelSD.choice()
simce = opt['Opt Simce']
np.var(simce[treatment == 1])
np.mean(simce[treatment == 1]) - np.mean(simce[treatment == 0])

"""

#ses_opt = np.load("D:\Git\ExpSIMCE/ses_model.npy")
w_matrix = np.zeros((ses_opt.shape[0],ses_opt.shape[0]))


for j in range(ses_opt.shape[0]):
    w_matrix[j,j] = ses_opt[j]**(-2)
    

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

qw = output_ins.objfunction(beta0)

corr_data = output_ins.simulation(50,modelSD)
sim = np.array([corr_data['Corr Simce and experience'],
            corr_data['Corr Portfolio and experience'],
            corr_data['Corr STEI and experience'],
            corr_data['Corr SIMCE and Portfolio'],
            corr_data['Corr SIMCE and STEI'],
            corr_data['SIMCE Mean (treated)'],
            corr_data['SIMCE Var (treated)'],
            corr_data['Portfolio Mean (treated)'],
            corr_data['STEI Mean (treated)'],
            corr_data['Portfolio Var (treated)'],
            corr_data['STEI Var (treated)'],
            corr_data['Corr Simce Past'],
            corr_data['Corr Portfolio Past'],
            corr_data['Corr STEI Past'],
            corr_data['Share Portfolio > 2.5 (treated)'],
            corr_data['Share STEI > 2.74 (treated)'],
            corr_data['Share teachers advancing from initial'],
            corr_data['SIMCE Mean (control)'],
            corr_data['SIMCE Var (control)'],
            corr_data['Portfolio Mean (control)'],
            corr_data['STEI Mean (control)'],
            corr_data['Portfolio Var (control)'],
            corr_data['STEI Var (control)'],
            corr_data['Share Portfolio > 2.5 (control)'],
            corr_data['Share STEI > 2.74 (control)']])

x_vector = moments_vector - sim 

q_w = np.dot(np.dot(np.transpose(x_vector),w_matrix),x_vector)


q_w2 = np.sum((x_vector**2/ses_opt**2))


#with open('/Users/jorge-home/Library/CloudStorage/Dropbox/Research/teachers-reform/teachers/Results/fit_table.tex','w') as f:
with open('/home/jrodriguezo/teachers/results/fit_table.tex','w') as f:
    f.write(r'\footnotesize{'+'\n')
    f.write(r'\begin{tabular}{llccccc}'+'\n')
    f.write(r'\toprule'+'\n')
    f.write(r'& Moment &  & Model &  & Data  &  & S.E. data \\'+'\n')
    f.write(r'\midrule'+'\n')
    f.write(r'A. Treatment group  (2016 teachers) &  &       &  &       &  & \\'+'\n')
    f.write(r'Mean SIMCE                          &  & '+'{:1.2f}'.format(sim[5]) +r' &  & '+'{:1.2f}'.format(moments_vector[5]) +r'   &  & '+'{:1.3f}'.format(ses_opt[5]) +r' \\'+'\n')
    f.write(r'Variance SIMCE                          &  & '+'{:1.2f}'.format(sim[6]) +r' &  & '+'{:1.2f}'.format(moments_vector[6]) +r'   &  & '+'{:1.3f}'.format(ses_opt[6]) +r' \\'+'\n')
    f.write(r'Mean Portfolio                          &  & '+'{:1.2f}'.format(sim[7]) +r' &  & '+'{:1.2f}'.format(moments_vector[7]) +r'   &  & '+'{:1.3f}'.format(ses_opt[7]) +r' \\'+'\n')
    f.write(r'Mean STEI                          &  & '+'{:1.2f}'.format(sim[8]) +r' &  & '+'{:1.2f}'.format(moments_vector[8]) +r'   &  & '+'{:1.3f}'.format(ses_opt[8]) +r' \\'+'\n')
    f.write(r'Variance Portfolio                          &  & '+'{:1.2f}'.format(sim[9]) +r' &  & '+'{:1.2f}'.format(moments_vector[9]) +r'   &  & '+'{:1.3f}'.format(ses_opt[9]) +r' \\'+'\n')
    f.write(r'Variance STEI                          &  & '+'{:1.2f}'.format(sim[10]) +r' &  & '+'{:1.2f}'.format(moments_vector[10]) +r'   &  & '+'{:1.3f}'.format(ses_opt[10]) +r' \\'+'\n')
    f.write(r'Corr(SIMCE, Past)                          &  & '+'{:1.2f}'.format(sim[11]) +r' &  & '+'{:1.2f}'.format(moments_vector[11]) +r'   &  & '+'{:1.3f}'.format(ses_opt[11]) +r' \\'+'\n')
    f.write(r'Corr(Portfolio, Past)                          &  & '+'{:1.2f}'.format(sim[12]) +r' &  & '+'{:1.2f}'.format(moments_vector[12]) +r'   &  & '+'{:1.3f}'.format(ses_opt[12]) +r' \\'+'\n')
    f.write(r'Corr(STEI, Past)                          &  & '+'{:1.2f}'.format(sim[13]) +r' &  & '+'{:1.2f}'.format(moments_vector[13]) +r'   &  & '+'{:1.3f}'.format(ses_opt[13]) +r' \\'+'\n')
    f.write(r'Share of teachers portfolio >= 2.5        &  & '+'{:1.2f}'.format(sim[14]) +r' &  & '+'{:1.2f}'.format(moments_vector[14]) +r'   &  & '+'{:1.3f}'.format(ses_opt[14]) +r' \\'+'\n')
    f.write(r'Share of teachers STEI >= 2.74        &  & '+'{:1.2f}'.format(sim[15]) +r' &  & '+'{:1.2f}'.format(moments_vector[15]) +r'   &  & '+'{:1.3f}'.format(ses_opt[15]) +r' \\'+'\n')
    f.write(r'Share of teachers advancing from initial        &  & '+'{:1.2f}'.format(sim[16]) +r' &  & '+'{:1.2f}'.format(moments_vector[16]) +r'   &  & '+'{:1.3f}'.format(ses_opt[16]) +r' \\'+'\n')

    f.write(r'                                    &  &       &  &       &  &       \\'+'\n')
    f.write(r'B. Control group (2018- teachers)   &  &       &  &       &  &       \\'+'\n')
    f.write(r'Mean SIMCE                          &  & '+'{:1.2f}'.format(sim[17]) +r' &  & '+'{:1.2f}'.format(moments_vector[17]) +r'   &  & '+'{:1.3f}'.format(ses_opt[17]) +r' \\'+'\n')
    f.write(r'Variance SIMCE                          &  & '+'{:1.2f}'.format(sim[18]) +r' &  & '+'{:1.2f}'.format(moments_vector[18]) +r'   &  & '+'{:1.3f}'.format(ses_opt[18]) +r' \\'+'\n')
    f.write(r'Mean Portfolio                          &  & '+'{:1.2f}'.format(sim[19]) +r' &  & '+'{:1.2f}'.format(moments_vector[19]) +r'   &  & '+'{:1.3f}'.format(ses_opt[19]) +r' \\'+'\n')
    f.write(r'Mean STEI                          &  & '+'{:1.2f}'.format(sim[20]) +r' &  & '+'{:1.2f}'.format(moments_vector[20]) +r'   &  & '+'{:1.3f}'.format(ses_opt[20]) +r' \\'+'\n')
    f.write(r'Variance Portfolio                          &  & '+'{:1.2f}'.format(sim[21]) +r' &  & '+'{:1.2f}'.format(moments_vector[21]) +r'   &  & '+'{:1.3f}'.format(ses_opt[21]) +r' \\'+'\n')
    f.write(r'Variance STEI                          &  & '+'{:1.2f}'.format(sim[22]) +r' &  & '+'{:1.2f}'.format(moments_vector[22]) +r'   &  & '+'{:1.3f}'.format(ses_opt[22]) +r' \\'+'\n')
    f.write(r'Share of teachers portfolio >= 2.5        &  & '+'{:1.2f}'.format(sim[23]) +r' &  & '+'{:1.2f}'.format(moments_vector[23]) +r'   &  & '+'{:1.3f}'.format(ses_opt[23]) +r' \\'+'\n')
    f.write(r'Share of teachers STEI >= 2.74        &  & '+'{:1.2f}'.format(sim[24]) +r' &  & '+'{:1.2f}'.format(moments_vector[24]) +r'   &  & '+'{:1.3f}'.format(ses_opt[24]) +r' \\'+'\n')

    f.write(r'                                    &  &       &  &       &  &       \\'+'\n')
    

    f.write(r'C. Full sample                      &  &       &  &       &  &       \\'+'\n')
    f.write(r'Corr(Exp, SIMCE)        &  & '+'{:1.2f}'.format(sim[0]) +r' &  & '+'{:1.2f}'.format(moments_vector[0]) +r'   &  & '+'{:1.3f}'.format(ses_opt[0]) +r' \\'+'\n')
    f.write(r'Corr(Exp, Port)        &  & '+'{:1.2f}'.format(sim[1]) +r' &  & '+'{:1.2f}'.format(moments_vector[1]) +r'   &  & '+'{:1.3f}'.format(ses_opt[1]) +r' \\'+'\n')
    f.write(r'Corr(Exp, STEI)        &  & '+'{:1.2f}'.format(sim[2]) +r' &  & '+'{:1.2f}'.format(moments_vector[2]) +r'   &  & '+'{:1.3f}'.format(ses_opt[2]) +r' \\'+'\n')
    f.write(r'Corr(Port, SIMCE)        &  & '+'{:1.2f}'.format(sim[3]) +r' &  & '+'{:1.2f}'.format(moments_vector[3]) +r'   &  & '+'{:1.3f}'.format(ses_opt[3]) +r' \\'+'\n')
    f.write(r'Corr(STEI, SIMCE)        &  & '+'{:1.2f}'.format(sim[4]) +r' &  & '+'{:1.2f}'.format(moments_vector[4]) +r'   &  & '+'{:1.3f}'.format(ses_opt[4]) +r' \\'+'\n')
    f.write(r'\bottomrule'+'\n')
    f.write(r'\end{tabular}'+'\n')
    f.write(r'}'+'\n')
    f.close()





"""
opt = modelSD.choice()
print('OPt effort % no effort, control', np.mean(opt['Opt Effort'][treatment == 0] == 0))
print('OPt effort % no effort, treatment', np.mean(opt['Opt Effort'][treatment == 1] == 0))

print('Mean income, control', np.mean(opt['Opt Income'][1][treatment == 0]))
print('Mean income, treatment', np.mean(opt['Opt Income'][0][treatment == 1]))
"""