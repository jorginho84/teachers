# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:59:57 2021

@author: pjac2
"""


from __future__ import division
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
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")

#Betas and var-cov matrix

betas_nelder  = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/betasopt_model_v2.npy")

data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/data_pythonpast.dta')

data_1 = data_1[data_1['d_trat']==1]

means = np.zeros((5,2))


for i in range(1,5):
       
    data =data_1[data_1['Rsquare_5']==i]
    
    for x in range(0,2):
        

        # TREATMENT #
        treatment = np.ones(np.array(data['experience']).shape[0])*x
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
        
        alphas = [[betas_nelder[0], betas_nelder[1],betas_nelder[2],betas_nelder[3],
          betas_nelder[4], betas_nelder[5]],
         [betas_nelder[6], betas_nelder[7],betas_nelder[8],betas_nelder[9],
          betas_nelder[10], betas_nelder[11]]]

        betas = [betas_nelder[12], betas_nelder[13], betas_nelder[14] ,betas_nelder[15]]

        gammas = [betas_nelder[16],betas_nelder[17],betas_nelder[18]]
        
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
        
        if x == 1:
            data_t = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1], 'Treatment': opt['Treatment']}
            data_st = pd.DataFrame(data_t, columns=['SIMCE','PORTFOLIO','TEST', 'Treatment'])
            simce_t = data_st['SIMCE']
            simce_tt = np.mean(np.array(data_st['SIMCE']))
            means[i,x]=simce_tt
        else:
            data_wt = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1], 'Treatment': opt['Treatment']}
            data_swt = pd.DataFrame(data_wt, columns=['SIMCE','PORTFOLIO','TEST', 'Treatment'])
            simce_wt = data_swt['SIMCE']
            simce_wtt = np.mean(np.array(data_swt['SIMCE']))
            means[i,x]=simce_wtt
            
    
    
 
conterfactual_late = means[:,1] - means[:,0]













