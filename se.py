# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:44:58 2021

@author: pjac2
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
import seaborn as sn
from statsmodels.iolib.summary2 import summary_col
from tabulate import tabulate
from texttable import Texttable
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
sys.path.append("D:\Git\TeachersMaster")


df = pd.read_stata('D:\Git\TeachersMaster\data_python.dta')

    
def simulation(times,data):
        "Function that simulate x times."
        
        n = data.shape[0]
        alpha_00 = np.zeros(times)
        alpha_01 = np.zeros(times)
        alpha_02 = np.zeros(times)
        alpha_03 = np.zeros(times)
        alpha_04 = np.zeros(times)
        alpha_10 = np.zeros(times)
        alpha_11 = np.zeros(times)
        alpha_12 =  np.zeros(times)
        alpha_13 =  np.zeros(times)
        alpha_14 =  np.zeros(times)
        beta_0 =  np.zeros(times)
        beta_1 = np.zeros(times)
        beta_2 = np.zeros(times)
        beta_3 = np.zeros(times)
        gamma_0 = np.zeros(times)
        gamma_1 = np.zeros(times)
        gamma_2 = np.zeros(times)
        
        
        for i in range(1,times):
            rev = data.sample(n, replace=True)            
            #the list of estimated parameters
            moments_vector = pd.read_excel("D:\Git\TeachersMaster\Outcomes.xlsx", header=3, usecols='C:F').values
            # TREATMENT #
            #treatmentOne = rev[['d_trat']]
            treatment = rev['d_trat'].to_numpy()
            # EXPERIENCE #
            #yearsOne = rev[['experience']]
            years = rev['experience'].to_numpy()
            # SCORE PORTFOLIO #
            #p1_0_1 = rev[['score_port']]
            p1_0 = rev['score_port'].to_numpy()
            p1 = rev['score_port'].to_numpy()
            # SCORE TEST #
            #p2_0_1 = rev[['score_test']]
            p2_0 = rev['score_test'].to_numpy()
            p2 = rev['score_test'].to_numpy()
            # CATEGORY PORTFOLIO #
            #categPortfolio = rev[['cat_port']]
            catPort = rev['cat_port'].to_numpy()
            # CATEGORY TEST #
            #categPrueba = rev[['cat_test']]
            catPrueba = rev['cat_test'].to_numpy()
            # TRAME #
            #TrameInitial = rev[['trame']]
            TrameI = data['trame'].to_numpy()
            # TYPE SCHOOL #
            #typeSchoolOne = rev[['typeschool']]
            typeSchool = rev['typeschool'].to_numpy()
            #### PARAMETERS MODEL ####
            N = np.size(p1_0)
            HOURS = np.array([44]*N)
            alphas = [[0.5,0.1,0.2,-0.01,0.1],
                      [0.5,0.1,0.2,-0.01,0.1]]
            betas = [-0.4,0.3,0.9,1]
            gammas = [-0.1,-0.2,0.8]
            dolar= 600
            value = [14403, 15155]
            hw = [value[0]/dolar,value[1]/dolar]
            porc = [0.0338, 0.0333]
            qualiPesos = [72100, 24034, 253076, 84360]
            pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]
            progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]
            pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,
                   progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]
            param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol)
            w_matrix = np.identity(17)

            
            output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, w_matrix,moments_vector)
            
            start_time = time.time()
            output = output_ins.optimizer()
            time_opt=time.time() - start_time
            print ('Done in')
            print("--- %s seconds ---" % (time_opt))
            
            alpha_00[i] = output.x[0]
            alpha_01[i] = output.x[1]
            alpha_02[i] = output.x[2]
            alpha_03[i] = output.x[3]
            alpha_04[i] = output.x[4]
            alpha_10[i] = output.x[5]
            alpha_11[i] = np.exp(output.x[6])
            alpha_12[i] = output.x[7]
            alpha_13[i] = output.x[8]
            alpha_14[i] = output.x[9]
            beta_0[i] = np.exp(output.x[10])
            beta_1[i] = output.x[11]
            beta_2[i] = output.x[12]
            beta_3[i] = output.x[13]
            gamma_0[i] = output.x[14]
            gamma_1[i] = output.x[15]
            gamma_2[i] = output.x[16]
        
        est_alpha_00 = np.std(alpha_00)
        est_alpha_01 = np.std(alpha_01)
        est_alpha_02 = np.std(alpha_02)
        est_alpha_03 = np.std(alpha_03)
        est_alpha_04 = np.std(alpha_04)
        est_alpha_10 = np.std(alpha_10)
        est_alpha_11 = np.std(alpha_11)
        est_alpha_12 = np.std(alpha_12)
        est_alpha_13 = np.std(alpha_13)
        est_alpha_14 = np.std(alpha_14)
        est_beta_0 = np.std(beta_0)
        est_beta_1 = np.std(beta_1)
        est_beta_2 = np.std(beta_2)
        est_beta_3 = np.std(beta_3)
        est_gamma_0 = np.std(gamma_0)
        est_gamma_1 = np.std(gamma_1)
        est_gamma_2 = np.std(gamma_2)
        
   
        return {'SE alpha_00': est_alpha_00,
                'SE alpha_01': est_alpha_01,
                'SE alpha_02': est_alpha_02,
                'SE alpha_03': est_alpha_03,
                'SE alpha_04': est_alpha_04,
                'SE alpha_10': est_alpha_10,
                'SE alpha_11': est_alpha_11,
                'SE alpha_12': est_alpha_12,
                'SE alpha_13': est_alpha_13,
                'SE alpha_14': est_alpha_14,
                'SE beta_0': est_beta_0,
                'SE beta_1': est_beta_1,
                'SE beta_2': est_beta_2,
                'SE beta_3': est_beta_3,
                'SE gamma_0': est_gamma_0,
                'SE gamma_1': est_gamma_1,
                'SE gamma_2': est_gamma_2}
    





result = simulation(2,df)
print(result)


betas_opt = np.array([result['SE alpha_00'], result['SE alpha_01'], 
                              result['SE alpha_02'],result['SE alpha_03'],result['SE alpha_04'],
                              result['SE alpha_10'],result['SE alpha_11'],result['SE alpha_12'], 
                                  result['SE alpha_13'],result['SE alpha_14'],result['SE beta_0'],
                                  result['SE beta_1'],result['SE beta_2'], 
                                      result['SE beta_3'],result['SE gamma_0'],result['SE gamma_1'], result['SE gamma_2']])



np.save('D:\Git\TeachersMaster\se_model.npy',betas_opt)
