# -*- coding: utf-8 -*-
"""

exec(open("/home/jrodriguez/teachers/codes/se_parallel.py").read())

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
import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
import time
#import int_linear
sys.path.append("/home/jrodriguez/teachers/codes")
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
from pathos.multiprocessing import ProcessPool


betas_nelder = np.load("/home/jrodriguez/teachers/codes/betasopt_model_v3.npy")

df = pd.read_stata('/home/jrodriguez/teachers/data/data_pythonpast.dta')

moments_vector = np.load("/home/jrodriguez/teachers/codes/moments.npy")


w_matrix = w_matrix = np.zeros((19,19))
ses_opt = np.load('/home/jrodriguez/teachers/codes/ses_model.npy')

for j in range(19):
    w_matrix[j,j] = ses_opt[j]**(-2)



def simulation(j):
    """
    Obtains one set of estimates
    """
    n = df.shape[0]
    np.random.seed(j+100)
    rev = df.sample(n, replace=True)            

    # TREATMENT #
    #treatmentOne = rev[['d_trat']]
    treatment = np.array(rev['d_trat'])
    # EXPERIENCE #
    #yearsOne = rev[['experience']]
    years = np.array(rev['experience'])
    # SCORE PORTFOLIO #
    #p1_0_1 = rev[['score_port']]
    p1_0 = np.array(rev['score_port'])
    p1 = np.array(rev['score_port'])
    # SCORE TEST #
    #p2_0_1 = rev[['score_test']]
    p2_0 = np.array(rev['score_test'])
    p2 = np.array(rev['score_test'])
    # CATEGORY PORTFOLIO #
    #categPortfolio = rev[['cat_port']]
    catPort = np.array(rev['cat_port'])
    # CATEGORY TEST #
    #categPrueba = rev[['cat_test']]
    catPrueba = np.array(rev['cat_test'])
    # TRAME #
    #TrameInitial = rev[['trame']]
    TrameI = np.array(rev['trame'])
    # TYPE SCHOOL #
    #typeSchoolOne = rev[['typeschool']]
    typeSchool = np.array(rev['typeschool'])
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
    qualiPesos = [72100, 24034, 253076, 84360]
    pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]
    progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]
    pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,
           progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]

    Asig = [150000,100000,50000]
    AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar] 

    param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP)
        
    output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
         typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, w_matrix,moments_vector)
    
    start_time = time.time()
    output = output_ins.optimizer()
    time_opt=time.time() - start_time
    print ('Done in')
    print("--- %s seconds ---" % (time_opt))


    return output.x

boot_n = 400

alpha_00 = np.zeros(boot_n)
alpha_01 = np.zeros(boot_n)
alpha_02 = np.zeros(boot_n)
alpha_03 = np.zeros(boot_n)
alpha_04 = np.zeros(boot_n)
alpha_05 = np.zeros(boot_n)
alpha_10 = np.zeros(boot_n)
alpha_11 = np.zeros(boot_n)
alpha_12 =  np.zeros(boot_n)
alpha_13 =  np.zeros(boot_n)
alpha_14 =  np.zeros(boot_n)
alpha_15 =  np.zeros(boot_n)
beta_0 =  np.zeros(boot_n)
beta_1 = np.zeros(boot_n)
beta_2 = np.zeros(boot_n)
beta_3 = np.zeros(boot_n)
gamma_0 = np.zeros(boot_n)
gamma_1 = np.zeros(boot_n)
gamma_2 = np.zeros(boot_n)

start_time = time.time()

pool = ProcessPool(nodes = 18)
dics = pool.map(simulation,range(boot_n))
pool.close()
pool.join()
pool.clear()


time_opt=time.time() - start_time
print ('Done in')
print("--- %s seconds ---" % (time_opt))

#saving results

for j in range(boot_n):

    alpha_00[j] = dics[j][0]
    alpha_01[j] = dics[j][1]
    alpha_02[j] = dics[j][2]
    alpha_03[j] = dics[j][3]
    alpha_04[j] = dics[j][4]
    alpha_05[j] = dics[j][5]
    alpha_10[j] = dics[j][6]
    alpha_11[j] = dics[j][7]
    alpha_12[j] = dics[j][8]
    alpha_13[j] = dics[j][9]
    alpha_14[j] = dics[j][10]
    alpha_15[j] = dics[j][11]
    beta_0[j] = dics[j][12]
    beta_1[j] = dics[j][13]
    beta_2[j] = dics[j][14]
    beta_3[j] = dics[j][15]
    gamma_0[j] = dics[j][16]
    gamma_1[j] = dics[j][17]
    gamma_2[j] = dics[j][18]



est_alpha_00 = np.std(alpha_00)
est_alpha_01 = np.std(alpha_01)
est_alpha_02 = np.std(alpha_02)
est_alpha_03 = np.std(alpha_03)
est_alpha_04 = np.std(alpha_04)
est_alpha_05 = np.std(alpha_05)
est_alpha_10 = np.std(alpha_10)
est_alpha_11 = np.std(alpha_11)
est_alpha_12 = np.std(alpha_12)
est_alpha_13 = np.std(alpha_13)
est_alpha_14 = np.std(alpha_14)
est_alpha_15 = np.std(alpha_15)
est_beta_0 = np.std(beta_0)
est_beta_1 = np.std(beta_1)
est_beta_2 = np.std(beta_2)
est_beta_3 = np.std(beta_3)
est_gamma_0 = np.std(gamma_0)
est_gamma_1 = np.std(gamma_1)
est_gamma_2 = np.std(gamma_2)

dics_se = {'SE alpha_00': est_alpha_00,
                'SE alpha_01': est_alpha_01,
                'SE alpha_02': est_alpha_02,
                'SE alpha_03': est_alpha_03,
                'SE alpha_04': est_alpha_04,
                'SE alpha_05': est_alpha_05,
                'SE alpha_10': est_alpha_10,
                'SE alpha_11': est_alpha_11,
                'SE alpha_12': est_alpha_12,
                'SE alpha_13': est_alpha_13,
                'SE alpha_14': est_alpha_14,
                'SE alpha_15': est_alpha_15,
                'SE beta_0': est_beta_0,
                'SE beta_1': est_beta_1,
                'SE beta_2': est_beta_2,
                'SE beta_3': est_beta_3,
                'SE gamma_0': est_gamma_0,
                'SE gamma_1': est_gamma_1,
                'SE gamma_2': est_gamma_2}
    
betas_opt = np.array([dics_se['SE alpha_00'], dics_se['SE alpha_01'], 
                              dics_se['SE alpha_02'],dics_se['SE alpha_03'],dics_se['SE alpha_04'],
                              dics_se['SE alpha_05'],
                              dics_se['SE alpha_10'],dics_se['SE alpha_11'],dics_se['SE alpha_12'], 
                                  dics_se['SE alpha_13'],dics_se['SE alpha_14'],dics_se['SE alpha_15'],
                                  dics_se['SE beta_0'],
                                  dics_se['SE beta_1'],dics_se['SE beta_2'], 
                                      dics_se['SE beta_3'],dics_se['SE gamma_0'],dics_se['SE gamma_1'], dics_se['SE gamma_2']])



np.save('/home/jrodriguez/teachers/results/se_model_v3.npy',betas_opt)


