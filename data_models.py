# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:00:56 2023

@author: Patricio De Araya
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
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from tabulate import tabulate
from texttable import Texttable
import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
import time
#import int_linear
sys.path.append("/home/jrodriguezo/teachers/codes")
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
from scipy.stats import norm
from linearmodels.panel import PanelOLS

#ver https://pythonspeed.com/articles/python-multiprocessing/
import multiprocessing as mp
from multiprocessing import Pool

#---This function delivers att data - att model---#

def data_model(j):
    
    data_reg = pd.read_pickle("data_reg.pkl")
    n = data_reg.shape[0]

    np.random.seed(j+100)
    
    
    #data_reg = data_reg.sample(n, replace=True)

    #---------------------------------------------#
    #--------Estimating ATT from data-------------#
    #---------------------------------------------#

    # first drop Stata 1083190 rows 
    data_reg = data_reg[(data_reg["stdsimce_m"].notna()) & (data_reg["stdsimce_l"].notna())]

    #destring
    data_reg["drun_l"] = pd.to_numeric(data_reg["drun_l"], errors='coerce')
    data_reg["drun_m"] = pd.to_numeric(data_reg["drun_m"], errors='coerce')


    ##### generates variables #####
    #eval_year
    data_reg.loc[data_reg["eval_year_m"]==data_reg["eval_year_l"],'eval_year'] = data_reg["eval_year_m"]
    data_reg.loc[(data_reg["eval_year_m"].notna()) & (data_reg["eval_year_l"].isna()),'eval_year'] = data_reg["eval_year_m"]
    data_reg.loc[(data_reg["eval_year_m"].isna()) & (data_reg["eval_year_l"].notna()),'eval_year'] = data_reg["eval_year_l"]

    #drun
    data_reg.loc[data_reg["drun_m"]==data_reg["drun_l"],'drun'] = data_reg["drun_m"]
    data_reg.loc[(data_reg["drun_m"].notna()) & (data_reg["drun_l"].isna()),'drun'] = data_reg["drun_m"]
    data_reg.loc[(data_reg["drun_m"].isna()) & (data_reg["drun_l"].notna()),'drun'] = data_reg["drun_l"]

    #experience
    data_reg.loc[data_reg["experience_m"]==data_reg["experience_l"],'experience'] = data_reg["experience_m"]
    data_reg.loc[(data_reg["experience_m"].notna()) & (data_reg["experience_l"].isna()),'experience'] = data_reg["experience_m"]
    data_reg.loc[(data_reg["experience_m"].isna()) & (data_reg["experience_l"].notna()),'experience'] = data_reg["experience_l"]

    #d_trat
    data_reg.loc[data_reg["d_trat_m"]==data_reg["d_trat_l"],'d_trat'] = data_reg["d_trat_m"]
    data_reg.loc[(data_reg["d_trat_m"].notna()) & (data_reg["d_trat_l"].isna()),'d_trat'] = data_reg["d_trat_m"]
    data_reg.loc[(data_reg["d_trat_m"].isna()) & (data_reg["d_trat_l"].notna()),'d_trat'] = data_reg["d_trat_l"]

    #inter
    data_reg.loc[data_reg["inter_m"]==data_reg["inter_l"],'inter'] = data_reg["inter_m"]
    data_reg.loc[(data_reg["inter_m"].notna()) & (data_reg["inter_l"].isna()),'inter'] = data_reg["inter_m"]
    data_reg.loc[(data_reg["inter_m"].isna()) & (data_reg["inter_l"].notna()),'inter'] = data_reg["inter_l"]

    #d_year
    data_reg.loc[data_reg["d_year_m"]==data_reg["d_year_l"],'d_year'] = data_reg["d_year_m"]
    data_reg.loc[(data_reg["d_year_m"].notna()) & (data_reg["d_year_l"].isna()),'d_year'] = data_reg["d_year_m"]
    data_reg.loc[(data_reg["d_year_m"].isna()) & (data_reg["d_year_l"].notna()),'d_year'] = data_reg["d_year_l"]
            
    ##### drop nan #####
    data_reg = data_reg[(data_reg["edp"].notna())]
    data_reg = data_reg[(data_reg["edm"].notna())]
    data_reg = data_reg[(data_reg["ingreso"].notna())]
    data_reg = data_reg[(data_reg["experience"].notna())]
    data_reg = data_reg[(data_reg["drun"].notna())]
    data_reg = data_reg[(data_reg["d_trat"].notna())]
    data_reg = data_reg[(data_reg["d_year"].notna())]
    data_reg = data_reg[(data_reg["inter"].notna())]
                                                                       
    # keep if eval_year == 1 | eval_year == 2018 | eval_year == 0
    data_reg = data_reg[(data_reg["eval_year"] == 1) | (data_reg["eval_year"] == 2018) | (data_reg["eval_year"] == 0)]

    # mean simce
    data_reg['stdsimce'] = data_reg[['stdsimce_m', 'stdsimce_l']].mean(axis=1)
    
    
    #Resampling
    personids = data_reg.drun.unique()
    sampled_personids = np.random.choice(personids, size=personids.size, replace=True)
    data_reg.set_index('drun',inplace = True)
    data_reg = data_reg.loc[sampled_personids]

    #Redefining index for school FEs
    data_reg.reset_index(inplace = True)
    data_reg.set_index(['rbd','agno'],inplace = True)
    data_reg['constant'] = np.ones(np.size(data_reg['stdsimce']))
    model_reg = PanelOLS(dependent = data_reg['stdsimce'], exog = data_reg[['constant', 'd_trat', 'd_year', 'inter', 'edp', 'edm', 'ingreso', 'experience']], entity_effects = True)
    results = model_reg.fit(cov_type='clustered', clusters=data_reg['drun'])
    att_data = results.params.inter.round(8)

    
    
    #---------------------------------------------#
    #--------Estimating ATT from model------------#
    #---------------------------------------------#

    ##get python data from bootstrapped data
    
    #---------------------------------------------#
    data_reg.reset_index(inplace = True)

    data_reg = data_reg[(((data_reg["agno"] == 2016) | (data_reg["agno"] == 2017)) & (data_reg["eval_year"] == 1))| ((data_reg["agno"] < 2016) & (data_reg["eval_year"] != 1))]
    

    #gen typeschool=0 replace typeschool = 1 if grado==4 | grado==8
    data_reg['typeschool'] = 1

    #egen stdsimce = rowmean(stdsimce_m stdsimce_l) drop stdsimce_m stdsimce_l
    # mean simce
    data_reg['stdsimce'] = data_reg[['stdsimce_m', 'stdsimce_l']].mean(axis=1)
    data_python = data_reg.drop(['stdsimce_m', 'stdsimce_l'], axis = 1)


    #tramo_a2016
    data_reg.loc[data_reg["tramo_a2016_m"]==data_reg["tramo_a2016_l"],'tramo_a2016'] = data_reg["tramo_a2016_m"]
    data_reg.loc[(data_reg["tramo_a2016_m"] != "") & (data_reg["tramo_a2016_l"]== ""),'tramo_a2016'] = data_reg["tramo_a2016_m"]
    data_reg.loc[(data_reg["tramo_a2016_m"]== "") & (data_reg["tramo_a2016_l"]!= ""),'tramo_a2016'] = data_reg["tramo_a2016_l"]

    #cat_portafolio_a2016
    data_reg.loc[data_reg["cat_portafolio_a2016_m"]==data_reg["cat_portafolio_a2016_l"],'cat_portafolio_a2016'] = data_reg["cat_portafolio_a2016_m"]
    data_reg.loc[(data_reg["cat_portafolio_a2016_m"] != "") & (data_reg["cat_portafolio_a2016_l"]== ""),'cat_portafolio_a2016'] = data_reg["cat_portafolio_a2016_m"]
    data_reg.loc[(data_reg["cat_portafolio_a2016_m"]== "") & (data_reg["cat_portafolio_a2016_l"]!= ""),'cat_portafolio_a2016'] = data_reg["cat_portafolio_a2016_l"]

    #cat_prueba_a2016
    data_reg.loc[data_reg["cat_prueba_a2016_m"]==data_reg["cat_prueba_a2016_l"],'cat_prueba_a2016'] = data_reg["cat_prueba_a2016_m"]
    data_reg.loc[(data_reg["cat_prueba_a2016_m"] != "") & (data_reg["cat_prueba_a2016_l"]== ""),'cat_prueba_a2016'] = data_reg["cat_prueba_a2016_m"]
    data_reg.loc[(data_reg["cat_prueba_a2016_m"]== "") & (data_reg["cat_prueba_a2016_l"]!= ""),'cat_prueba_a2016'] = data_reg["cat_prueba_a2016_l"]

    #cat_portafolio_rec2016
    data_reg.loc[data_reg["cat_portafolio_rec2016_m"]==data_reg["cat_portafolio_rec2016_l"],'cat_portafolio_rec2016'] = data_reg["cat_portafolio_rec2016_m"]
    data_reg.loc[(data_reg["cat_portafolio_rec2016_m"] != "") & (data_reg["cat_portafolio_rec2016_l"]== ""),'cat_portafolio_rec2016'] = data_reg["cat_portafolio_rec2016_m"]
    data_reg.loc[(data_reg["cat_portafolio_rec2016_m"]== "") & (data_reg["cat_portafolio_rec2016_l"]!= ""),'cat_portafolio_rec2016'] = data_reg["cat_portafolio_rec2016_l"]

    #cat_prueba_rec2016
    data_reg.loc[data_reg["cat_prueba_rec2016_m"]==data_reg["cat_prueba_rec2016_l"],'cat_prueba_rec2016'] = data_reg["cat_prueba_rec2016_m"]
    data_reg.loc[(data_reg["cat_prueba_rec2016_m"] != "") & (data_reg["cat_prueba_rec2016_l"]== ""),'cat_prueba_rec2016'] = data_reg["cat_prueba_rec2016_m"]
    data_reg.loc[(data_reg["cat_prueba_rec2016_m"]== "") & (data_reg["cat_prueba_rec2016_l"]!= ""),'cat_prueba_rec2016'] = data_reg["cat_prueba_rec2016_l"]

    #tramo_rec2016
    data_reg.loc[data_reg["tramo_rec2016_m"]==data_reg["tramo_rec2016_l"],'tramo_rec2016'] = data_reg["tramo_rec2016_m"]
    data_reg.loc[(data_reg["tramo_rec2016_m"] != "") & (data_reg["tramo_rec2016_l"]== ""),'tramo_rec2016'] = data_reg["tramo_rec2016_m"]
    data_reg.loc[(data_reg["tramo_rec2016_m"]== "") & (data_reg["tramo_rec2016_l"]!= ""),'tramo_rec2016'] = data_reg["tramo_rec2016_l"]

    #cat_portafolio_rec2018
    data_reg.loc[data_reg["cat_portafolio_rec2018_m"]==data_reg["cat_portafolio_rec2018_l"],'cat_portafolio_rec2018'] = data_reg["cat_portafolio_rec2018_m"]
    data_reg.loc[(data_reg["cat_portafolio_rec2018_m"] != "") & (data_reg["cat_portafolio_rec2018_l"]== ""),'cat_portafolio_rec2018'] = data_reg["cat_portafolio_rec2018_m"]
    data_reg.loc[(data_reg["cat_portafolio_rec2018_m"]== "") & (data_reg["cat_portafolio_rec2018_l"]!= ""),'cat_portafolio_rec2018'] = data_reg["cat_portafolio_rec2018_l"]

    #cat_prueba_rec2018
    data_reg.loc[data_reg["cat_prueba_rec2018_m"]==data_reg["cat_prueba_rec2018_l"],'cat_prueba_rec2018'] = data_reg["cat_prueba_rec2018_m"]
    data_reg.loc[(data_reg["cat_prueba_rec2018_m"] != "") & (data_reg["cat_prueba_rec2018_l"]== ""),'cat_prueba_rec2018'] = data_reg["cat_prueba_rec2018_m"]
    data_reg.loc[(data_reg["cat_prueba_rec2018_m"]== "") & (data_reg["cat_prueba_rec2018_l"]!= ""),'cat_prueba_rec2018'] = data_reg["cat_prueba_rec2018_l"]

    #tramo_rec2018
    data_reg.loc[data_reg["tramo_rec2018_m"]==data_reg["tramo_rec2018_l"],'tramo_rec2018'] = data_reg["tramo_rec2018_m"]
    data_reg.loc[(data_reg["tramo_rec2018_m"] != "") & (data_reg["tramo_rec2018_l"]== ""),'tramo_rec2018'] = data_reg["tramo_rec2018_m"]
    data_reg.loc[(data_reg["tramo_rec2018_m"]== "") & (data_reg["tramo_rec2018_l"]!= ""),'tramo_rec2018'] = data_reg["tramo_rec2018_l"]

    #distance
    data_reg.loc[data_reg["distance_m"]==data_reg["distance_l"],'distance'] = data_reg["distance_m"]
    data_reg.loc[(data_reg["distance_m"] != "") & (data_reg["distance_l"]== ""),'distance'] = data_reg["distance_m"]
    data_reg.loc[(data_reg["distance_m"]== "") & (data_reg["distance_l"]!= ""),'distance'] = data_reg["distance_l"]

    #xtile distance2 = XY_distance, nq(5) replace distance2 = 1 if  distance=="TOP TEACHER"
    #data_reg['distance2'] = pd.qcut(data_reg['XY_distance'], 5, labels = False)
    #data_reg.loc[data_reg["distance"]=="TOP TEACHER",'distance2'] = 0

    #save the string variables
    data_pythonpast2 = data_reg[['cat_portafolio_a2016', 'cat_prueba_a2016', 'cat_portafolio_rec2016', 'cat_prueba_rec2016',
                                        'tramo_rec2016', 'cat_portafolio_rec2018', 'tramo_a2016', 'cat_prueba_rec2018', 'tramo_rec2018',
                                        'drun']]


    # Merged numeric and string variables by drun
    data_python = pd.merge(data_reg, data_pythonpast2, on='drun')

    #score_port
    #gen score_port = . replace score_port = ptj_portafolio_rec2016 if eval_year == 1
    data_python.loc[data_python["eval_year"]==1,'score_port'] = data_python["ptj_portafolio_rec2016"]

    #drop ptj_portafolio_rec2016 ptj_portafolio_rec2018
    data_python = data_python.drop(['ptj_portafolio_rec2016', 'ptj_portafolio_rec2018'], axis=1)

    #gen score_test = . replace score_test = ptj_prueba_rec2016 if eval_year == 1
    data_python.loc[data_python["eval_year"]==1,'score_test'] = data_python["ptj_prueba_rec2016"]

    #drop ptj_prueba_rec2016 ptj_prueba_rec2018
    data_python = data_python.drop(['ptj_prueba_rec2016', 'ptj_prueba_rec2018'], axis=1)

    #gen cat_port = "" replace cat_port = cat_portafolio_rec2016 if cat_portafolio_rec2018 == ""
    data_python.loc[data_python["cat_portafolio_rec2018"]=="",'cat_port'] = data_python["cat_portafolio_rec2016"]

    #drop cat_portafolio_rec2016 cat_portafolio_rec2018
    data_python = data_python.drop(['cat_portafolio_rec2016', 'cat_portafolio_rec2018'], axis=1)

    #gen cat_test = "" replace cat_test = cat_prueba_rec2016 if cat_prueba_rec2018 == ""
    data_python.loc[data_python["cat_prueba_rec2018"]=="",'cat_test'] = data_python["cat_prueba_rec2016"]

    #drop cat_prueba_rec2016 cat_prueba_rec2018
    data_python = data_python.drop(['cat_prueba_rec2016', 'cat_prueba_rec2018'], axis=1)

    #gen trame = "" replace trame = tramo_rec2016 if eval_year == 1 replace trame = tramo_a2016 if eval_year != 1
    data_python.loc[data_python["eval_year"]==1,'trame'] = data_python["tramo_rec2016"]
    data_python.loc[data_python["eval_year"] != 1,'trame'] = data_python["tramo_a2016"]

    #drop tramo_rec2016 tramo_rec2018
    data_python = data_python.drop(['tramo_rec2016', 'tramo_rec2018'], axis=1)

    #rename  ptj_portafolio_a2016 score_port_past rename ptj_prueba_a2016 score_test_past
    data_python = data_python.rename(columns={'ptj_portafolio_a2016': 'score_port_past'})
    data_python = data_python.rename(columns={'ptj_prueba_a2016': 'score_test_past'})

    #drop if score_test_past == . & score_port_past == .
    indexAge = data_python[ (data_python['score_test_past'].isna()) & (data_python['score_port_past'].isna()) ].index
    data_python.drop(indexAge , inplace=True)

    #foreach variable in "stdsimce" "experience" "d_trat" "eval_year" "typeschool" "por_priority" "rural_rbd" "AsignacionZona"{
    #   drop if `variable' == .
    #   }
    data_python.drop(data_python[data_python['stdsimce'].isna()].index, inplace = True)
    data_python.drop(data_python[data_python['experience'].isna()].index, inplace = True)
    data_python.drop(data_python[data_python['d_trat'].isna()].index, inplace = True)
    data_python.drop(data_python[data_python['eval_year'].isna()].index, inplace = True)
    data_python.drop(data_python[data_python['typeschool'].isna()].index, inplace = True)
    data_python.drop(data_python[data_python['por_priority'].isna()].index, inplace = True)
    data_python.drop(data_python[data_python['rural_rbd'].isna()].index, inplace = True)
    data_python.drop(data_python[data_python['AsignacionZona'].isna()].index, inplace = True)

    #---------------------------------------------#

    ses_opt = np.load("/home/jrodriguezo/teachers/codes/ses_model_v2023.npy")
    w_matrix = np.zeros((ses_opt.shape[0],ses_opt.shape[0]))
    for j in range(ses_opt.shape[0]):
        w_matrix[j,j] = ses_opt[j]**(-2)
    
    n = data_python.shape[0]
    
    n_sim = 100


    # TREATMENT #
    #treatmentOne = rev[['d_trat']]
    treatment = np.array(data_python['d_trat'])
    # EXPERIENCE #
    #yearsOne = rev[['experience']]
    years = np.array(data_python['experience'])
    # SCORE PORTFOLIO #
    #p1_0_1 = rev[['score_port']]
    p1_0 = np.array(data_python['score_port'])
    p1 = np.array(data_python['score_port'])
    # SCORE TEST #
    #p2_0_1 = rev[['score_test']]
    p2_0 = np.array(data_python['score_test'])
    p2 = np.array(data_python['score_test'])
    # CATEGORY PORTFOLIO #
    #categPortfolio = rev[['cat_port']]
    catPort = np.array(data_python['cat_port'])
    # CATEGORY TEST #
    #categPrueba = data_python[['cat_test']]
    catPrueba = np.array(data_python['cat_test'])
    # TRAME #
    #TrameInitial = rev[['trame']]
    TrameI = np.array(data_python['trame'])
    # TYPE SCHOOL #
    #typeSchoolOne = data_python[['typeschool']]
    typeSchool = np.array(data_python['typeschool'])
    
    
    # Priority #
    priotity = np.array(data_python['por_priority'])
    
    priotity_aep = np.array(data_python['priority_aep'])
    
    rural_rbd = np.array(data_python['rural_rbd'])
    
    locality = np.array(data_python['AsignacionZona'])


    

    #### PARAMETERS MODEL ####
    N = np.size(p1_0)
    HOURS = np.array([44]*N)
    alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
          betas_nelder[3], betas_nelder[4]],
         [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
          betas_nelder[8], betas_nelder[9]]]
    
    betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12] ,betas_nelder[13],betas_nelder[14],betas_nelder[15]]
    gammas = [betas_nelder[16],betas_nelder[17],betas_nelder[18]]
    
    dolar= 600
    value = [14403, 15155]
    hw = [value[0]/dolar,value[1]/dolar]
    porc = [0.0338, 0.0333]
    #inflation adjustment: 2012Jan-2019Dec: 1.266
    qualiPesos = [72100*1.266, 24034*1.266, 253076, 84360] 
    pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]
    progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]
    pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,
           progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]
    
    pri = [47872,113561]
    priori = [pri[0]/dolar, pri[1]/dolar]

    Asig = [150000*1.111,100000*1.111,50000*1.111]
    AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar] 

    param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)
        
    output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
         typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, priotity,rural_rbd,locality, priotity_aep, \
         w_matrix,moments_vector)
    
    start_time = time.time()
    output_me = output_ins.optimizer()
    time_opt=time.time() - start_time
    print ('Done in')
    print("--- %s seconds ---" % (time_opt))

    beta_1 = output_me.x[0]
    beta_2 = output_me.x[1]
    beta_3 = output_me.x[2]
    beta_4 = np.exp(output_me.x[3])
    beta_5 = output_me.x[4]
    beta_6 = output_me.x[5]
    beta_7 = output_me.x[6]
    beta_8 = output_me.x[7]
    beta_9 = np.exp(output_me.x[8])
    beta_10 = output_me.x[9]
    beta_11 = output_me.x[10]
    beta_12 = output_me.x[11]
    beta_13 = output_me.x[12]
    beta_14 = output_me.x[13]
    beta_15 = output_me.x[14]
    beta_16 = output_me.x[15]
    beta_17 = output_me.x[16]
    beta_18 = output_me.x[17]
    beta_19 = output_me.x[18]


    betas_opt = np.array([beta_1, beta_2,
		beta_3,
		beta_4,beta_5,beta_6,beta_7,beta_8,
		beta_9,beta_10,beta_11,beta_12,
		beta_13,beta_14,beta_15,
		beta_16,beta_17,beta_18,beta_19])

    #Updating parameters to compute ATT
    alphas = [[betas_opt[0], betas_opt[1],0,betas_opt[2],
          betas_opt[3], betas_opt[4]],
         [betas_opt[5], 0,betas_opt[6],betas_opt[7],
          betas_opt[8], betas_opt[9]]]
    
    betas = [betas_opt[10], betas_opt[11], betas_opt[12] ,betas_opt[13],betas_opt[14],betas_opt[15]]
    
    gammas = [betas_opt[16],betas_opt[17],betas_opt[18]]

    param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)



    # SIMULACIÓN SIMDATA
    
    simce_sims = np.zeros((N,n_sim))

    data_python = data_python[data_python['d_trat']==1]

    simce = []

    for x in range(0,2):

        treatment = np.ones(N)*x

        model = Count_att(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality,priotity_aep)


        for j in range(n_sim):
    	        modelSD = sd.SimData(N,model)
    	        opt = modelSD.choice()
    	        simce_sims[:,j] = opt['Opt Simce']
	    
        simce.append(np.mean(simce_sims,axis=1))

    att_sim = simce[1] - simce[0]


    return [att_data - np.mean(att_sim)]
