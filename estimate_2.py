# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:23:40 2020

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
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
#from pathos.multiprocessing import ProcessPool
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est




class estimate_2:
    "This class generate descriptive statiscals"
    
    def __init__(self,N,modelSD, years,treatment,p1_0,p2_0):
        "Initial class"
        
        self.N = N
        self.modelSD = modelSD
        self.years = years
        self.treatment = treatment
        self.p1_0 = p1_0
        self.p2_0 = p2_0        
        
    
    def simulation_2(self,times):
        "Function that simulate x times."
        
        est_corrSPort = np.zeros(times)
        est_corrSPrue = np.zeros(times)
        est_mean_Port = np.zeros(times)
        est_mean_Pru = np.zeros(times)
        est_var_Port = np.zeros(times)
        est_var_Pru = np.zeros(times)
        perc_init =  np.zeros(times)
        perc_inter =  np.zeros(times)
        perc_advan =  np.zeros(times)
        perc_expert =  np.zeros(times)
        est_corr_EXPPort = np.zeros(times)
        est_corr_EXPPru = np.zeros(times)
        est_mean_SIMCE = np.zeros(times)
        est_var_SIMCE = np.zeros(times)
        est_mean_PortTest = np.zeros(times)
        perc_inter_c = np.zeros(times)
        perc_avanexpet_c = np.zeros(times)
        est_corrTestp = np.zeros(times)
        est_corrPortp = np.zeros(times)
        
        for i in range(1,times):
            #defino el seed como el del profe, constante para cada muestra, seed i.
            np.random.seed(i+100)
            opt = self.modelSD.choice(self.treatment)
            dataf = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1], 
                     'EXP': self.years, 'TREATMENT': opt['Treatment'], 'PLACEMENT': opt['Opt Placement'], 
                     'PORTPAST': self.p1_0, 'TESTPAST': self.p2_0}
            # Here we consider the database complete
            datadfT = pd.DataFrame(dataf, columns=['SIMCE','PORTFOLIO','TEST', 'EXP', 'TREATMENT', 'PLACEMENT', 'PORTPAST', 'TESTPAST'])
            #1 Portfolio mean
            est_mean_Port[i] = np.mean(datadfT['PORTFOLIO'].to_numpy())
            #2 Portfolio var
            est_var_Port[i] = np.var(datadfT['PORTFOLIO'].to_numpy())
            #3 SIMCE mean
            est_mean_SIMCE[i] = np.mean(datadfT['SIMCE'].to_numpy())
            #4 SIMCE var
            est_var_SIMCE[i] = np.var(datadfT['SIMCE'].to_numpy())
            #5 Test mean
            est_mean_Pru[i] = np.mean(datadfT['TEST'].to_numpy())
            #6 Test var
            est_var_Pru[i] = np.var(datadfT['TEST'].to_numpy())
            #7 Portfolio-Test mean
            p1_past = np.array(datadfT['PORTPAST'])
            p2_past = np.array(datadfT['TESTPAST'])
            p1v1 = np.where(np.isnan(p1_past), 0, p1_past)
            p2v1 = np.where(np.isnan(p2_past), 0, p2_past)
            p0_past = np.zeros(p1_past.shape)
            p0_past = np.where((p1v1 == 0),p2v1, p0_past)
            p0_past = np.where((p2v1 == 0),p1v1, p0_past)
            p0_past = np.where((p1v1 != 0) & (p2v1 != 0) ,(p1_past + p2_past)/2, p0_past)
            dataf_past = {'TEST': opt['Opt Teacher'][1], 'PORTFOLIO': opt['Opt Teacher'][0], 'P_past': p0_past}
            datadf_past = pd.DataFrame(dataf_past, columns=['P_past','TEST','PORTFOLIO'])
            corrM = datadf_past.corr()
            est_corrTestp[i] = corrM.iloc[0]['TEST']
            est_corrPortp[i] = corrM.iloc[0]['PORTFOLIO']
            #est_mean_PortTest[i] = np.mean(p0)
            # Here we consider the data for treatmetn group
            datav = datadfT[datadfT['TREATMENT']==1]
            #8
            perc_init[i] = (sum(datav['PLACEMENT']==1) / len(datav['PLACEMENT'])) 
            #9
            perc_inter[i] = (sum(datav['PLACEMENT']==2) / len(datav['PLACEMENT'])) 
            #10
            perc_advan[i] = (sum(datav['PLACEMENT']==3) / len(datav['PLACEMENT'])) 
            #11
            perc_expert[i] = ((sum(datav['PLACEMENT']==4)+sum(datav['PLACEMENT']==5)) / len(datav['PLACEMENT'])) 
            datadf = pd.DataFrame(datav, columns=['SIMCE','PORTFOLIO','TEST', 'EXP'])
            corrM = datadf.corr()
            #12 SIMCE vs Portfolio
            est_corrSPort[i] = corrM.iloc[0]['PORTFOLIO']
            #13 SIMCE vs Test
            est_corrSPrue[i] = corrM.iloc[0]['TEST']
            # We don't calculate this corr   
            #est_corrPP[i] = corrM.iloc[1]['TEST']
            #14 Experience vs Portfolio 
            est_corr_EXPPort[i] = corrM.iloc[3]['PORTFOLIO']
            #15 Experience vs Test
            est_corr_EXPPru[i] = corrM.iloc[3]['TEST']
            #datav0 = datadfT[datadfT['TREATMENT']==0]
            datav_2 = datadfT[datadfT['TREATMENT']==0]
            perc_inter_c[i] = (sum(datav_2['PLACEMENT']==2) / len(datav_2['PLACEMENT']))
            perc_avanexpet_c[i] = (sum(datav_2['PLACEMENT']==3) / (sum(datav_2['PLACEMENT']==4)+sum(datav_2['PLACEMENT']==5))) / len(datav_2['PLACEMENT'])
            p1 = datav_2['PORTFOLIO'].to_numpy()
            p2 = datav_2['TEST'].to_numpy()
            p1v1 = np.where(np.isnan(p1), 0, p1)
            p2v1 = np.where(np.isnan(p2), 0, p2)
            p0 = np.zeros(p1.shape)
            p0 = np.where((p1v1 == 0),p2v1, p0)
            p0 = np.where((p2v1 == 0),p1v1, p0)
            p0 = np.where((p1v1 != 0) & (p2v1 != 0) ,(p1 + p2)/2, p0)
            est_mean_PortTest[i] = np.mean(p0)
        
        est_bootsSPort = np.mean(est_corrSPort)
        est_bootsSPrue = np.mean(est_corrSPrue)
        #est_sim_PP = np.mean(est_corrPP)
        est_sim_mean_Port = np.mean(est_mean_Port)
        est_sim_var_Port = np.mean(est_var_Port)
        est_sim_mean_Pru = np.mean(est_mean_Pru)
        est_sim_var_Test = np.mean(est_var_Pru)
        est_sim_perc_init = np.mean(perc_init)
        est_sim_EXPPort = np.mean(est_corr_EXPPort)
        est_sim_EXPPru = np.mean(est_corr_EXPPru)
        est_sim_perc_inter = np.mean(perc_inter)
        est_sim_perc_advan = np.mean(perc_advan)
        est_sim_perc_expert = np.mean(perc_expert)
        est_sim_mean_SIMCE = np.mean(est_mean_SIMCE)
        est_sim_var_SIMCE = np.mean(est_var_SIMCE)
        est_sim_mean_PP = np.mean(est_mean_PortTest)
        est_sim_inter_c = np.mean(perc_inter_c)
        est_sim_advexp_c = np.mean(perc_avanexpet_c)
        est_sim_Testp = np.mean(est_corrTestp)
        est_sim_Portp = np.mean(est_corrPortp)
        #error_bootsSPort = np.std(est_corrSPort)
        #error_bootsSPrue = np.std(est_corrSPrue)
        #plt.hist(est_corrSPort, bins=100)
        #plt.axvline(est_bootsSPort, color='r', linestyle='dashed', linewidth=1)
        #plt.title("Histogram Portfolio")
        #plt.hist(est_corrSPrue, bins=100)
        #plt.axvline(est_bootsSPrue, color='r', linestyle='dashed', linewidth=1)
        #plt.title("Histogram Test")
        #sn.heatmap(corrMatrix, annot=True)
        #plt.show()
        
        return {'Estimation SIMCE vs Portfolio': est_bootsSPort,
            'Estimation SIMCE vs Prueba': est_bootsSPrue,
            #'Estimation Portfolio vs Prueba': est_sim_PP,
            'Estimation EXP vs Portfolio': est_sim_EXPPort,
            'Estimation EXP vs Prueba': est_sim_EXPPru,
            'Mean Portfolio': est_sim_mean_Port,
            'Var Port': est_sim_var_Port,
            'Mean Test': est_sim_mean_Pru,
            'Var Test': est_sim_var_Test,
            'perc init': est_sim_perc_init,
            'perc inter': est_sim_perc_inter,
            'perc advanced': est_sim_perc_advan,
            'perc expert': est_sim_perc_expert,
            'Mean SIMCE': est_sim_mean_SIMCE,
            'Var SIMCE': est_sim_var_SIMCE,
            'Mean PortTest' : est_sim_mean_PP,
            'perc inter control': est_sim_inter_c,
            'perc adv/exp control': est_sim_advexp_c,
            'Estimation Test vs p': est_sim_Testp,
            'Estimation Portfolio vs p': est_sim_Portp}
                


