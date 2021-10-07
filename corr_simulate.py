# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:14:43 2020

@author: pjac2
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:49:00 2020

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
import statsmodels.api as sm
from tabulate import tabulate
from texttable import Texttable
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook


#### LOAD DATA ####

#df = pd.read_stata('D:\Git\ExpSIMCE/data_pythonpast.dta')
df = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/data_pythonpast.dta')

pd.value_counts(df['trame'])

df['trame'] = df['trame'].replace(['INICIAL', 'TEMPRANO', 'AVANZADO', 'EXPERTO I', 'EXPERTO II'], [1,2,3,4,5]) 
print(df['trame'])


#### BOOTSTRAP ####

def corr_simulate(data, B):
    n = data.shape[0]
    est_corrSPort = np.zeros(B)
    est_corrSPrue = np.zeros(B)
    est_corrSEXP = np.zeros(B)
    est_corr_EXPPort = np.zeros(B)
    est_corr_EXPPru = np.zeros(B)
    est_mean_Port = np.zeros(B)
    est_mean_Pru = np.zeros(B)
    est_var_Port = np.zeros(B)
    est_var_Pru = np.zeros(B)
    #perc_init =  np.zeros(B)
    perc_inter =  np.zeros(B)
    perc_advan =  np.zeros(B)
    perc_expert =  np.zeros(B)
    est_mean_SIMCE = np.zeros(B)
    est_var_SIMCE = np.zeros(B)
    est_mean_PortTest = np.zeros(B)
    perc_avanexpet_c = np.zeros(B)
    est_corrTestp = np.zeros(B)
    est_corrPortp = np.zeros(B)

    
    for i in range(1,B):
        rev = data.sample(n, replace=True)

        est_mean_SIMCE[i] = np.mean(rev['stdsimce_m'])
        est_var_SIMCE[i] = np.var(rev['stdsimce_m'])

        p1_past = np.array(rev['score_port_past'])
        p2_past = np.array(rev['score_test_past'])
        p1v1 = np.where(np.isnan(p1_past), 0, p1_past)
        p2v1 = np.where(np.isnan(p2_past), 0, p2_past)
        p0_past = np.zeros(p1_past.shape)
        p0_past = np.where((p1v1 == 0),p2v1, p0_past)
        p0_past = np.where((p2v1 == 0),p1v1, p0_past)
        p0_past = np.where((p1v1 != 0) & (p2v1 != 0) ,(p1_past + p2_past)/2, p0_past)
        dataf_past = {'TEST': rev['score_port'], 'PORTFOLIO': rev['score_test'], 'P_past': p0_past, 'SIMCE': rev['stdsimce_m'],
                      'EXP': rev['experience']}
        datadf_past = pd.DataFrame(dataf_past, columns=['P_past','TEST','PORTFOLIO','SIMCE','EXP'])
        
        corrM = datadf_past.corr()
        est_corrSEXP[i] = corrM.iloc[3]['EXP']

        
        datav = rev[rev['d_trat']==1]
        
        corrM = datadf_past[rev['d_trat']==1].corr()
        est_corrTestp[i] = corrM.iloc[0]['TEST']
        est_corrPortp[i] = corrM.iloc[0]['PORTFOLIO']
        
        est_mean_Port[i] = np.mean(datav['score_port'])
        est_var_Port[i] = np.var(datav['score_port'])
        est_mean_Pru[i] = np.mean(datav['score_test'])
        est_var_Pru[i] = np.var(datav['score_test'])
        #perc_init[i] = (sum(datav['trame']==1) / len(datav['trame'])) 
        perc_inter[i] = (sum(datav['trame']==2) / len(datav['trame'])) 
        perc_advan[i] = (sum(datav['trame']==3) / len(datav['trame'])) 
        perc_expert[i] = ((sum(datav['trame']==4)+sum(datav['trame']==5)) / len(datav['trame'])) 
        datav1 = {'SIMCE': datav['stdsimce_m'], 'PORTFOLIO': datav['score_port'], 'TEST': datav['score_test'], 'EXP': datav['experience']}
        datadf = pd.DataFrame(datav1, columns=['SIMCE','PORTFOLIO','TEST', 'EXP'])
        corrM = datadf.corr()
        est_corrSPort[i] = corrM.iloc[0]['PORTFOLIO']
        est_corrSPrue[i] = corrM.iloc[0]['TEST']

        est_corr_EXPPort[i] = corrM.iloc[3]['PORTFOLIO']
        est_corr_EXPPru[i] = corrM.iloc[3]['TEST']
       
         
        datav_2 = rev[rev['d_trat']==0]
        perc_avanexpet_c[i] = ((sum(datav_2['trame']==3) + sum(datav_2['trame']==4)+sum(datav_2['trame']==5))) / len(datav_2['trame'])
        p1 = datav_2['score_port_past'].to_numpy()
        p2 = datav_2['score_test_past'].to_numpy()
        p1v1 = np.where(np.isnan(p1), 0, p1)
        p2v1 = np.where(np.isnan(p2), 0, p2)
        p0 = np.zeros(p1.shape)
        p0 = np.where((p1v1 == 0),p2v1, p0)
        p0 = np.where((p2v1 == 0),p1v1, p0)
        p0 = np.where((p1v1 != 0) & (p2v1 != 0) ,(p1 + p2)/2, p0)
        est_mean_PortTest[i] = np.mean(p0)
        
    est_sim_SPort = np.mean(est_corrSPort)
    est_sim_Prue = np.mean(est_corrSPrue)
    est_sim_SEXP = np.mean(est_corrSEXP)
    est_sim_EXPPort = np.mean(est_corr_EXPPort)
    est_sim_EXPPru = np.mean(est_corr_EXPPru)
    est_sim_mean_Port = np.mean(est_mean_Port)
    est_sim_var_Port = np.mean(est_var_Port)
    est_sim_mean_Pru = np.mean(est_mean_Pru)
    est_sim_var_Test = np.mean(est_var_Pru)
    #est_sim_perc_init = np.mean(perc_init)
    est_sim_perc_inter = np.mean(perc_inter)
    est_sim_perc_advan = np.mean(perc_advan)
    est_sim_perc_expert = np.mean(perc_expert)
    est_sim_mean_SIMCE = np.mean(est_mean_SIMCE)
    est_sim_var_SIMCE = np.mean(est_var_SIMCE)
    est_sim_mean_PP = np.mean(est_mean_PortTest)
    est_sim_advexp_c = np.mean(perc_avanexpet_c)
    est_sim_Testp = np.mean(est_corrTestp)
    est_sim_Portp = np.mean(est_corrPortp)

    
    error_SPort = np.std(est_corrSPort)
    error_SPru = np.std(est_corrSPrue)
    error_SEXP = np.std(est_corrSEXP)
    error_EXPPort = np.std(est_corr_EXPPort)
    error_EXPPru = np.std(est_corr_EXPPru)
    error_mean_Port = np.std(est_mean_Port)
    error_var_Port = np.std(est_var_Port)
    error_mean_Pru = np.std(est_mean_Pru)
    error_var_Pru = np.std(est_var_Pru)
    #error_init = np.std(perc_init)
    error_inter = np.std(perc_inter)
    error_advan = np.std(perc_advan)
    error_expert = np.std(perc_expert)
    error_mean_SIMCE = np.std(est_mean_SIMCE)
    error_var_SIMCE = np.std(est_var_SIMCE)
    error_mean_PP = np.std(est_mean_PortTest)
    error_advexp_c_PP = np.std(perc_avanexpet_c)
    error_Testp = np.std(est_corrTestp)
    error_Portp = np.std(est_corrPortp)

    
    
    #var-cov matrix
    samples = np.array([est_corrSPort,est_corrSPrue,est_corrSEXP,est_corr_EXPPort,est_corr_EXPPru,est_mean_Port,
                     est_var_Port,est_mean_Pru,est_var_Pru,
                     perc_inter,perc_advan,perc_expert,
                     est_mean_SIMCE,est_var_SIMCE,
                     est_mean_PortTest,perc_avanexpet_c,est_corrTestp,est_corrPortp])
    
    varcov = np.cov(samples)

    return {'Estimation SIMCE vs Portfolio': est_sim_SPort,
            'Estimation SIMCE vs Prueba': est_sim_Prue,
            'Estimation SIMCE vs Experience': est_sim_SEXP,
            'Estimation EXP vs Portfolio': est_sim_EXPPort,
            'Estimation EXP vs Prueba': est_sim_EXPPru,
            'Mean Portfolio': est_sim_mean_Port,
            'Var Portfolio': est_sim_var_Port,
            'Mean Test': est_sim_mean_Pru,
            'Var Test': est_sim_var_Test,
            #'perc init': est_sim_perc_init,
            'perc inter': est_sim_perc_inter,
            'perc advanced': est_sim_perc_advan,
            'perc expert': est_sim_perc_expert,
            'Mean SIMCE': est_sim_mean_SIMCE,
            'Var SIMCE': est_sim_var_SIMCE,
            'Mean PortTest': est_sim_mean_PP,
            'perc adv/exp control': est_sim_advexp_c,
            'Estimation Portfolio vs p': est_sim_Testp,
            'Estimation Test vs p': est_sim_Portp,
                'Error SIMCE vs Portfolio': error_SPort,
                'Error SIMCE vs Test': error_SPru,
                'Error SIMCE vs Experience': error_SEXP,
                'Error Exp vs Portfolio': error_EXPPort,
                'Error Exp vs Pru': error_EXPPru,
                'Error mean Port': error_mean_Port,
                'Error var Portfolio': error_var_Port,
                'Error mean Test': error_mean_Pru,
                'Error var Test': error_var_Pru,
                #'Error init': error_init,
                'Error inter': error_inter,
                'Error advanced': error_advan,
                'Error expert': error_expert,
                'Error SIMCE': error_mean_SIMCE,
                'Error var SIMCE': error_var_SIMCE,
                'Error mean Port-Test': error_mean_PP,
                'Error adv/exp control': error_advexp_c_PP,
                'Error Portfolio vs p': error_Testp,
                'Error Test vs p': error_Portp,
                'Var Cov Matrix': varcov}


result = corr_simulate(df,1000)
print(result)

varcov = result['Var Cov Matrix']

##### PYTHON TO EXCEL #####

wb = load_workbook('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/Outcomes.xlsx')
sheet = wb["data"]


sheet['E5'] = result['Mean Portfolio']
sheet['E6'] = result['Var Portfolio']
sheet['E7'] = result['Mean SIMCE']
sheet['E8'] = result['Var SIMCE']
sheet['E9'] = result['Mean Test']
sheet['E10'] = result['Var Test']
sheet['E11'] = result['Mean PortTest']
sheet['E12'] = result['perc inter']
sheet['E13'] = result['perc advanced']
sheet['E14'] = result['perc expert']
sheet['E15'] = result['Estimation SIMCE vs Portfolio']
sheet['E16'] = result['Estimation SIMCE vs Prueba']
sheet['E17'] = result['Estimation EXP vs Portfolio']
sheet['E18'] = result['Estimation EXP vs Prueba']
sheet['E19'] = result['perc adv/exp control']
sheet['E20'] = result['Estimation Test vs p']
sheet['E21'] = result['Estimation Portfolio vs p']
sheet['E22'] = result['Estimation SIMCE vs Experience']




sheet['F5'] = result['Error mean Port']
sheet['F6'] = result['Error var Portfolio']
sheet['F7'] = result['Error SIMCE']
sheet['F8'] = result['Error var SIMCE']
sheet['F9'] = result['Error mean Test']
sheet['F10'] = result['Error var Test']
sheet['F11'] = result['Error mean Port-Test']
sheet['F12'] = result['Error inter']
sheet['F13'] = result['Error advanced']
sheet['F14'] = result['Error expert']
sheet['F15'] = result['Error SIMCE vs Portfolio']
sheet['F16'] = result['Error SIMCE vs Test']
sheet['F17'] = result['Error Exp vs Portfolio']
sheet['F18'] = result['Error Exp vs Pru']
sheet['F19'] = result['Error adv/exp control']
sheet['F20'] = result['Error Test vs p']
sheet['F21'] = result['Error Portfolio vs p']
sheet['F22'] = result['Error SIMCE vs Experience']




#workbook.close()

ses = np.array([result['Error mean Port'],
result['Error var Portfolio'],
result['Error SIMCE'],
result['Error var SIMCE'],
result['Error mean Test'],
result['Error var Test'],
result['Error mean Port-Test'],
result['Error inter'],
result['Error advanced'],
result['Error expert'],
result['Error SIMCE vs Portfolio'],
result['Error SIMCE vs Test'],
result['Error Exp vs Portfolio'],
result['Error Exp vs Pru'],
result['Error SIMCE vs Experience'],
result['Error adv/exp control'],
result['Error Test vs p'],
result['Error Portfolio vs p']])

means = np.array([result['Mean Portfolio'],
result['Var Portfolio'],
result['Mean SIMCE'],
result['Var SIMCE'],
result['Mean Test'],
result['Var Test'],
result['Mean PortTest'],
result['perc inter'],
result['perc advanced'],
result['perc expert'],
result['Estimation SIMCE vs Portfolio'],
result['Estimation SIMCE vs Prueba'],
result['Estimation EXP vs Portfolio'],
result['Estimation EXP vs Prueba'],
result['Estimation SIMCE vs Experience'],
result['perc adv/exp control'],
result['Estimation Test vs p'],
result['Estimation Portfolio vs p']])

np.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/ses_model.npy',ses)

np.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/moments.npy',means)

np.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/var_cov.npy',varcov)

wb.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/Outcomes.xlsx')


