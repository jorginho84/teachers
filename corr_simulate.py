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


#Load data_python
data_python = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/DATA/data_pythonpast_v2023.dta')
data_python['trame'] = data_python['trame'].replace(['INICIAL', 'TEMPRANO', 'AVANZADO', 'EXPERTO I', 'EXPERTO II'], [1,2,3,4,5])
data_python['tramo_a2016'] = data_python['tramo_a2016'].replace(['INICIAL', 'TEMPRANO', 'AVANZADO', 'EXPERTO I', 'EXPERTO II'], [1,2,3,4,5]) 

#### BOOTSTRAP ####

def corr_simulate(data, B):
    n = data.shape[0]
    
    #Full Sample
    est_corrSEXP = np.zeros(B)
    est_corr_EXPPort = np.zeros(B)
    est_corr_EXPPru = np.zeros(B)
    est_corrSPort = np.zeros(B)
    est_corrSPrue = np.zeros(B)

    #Treated sample
    est_mean_SIMCE_treated = np.zeros(B)
    est_var_SIMCE_treated = np.zeros(B)
    est_mean_Port_treated = np.zeros(B)
    est_mean_Pru_treated = np.zeros(B)
    est_var_Port_treated = np.zeros(B)
    est_var_Pru_treated = np.zeros(B)
    est_corrSPast = np.zeros(B)
    est_corrPortp = np.zeros(B)
    est_corrTestp = np.zeros(B)
    est_share_port_treated = np.zeros(B)
    est_share_stei_treated = np.zeros(B)

    #Control sample
    est_mean_SIMCE_control = np.zeros(B)
    est_mean_Port_control = np.zeros(B)
    est_mean_Pru_control = np.zeros(B)
    
    
    
    for i in range(1,B):
        rev = data.sample(n, replace=True)
        
        #A. Full Sample
        p1_past = np.array(rev['score_port_past'])
        p2_past = np.array(rev['score_test_past'])
        p1v1 = np.where(np.isnan(p1_past), 0, p1_past)
        p2v1 = np.where(np.isnan(p2_past), 0, p2_past)
        p0_past = np.zeros(p1_past.shape)
        p0_past = np.where((p1v1 == 0),p2v1, p0_past)
        p0_past = np.where((p2v1 == 0),p1v1, p0_past)
        p0_past = np.where((p1v1 != 0) & (p2v1 != 0) ,(p1_past + p2_past)/2, p0_past)
        dataf_past = {'PORTFOLIO': rev['score_port'], 'TEST': rev['score_test'], 'P_past': p0_past, 'SIMCE': rev['stdsimce'],
                      'EXP': rev['experience'], 'ASIM': rev['tramo_a2016'], 'RECON': rev['trame']}
        datadf_past = pd.DataFrame(dataf_past, columns=['P_past','TEST','PORTFOLIO','SIMCE','EXP','ASIM','RECON'])
                
        corrM = datadf_past.corr()
        #Moments: Corr of experience and SIMCE
        est_corrSEXP[i] = corrM.iloc[3]['EXP']
        
        #Moments: Corr experience and teacher test scores
        est_corr_EXPPort[i] = corrM.iloc[2]['EXP']
        est_corr_EXPPru[i] = corrM.iloc[1]['EXP']
        
        #Moments: Corr teacher test scores and SIMCE
        est_corrSPort[i] = corrM.iloc[2]['SIMCE']
        est_corrSPrue[i] = corrM.iloc[1]['SIMCE']
        
        #B. Treated Sample
        data_treated = datadf_past[rev['d_trat']==1]
        
        #Moments: means and vars of SIMCE, Portfolio and STEI
        est_mean_SIMCE_treated[i] = np.mean(data_treated['SIMCE'])
        est_var_SIMCE_treated[i] = np.var(data_treated['SIMCE'])
        est_mean_Port_treated[i] = np.mean(data_treated['PORTFOLIO'])
        est_mean_Pru_treated[i] = np.mean(data_treated['TEST'])
        est_var_Port_treated[i] = np.var(data_treated['PORTFOLIO'])
        est_var_Pru_treated[i] = np.var(data_treated['TEST'])
        
        corrM = data_treated.corr()
        #Moments: corr of SIMCE, Portfolio and STEU with past test scores
        est_corrSPast[i] = corrM.iloc[3]['P_past']
        est_corrPortp[i] = corrM.iloc[2]['P_past']
        est_corrTestp[i] = corrM.iloc[1]['P_past']

        #Moments: share of teachers with Portfolio > 2.5
        boo_port = data_treated['PORTFOLIO'] >= 2.5
        est_share_port_treated[i] = np.mean(boo_port)
        
        #Moments: share of teachers with STEI > 2.74
        boo_stei = data_treated['TEST'] >= 2.74
        est_share_stei_treated[i] = np.mean(boo_stei)
        
       

        #C. Control Sample
        data_control = datadf_past[rev['d_trat']==0]
        
        #Moments: means and vars of SIMCE, Port and STEI (control groups)
        est_mean_SIMCE_control[i] = np.mean(data_control['SIMCE'])
        est_mean_Port_control[i] = np.mean(data_control['PORTFOLIO'])
        est_mean_Pru_control[i] = np.mean(data_control['TEST'])
  
        
    

    ####MEANS####
    #Full Sample
    corrSEXP = np.mean(est_corrSEXP)
    corr_EXPPort = np.mean(est_corr_EXPPort)
    corr_EXPPru = np.mean(est_corr_EXPPru)
    corrSPort = np.mean(est_corrSPort)
    corrSPrue = np.mean(est_corrSPrue)

    #Treated sample
    mean_SIMCE_treated = np.mean(est_mean_SIMCE_treated)
    var_SIMCE_treated = np.mean(est_var_SIMCE_treated)
    mean_Port_treated = np.mean(est_mean_Port_treated)
    mean_Pru_treated = np.mean(est_mean_Pru_treated)
    var_Port_treated = np.mean(est_var_Port_treated)
    var_Pru_treated = np.mean(est_var_Pru_treated)
    corrSPast = np.mean(est_corrSPast)
    corrPortp = np.mean(est_corrPortp)
    corrTestp = np.mean(est_corrTestp)
    share_port_treated = np.mean(est_share_port_treated)
    share_stei_treated = np.mean(est_share_stei_treated)


    #Control sample
    mean_SIMCE_control = np.mean(est_mean_SIMCE_control)
    mean_Port_control = np.mean(est_mean_Port_control)
    mean_Pru_control = np.mean(est_mean_Pru_control)


    ####VARIANCE####
    #Full Sample
    std_corrSEXP = np.std(est_corrSEXP)
    std_corr_EXPPort = np.std(est_corr_EXPPort)
    std_corr_EXPPru = np.std(est_corr_EXPPru)
    std_corrSPort = np.std(est_corrSPort)
    std_corrSPrue = np.std(est_corrSPrue)

    #Treated sample
    std_mean_SIMCE_treated = np.std(est_mean_SIMCE_treated)
    std_var_SIMCE_treated = np.std(est_var_SIMCE_treated)
    std_mean_Port_treated = np.std(est_mean_Port_treated)
    std_mean_Pru_treated = np.std(est_mean_Pru_treated)
    std_var_Port_treated = np.std(est_var_Port_treated)
    std_var_Pru_treated = np.std(est_var_Pru_treated)
    std_corrSPast = np.std(est_corrSPast)
    std_corrPortp = np.std(est_corrPortp)
    std_corrTestp = np.std(est_corrTestp)
    std_share_port_treated = np.std(est_share_port_treated)
    std_share_stei_treated = np.std(est_share_stei_treated)

    #Control sample
    std_mean_SIMCE_control = np.std(est_mean_SIMCE_control)
    std_mean_Port_control = np.std(est_mean_Port_control)
    std_mean_Pru_control = np.std(est_mean_Pru_control)

    
    #var-cov matrix
    samples = np.array([est_corrSEXP,est_corr_EXPPort,est_corr_EXPPru,est_corrSPort,est_corrSPrue,
        est_mean_SIMCE_treated,est_var_SIMCE_treated,est_mean_Port_treated,est_mean_Pru_treated,
        est_var_Port_treated,est_var_Pru_treated,est_corrSPast,est_corrPortp,est_corrTestp,est_share_port_treated,
        est_mean_SIMCE_control,est_mean_Port_control,est_mean_Pru_control])
    
    varcov = np.cov(samples)


    return {'Corr Simce and experience': corrSEXP,
            'Corr Portfolio and experience': corr_EXPPort,
            'Corr STEI and experience': corr_EXPPru,
            'Corr SIMCE and Portfolio': corrSPort,
            'Corr SIMCE and STEI': corrSPrue,

            'SIMCE Mean (treated)': mean_SIMCE_treated,
            'SIMCE Var (treated)': var_SIMCE_treated,
            'Portfolio Mean (treated)': mean_Port_treated,
            'STEI Mean (treated)': mean_Pru_treated,
            'Portfolio Var (treated)': var_Port_treated,
            'STEI Var (treated)': var_Pru_treated,
            'Corr Simce Past': corrSPast,
            'Corr Portfolio Past': corrPortp,
            'Corr STEI Past': corrTestp,
            'Share Portfolio > 2.5 (treated)': share_port_treated,
            'Share STEI > 2.74 (treated)': share_stei_treated,

            'SIMCE Mean (control)': mean_SIMCE_control,
            'Portfolio Mean (control)': mean_Port_control,
            'STEI Mean (control)': mean_Pru_control,
                        
            'S.E. Corr Simce and experience': std_corrSEXP,
            'S.E. Corr Portfolio and experience': std_corr_EXPPort,
            'S.E. Corr STEI and experience': std_corr_EXPPru,
            'S.E. Corr SIMCE and Portfolio': std_corrSPort,
            'S.E. Corr SIMCE and STEI': std_corrSPrue,

            'S.E. SIMCE Mean (treated)': std_mean_SIMCE_treated,
            'S.E. SIMCE Var (treated)': std_var_SIMCE_treated,
            'S.E. Portfolio Mean (treated)': std_mean_Port_treated,
            'S.E. STEI Mean (treated)': std_mean_Pru_treated,
            'S.E. Portfolio Var (treated)': std_var_Port_treated,
            'S.E. STEI Var (treated)': std_var_Pru_treated,
            'S.E. Corr Simce Past': std_corrSPast,
            'S.E. Corr Portfolio Past': std_corrPortp,
            'S.E. Corr STEI Past': std_corrTestp,
            'S.E. Share Portfolio > 2.5 (treated)': std_share_port_treated,
            'S.E. Share STEI > 2.74 (treated)': std_share_stei_treated,

            'S.E. SIMCE Mean (control)': std_mean_SIMCE_control,
            'S.E. Portfolio Mean (control)': std_mean_Port_control,
            'S.E. STEI Mean (control)': std_mean_Pru_control,
            'Var Cov Matrix': varcov}


result = corr_simulate(data_python,1000)
#print(result)

varcov = result['Var Cov Matrix']


means = np.array([result['Corr Simce and experience'],
            result['Corr Portfolio and experience'],
            result['Corr STEI and experience'],
            result['Corr SIMCE and Portfolio'],
            result['Corr SIMCE and STEI'],
            result['SIMCE Mean (treated)'],
            result['SIMCE Var (treated)'],
            result['Portfolio Mean (treated)'],
            result['STEI Mean (treated)'],
            result['Portfolio Var (treated)'],
            result['STEI Var (treated)'],
            result['Corr Simce Past'],
            result['Corr Portfolio Past'],
            result['Corr STEI Past'],
            result['Share Portfolio > 2.5 (treated)'],
            result['Share STEI > 2.74 (treated)'],
            result['SIMCE Mean (control)'],
            result['Portfolio Mean (control)'],
            result['STEI Mean (control)']])

ses = np.array([result['S.E. Corr Simce and experience'],
            result['S.E. Corr Portfolio and experience'],
            result['S.E. Corr STEI and experience'],
            result['S.E. Corr SIMCE and Portfolio'],
            result['S.E. Corr SIMCE and STEI'],
            result['S.E. SIMCE Mean (treated)'],
            result['S.E. SIMCE Var (treated)'],
            result['S.E. Portfolio Mean (treated)'],
            result['S.E. STEI Mean (treated)'],
            result['S.E. Portfolio Var (treated)'],
            result['S.E. STEI Var (treated)'],
            result['S.E. Corr Simce Past'],
            result['S.E. Corr Portfolio Past'],
            result['S.E. Corr STEI Past'],
            result['S.E. Share Portfolio > 2.5 (treated)'],
            result['S.E. Share STEI > 2.74 (treated)'],
            result['S.E. SIMCE Mean (control)'],
            result['S.E. Portfolio Mean (control)'],
            result['S.E. STEI Mean (control)']])

np.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/ses_model_new.npy',ses)
#np.save('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/ses_model_v2023.npy',ses)

np.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/moments_new.npy',means)
#np.save('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/moments_v2023.npy',means)

np.save('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/estimates/var_cov_new.npy',varcov)
#np.save('C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\Local_teacher_julio13/var_cov_v2023.npy',varcov)



