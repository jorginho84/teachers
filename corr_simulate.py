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
from tabulate import tabulate
from texttable import Texttable
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook


#### LOAD DATA ####

df = pd.read_stata('D:\Git\TeacherPrincipal\data_python.dta')

pd.value_counts(df['trame'])

df['trame'] = df['trame'].replace(['INICIAL', 'TEMPRANO', 'AVANZADO', 'EXPERTO I', 'EXPERTO II'], [1,2,3,4,5]) 
print(df['trame'])


#### BOOTSTRAP ####

def corr_simulate(data, B):
    n = data.shape[0]
    est_corrSPort = np.zeros(B)
    est_corrSPrue = np.zeros(B)
    est_corrPP = np.zeros(B)
    est_corr_EXPPort = np.zeros(B)
    est_corr_EXPPru = np.zeros(B)
    est_mean_Port = np.zeros(B)
    est_mean_Pru = np.zeros(B)
    est_var_Port = np.zeros(B)
    est_var_Pru = np.zeros(B)
    perc_init =  np.zeros(B)
    perc_inter =  np.zeros(B)
    perc_advan =  np.zeros(B)
    perc_expert =  np.zeros(B)
    est_mean_SIMCE = np.zeros(B)
    
    for i in range(1,B):
        rev = data.sample(n, replace=True)
        datav = rev[rev['d_trat']==1]
        est_mean_Port[i] = np.mean(datav['score_port'])
        est_var_Port[i] = np.var(datav['score_port'])
        est_mean_Pru[i] = np.mean(datav['score_test'])
        est_var_Pru[i] = np.var(datav['score_test'])
        perc_init[i] = (sum(datav['trame']==1) / len(datav['trame'])) * 100
        perc_inter[i] = (sum(datav['trame']==2) / len(datav['trame'])) * 100
        perc_advan[i] = (sum(datav['trame']==3) / len(datav['trame'])) * 100
        perc_expert[i] = ((sum(datav['trame']==4)+sum(datav['trame']==5)) / len(datav['trame'])) * 100
        datav1 = {'SIMCE': datav['stdsimce_m'], 'PORTFOLIO': datav['zpjeport'], 'TEST': datav['zpjeprue'], 'EXP': datav['experience']}
        datadf = pd.DataFrame(datav1, columns=['SIMCE','PORTFOLIO','TEST', 'EXP'])
        corrM = datadf.corr()
        est_corrSPort[i] = corrM.iloc[0]['PORTFOLIO']
        est_corrSPrue[i] = corrM.iloc[0]['TEST']
        est_corrPP[i] = corrM.iloc[1]['TEST']
        est_corr_EXPPort[i] = corrM.iloc[3]['PORTFOLIO']
        est_corr_EXPPru[i] = corrM.iloc[3]['TEST']
        datav0 = rev[rev['d_trat']==0]
        est_mean_SIMCE[i] = np.mean(datav0['stdsimce_m'].to_numpy())
        
        
    est_sim_SPort = np.mean(est_corrSPort)
    est_sim_Prue = np.mean(est_corrSPrue)
    est_sim_PP = np.mean(est_corrPP)
    est_sim_EXPPort = np.mean(est_corr_EXPPort)
    est_sim_EXPPru = np.mean(est_corr_EXPPru)
    est_sim_mean_Port = np.mean(est_mean_Port)
    est_sim_var_Port = np.mean(est_var_Port)
    est_sim_mean_Pru = np.mean(est_mean_Pru)
    est_sim_var_Test = np.mean(est_var_Pru)
    est_sim_perc_init = np.mean(perc_init)
    est_sim_perc_inter = np.mean(perc_inter)
    est_sim_perc_advan = np.mean(perc_advan)
    est_sim_perc_expert = np.mean(perc_expert)
    est_sim_mean_SIMCE = np.mean(est_mean_SIMCE)
    
    error_SPort = np.std(est_corrSPort)
    error_SPru = np.std(est_corrSPrue)
    error_PP = np.std(est_corrPP)
    error_EXPPort = np.std(est_corr_EXPPort)
    error_EXPPru = np.std(est_corr_EXPPru)
    error_mean_Port = np.std(est_mean_Port)
    error_var_Port = np.std(est_var_Port)
    error_mean_Pru = np.std(est_mean_Pru)
    error_var_Pru = np.std(est_var_Pru)
    error_init = np.std(perc_init)
    error_inter = np.std(perc_inter)
    error_advan = np.std(perc_advan)
    error_expert = np.std(perc_expert)
    error_mean_SIMCE = np.std(est_mean_SIMCE)
    
    
    plt.hist(est_corrSPort, bins=100)
    plt.axvline(error_SPort, color='r', linestyle='dashed', linewidth=1)
    plt.title("Histogram Portfolio")
    plt.hist(est_corrSPrue, bins=100)
    plt.axvline(error_SPru, color='r', linestyle='dashed', linewidth=1)
    plt.title("Histogram Test")
    #sn.heatmap(corrMatrix, annot=True)
    plt.show()
    return {'Estimation SIMCE vs Portfolio': est_sim_SPort,
            'Estimation SIMCE vs Prueba': est_sim_Prue,
            'Estimation Portfolio vs Prueba': est_sim_PP,
            'Estimation EXP vs Portfolio': est_sim_EXPPort,
            'Estimation EXP vs Prueba': est_sim_EXPPru,
            'Mean Portfolio': est_sim_mean_Port,
            'Var Portfolio': est_sim_var_Port,
            'Mean Test': est_sim_mean_Pru,
            'Var Test': est_sim_var_Test,
            'perc init': est_sim_perc_init,
            'perc inter': est_sim_perc_inter,
            'perc advanced': est_sim_perc_advan,
            'perc expert': est_sim_perc_expert,
            'Mean SIMCE': est_sim_mean_SIMCE,
                'Error SIMCE vs Portfolio': error_SPort,
                'Error SIMCE vs Test': error_SPru,
                'Error Portfolio vs Test': error_PP,
                'Error Exp vs Portfolio': error_EXPPort,
                'Error Exp vs Pru': error_EXPPru,
                'Error mean Port': error_mean_Port,
                'Error var Portfolio': error_var_Port,
                'Error mean Test': error_mean_Pru,
                'Error var Test': error_var_Pru,
                'Error init': error_init,
                'Error inter': error_inter,
                'Error advanced': error_advan,
                'Error expert': error_expert,
                'Error SIMCE': error_mean_SIMCE}


result = corr_simulate(df,100)
print(result)
#result_1 = result['Estimation SIMCE vs Portfolio']
#result_2 = result['Estimation SIMCE vs Prueba']
#result_3 = result['Error SIMCE vs Portfolio']
#result_4 = result['Error SIMCE vs Test']

#### INITIAL CORRELATION #### 

j = df['stdsimce_m']
j_1 = df['zpjeport']
j_2 = df['zpjeprue']
j_3 = df['experience']
data = {'SIMCE': j, 'PORTFOLIO': j_1, 'TEST': j_2, 'EXP': j_3}
datadf2 = pd.DataFrame(data, columns=['SIMCE','PORTFOLIO','TEST', 'EXP'])
corrM = datadf2.corr()
print(corrM)
corr_port_ini = corrM.iloc[0]['PORTFOLIO']
corr_test_ini = corrM.iloc[0]['TEST']
print(corr_port_ini)
sn.heatmap(corrM, annot=True)
plt.show()



##### PYTHON TO EXCEL #####

#workbook = xlsxwriter.Workbook('D:\Git\TeacherPrincipal\OutcomesData.xlsx')
#worksheet = workbook.add_worksheet()

wb = load_workbook('D:\Git\TeacherPrincipal\Outcomes.xlsx')

#sheet('C5', 'corr(Port,Simce)')
#sheet('C6', 'corr(Pru,Simce)')
#sheet('C7', 'corr(Port,Pru)')
#sheet('C8', '\ alpha_0 E[Port]')
#worksheet.write('C9', '\ alpha_0 E[Pru]')
#worksheet.write('C10', '\ alpha_3 corr(exp,Port)')
#worksheet.write('C11', '\ alpha_3 corr(exp,Pru)')
#worksheet.write('C12', '\sigma_1 Var(Port)')
#worksheet.write('C13', '\sigma_1 Var(Pru)')
#worksheet.write('C14', '\% Initial')
#worksheet.write('C15', '\% Intermediate')
#worksheet.write('C16', '\% Advanced')
#worksheet.write('C17', '\% Expert')
#worksheet.write('D4', 'simulation')
#worksheet.write('E4', 'data')
#worksheet.write('F4', 'se')

#book = Workbook()
sheet = wb.active


sheet['E5'] = result['Estimation SIMCE vs Portfolio']
sheet['E6'] = result['Estimation SIMCE vs Prueba']
sheet['E7'] = result['Estimation Portfolio vs Prueba']
sheet['E8'] = result['Mean Portfolio']
sheet['E9'] = result['Mean Test']
sheet['E10'] = result['Estimation EXP vs Portfolio']
sheet['E11'] = result['Estimation EXP vs Prueba']
sheet['E12'] = result['Var Portfolio']
sheet['E13'] = result['Var Test']
sheet['E14'] = result['perc init']
sheet['E15'] = result['perc inter']
sheet['E16'] = result['perc advanced']
sheet['E17'] = result['perc expert']
sheet['E18'] = result['Mean SIMCE']


sheet['F5'] = result['Error SIMCE vs Portfolio']
sheet['F6'] = result['Error SIMCE vs Test']
sheet['F7'] = result['Error Portfolio vs Test']
sheet['F8'] = result['Error mean Port']
sheet['F9'] = result['Error mean Test']
sheet['F10'] = result['Error Exp vs Portfolio']
sheet['F11'] = result['Error Exp vs Pru']
sheet['F12'] = result['Error var Portfolio']
sheet['F13'] = result['Error var Test']
sheet['F14'] = result['Error init']
sheet['F15'] = result['Error inter']
sheet['F16'] = result['Error advanced']
sheet['F17'] = result['Error expert']
sheet['F18'] = result['Error SIMCE']


#workbook.close()


wb.save('D:\Git\TeacherPrincipal\Outcomes.xlsx')


"""
#### TABLA LATEX ####

rows = [[ r'$\sigma_SP$', result_1 , corr_port_ini  , result_3],
        [r'$\sigma_ST$', result_2, corr_test_ini, result_4]]

table = Texttable()
table.set_cols_align(["c"] * 4)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('Tabulate Table:')
headers = ['','simdata', 'data', 's.e data']
print(tabulate(rows, headers))

print('\nTexttable Table:')
print(table.draw())

print('\nTabulate Latex:')
print(tabulate(rows, headers, tablefmt='latex'))
"""

