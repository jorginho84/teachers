# -*- coding: utf-8 -*-
"""
Simulates data

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
sys.path.append("D:\Git\TeacherPrincipal")
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
import between
import random
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
import time

# DATA 2018


data = pd.read_stata('D:\Git\TeacherPrincipal\data_python.dta')



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

alphas = [[0.5,0.1,0.2,-0.01],
		[0.5,0.1,0.2,-0.01]]

#betas = [100,0.9,0.9,-0.05,-0.05,20]
#Parámetros más importantes
#betas = [100,10,33,20]

betas = [-0.4,0.3,0.9,1]

gammas = [-0.1,-0.2,0.8]

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

jashdkjhsa = opt['Opt Teacher'][0]



datah = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1], 'Treatment': opt['Treatment']}
datadfh = pd.DataFrame(datah, columns=['SIMCE','PORTFOLIO','TEST', 'Treatment'])
datahgf = datadfh[datadfh['Treatment']==0]

print(np.mean(datahgf['SIMCE'].to_numpy()))


jdhd = np.mean(datahgf['Treatment'].to_numpy())
print(datahgf['Treatment'])

tratatat = datadfh['Treatment'].to_numpy()
tratatat = sum(datadfh['Treatment']==1)
print(tratatat)

datav2 = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1]}
datadf = pd.DataFrame(datahgf, columns=['SIMCE','PORTFOLIO','TEST'])
corrM = datadf.corr()
print(corrM)


wb = load_workbook('D:\Git\TeacherPrincipal\Outcomes.xlsx')
sheet = wb.active

sheet['C2'] = 'Hello*2'

wb.save('D:\Git\TeacherPrincipal\Outcomes.xlsx')

between.betweenOne()


print('Estimate correlation')

modelestimate = est.estimate(N,modelSD,years,treatment) 

corr_data = modelestimate.simulation(50)
print(corr_data)


##### PYTHON TO EXCEL #####

#workbook = xlsxwriter.Workbook('D:\Git\TeacherPrincipal\Outcomes.xlsx')
#worksheet = workbook.add_worksheet()

#book = Workbook()
#sheet = book.active

wb = load_workbook('D:\Git\TeacherPrincipal\Outcomes.xlsx')
sheet = wb.active

sheet['C5'] = 'corr(Port,Simce)'
sheet['C6'] = 'corr(Pru,Simce)'
sheet['C7'] = 'corr(Port,Pru)'
sheet['C8'] = '\ alpha_0 E[Port]'
sheet['C9'] = '\ alpha_0 E[Pru]'
sheet['C10'] = '\ alpha_3 corr(exp,Port)'
sheet['C11'] = '\ alpha_3 corr(exp,Pru)'
sheet['C12'] = '\sigma_1 Var(Port)'
sheet['C13'] = '\sigma_1 Var(Pru)'
sheet['C14'] = '\% Initial'
sheet['C15'] = '\% Intermediate'
sheet['C16'] = '\% Advanced'
sheet['C17'] = '\% Expert'
sheet['C18'] = '\ alpha_1 E[SIMCE]'
sheet['D4'] = 'simulation'
sheet['E4'] = 'data'
sheet['F4'] = 'se'

sheet['D5'] = corr_data['Estimation SIMCE vs Portfolio']
sheet['D6'] = corr_data['Estimation SIMCE vs Prueba']
sheet['D7'] = corr_data['Estimation Portfolio vs Prueba']
sheet['D8'] = corr_data['Mean Portfolio']
sheet['D9'] = corr_data['Mean Test']
sheet['D10'] = corr_data['Estimation EXP vs Portfolio']
sheet['D11'] = corr_data['Estimation EXP vs Prueba']
sheet['D12'] = corr_data['Var Port']
sheet['D13'] = corr_data['Var Test']
sheet['D14'] = corr_data['perc init']
sheet['D15'] = corr_data['perc inter']
sheet['D16'] = corr_data['perc advanced']
sheet['D17'] = corr_data['perc expert']
sheet['D18'] = corr_data['Mean SIMCE']






wb.save('D:\Git\TeacherPrincipal\Outcomes.xlsx')



"""
worksheet.write('C5', 'corr(Port,Simce)')
worksheet.write('C6', 'corr(Pru,Simce)')
worksheet.write('C7', 'corr(Port,Pru)')
worksheet.write('C8', '\ alpha_0 E[Port]')
worksheet.write('C9', '\ alpha_0 E[Pru]')
worksheet.write('C10', '\ alpha_3 corr(exp,Port)')
worksheet.write('C11', '\ alpha_3 corr(exp,Pru)')
worksheet.write('C12', '\sigma_1 Var(Port)')
worksheet.write('C13', '\sigma_1 Var(Pru)')
worksheet.write('C14', '\% Initial')
worksheet.write('C15', '\% Intermediate')
worksheet.write('C16', '\% Advanced')
worksheet.write('C17', '\% Expert')
worksheet.write('D4', 'simulation')
worksheet.write('E4', 'data')
worksheet.write('F4', 'se')



worksheet.write('D5', corr_data['Estimation SIMCE vs Portfolio'])
worksheet.write('D6', corr_data['Estimation SIMCE vs Prueba'])
worksheet.write('D7', corr_data['Estimation Portfolio vs Prueba'])
worksheet.write('D8', corr_data['Mean Portfolio'])
worksheet.write('D9', corr_data['Mean Test'])
worksheet.write('D12', corr_data['Var Port'])
worksheet.write('D13', corr_data['Var Test'])
worksheet.write('D14', corr_data['perc init'])
worksheet.write('D15', corr_data['perc inter'])
worksheet.write('D16', corr_data['perc advanced'])
worksheet.write('D17', corr_data['perc expert'])





workbook.close()
"""

"""
'perc init': est_sim_perc_init,
            'perc inter': est_sim_perc_inter,
            'perc advanced': est_sim_perc_advan,
            'perc expert': est_sim_perc_expert,



#con esfuerzo optimo, puedes simular simce, test scores, placement.

pd.value_counts(opt['Opt Placement'])

lkj = sum(opt['Opt Placement']==1)
print(lkj)

perc_init = (sum(opt['Opt Placement']==1) / len(opt['Opt Placement'])) * 100
print(perc_init)

len(opt['Opt Placement']==1)

perc_init = (opt['Opt Placement'].sum()/(len(opt['Opt Placement']) - opt['Opt Placement'].isnull().sum()))*100
#data_1982['perct'] = (data_1982[data_1982['TELEPHONE']==1]/data_1982['TELEPHONE'].sum())*100

print('\nTelephone Percentaje\n', perc_init,'%')
"""


