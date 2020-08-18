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
sys.path.append("")
import utility as util
import parameters as parameters

# DATA 2018

data = pd.read_stata('PythonDataSt.dta')

count_nan = data['zpjeport'].isnull().sum()
print('Count of nan: ' +str(count_nan))
count_nan_1 = data['zpjeprue'].isnull().sum()
print('Count of nan: ' +str(count_nan_1))

treatmentOne = data[['d_trat']]

treatment = data['d_trat'].to_numpy()

p1_0_1 = data[['ptj_portafolio_a2016']]

p1_0 = data['ptj_portafolio_a2016'].to_numpy()

p1 = data['ptj_portafolio_a2016'].to_numpy()

p2_0_1 = data[['ptj_prueba_a2016']]

p2_0 = data['ptj_prueba_a2016'].to_numpy()

p2 = data['ptj_prueba_a2016'].to_numpy()

#p1_0One = data[['ptj_portafolio_rec2018']]

#p1 = data['ptj_portafolio_rec2018'].to_numpy()

#p1_0Two = data[['ptj_portafolio_rec2018']]

#p1_0 = data[['ptj_portafolio_rec2018']].to_numpy()

#p2_0One = data[['ptj_prueba_rec2018']]

#p2 = data['ptj_prueba_rec2018'].to_numpy()

#p2_0Two = data[['ptj_prueba_rec2018']]

#p2_0 = data[['ptj_portafolio_rec2018']].to_numpy()

yearsOne = data[['experience']]

years = data['experience'].to_numpy()

typeSchoolOne = data[['typeschool']]

typeSchool = data['typeschool'].to_numpy()

categPortfolio = data[['cat_portafolio_a2016']]

catPort = data['cat_portafolio_a2016'].to_numpy()

categPrueba = data[['cat_prueba_a2016']]

catPrueba = data['cat_prueba_a2016'].to_numpy()

#recuperar placement inicial de datos (2016)

TrameInitial = data[['tramo_a2016']]

TrameI = data['tramo_a2016'].to_numpy()



N = np.size(p1_0)

HOURS = np.array([44]*N)

alphas = [[0.5,0.5,-0.05,0.05],
		[0.5,0.5,-0.05,0.05]]

#betas = [100,0.9,0.9,-0.05,-0.05,20]

betas = [100,0.9,-0.05,20]

gammas = [-0.1,0.1]

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



mu, sigma = 0, 1 # mean and standard deviation
misj = len(initial_p)
effort = np.random.normal(mu, sigma, misj)

a, b = model.t_test(effort)
print(a)
print(b)

tscores = model.t_test(effort)
print(tscores)

placement = model.placement(tscores)
print(placement)
    
income = model.income(placement)
print(income)

nextT, distancetrame = model.distance(placement)
print(nextT)
print(distancetrame)


h_student = model.student_h(effort)
print(h_student)



