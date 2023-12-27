# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:37:16 2023

@author: Patricio De Araya
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
#sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
sys.path.append("C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy")
#import gridemax
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd
import simdata_c as sdc
import estimate as est
from utility_counterfactual_att import Count_att_2
from utility_counterfactual_att_categories import Count_att_2_cat
from utility_counterfactual_att_pfp import Count_att_2_pfp
#import pybobyqa
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
from scipy import interpolate
import time
import openpyxl

# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
import linearmodels as lm
from linearmodels.panel import PanelOLS


np.random.seed(123)


betas_nelder  = np.load(r"C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/estimates/betasopt_model_v54.npy")

#Only treated teachers
#data_1 = pd.read_stata('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/DATA/data_pythonpast_v2023.dta')
data_1 = pd.read_stata(r'C:/Users\Patricio De Araya\Dropbox\LocalRA\LocalTeacher\teacher_dec23_copy/data_pythonpast_v2023.dta')
data = data_1[data_1['d_trat']==1]
N = np.array(data['experience']).shape[0]

n_sim = 500


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
TrameI = np.array(data['tramo_a2016'])
# TYPE SCHOOL #
typeSchool = np.array(data['typeschool'])

# Priority #
priotity = np.array(data['por_priority'])

priotity_aep = np.array(data['priority_aep'])

rural_rbd = np.array(data['rural_rbd'])

locality = np.array(data['AsignacionZona'])


#### PARAMETERS MODEL ####
N = np.size(p1_0)
HOURS = np.array([44]*N)

alphas = [[0, betas_nelder[0],0,betas_nelder[1],
             betas_nelder[2], betas_nelder[3]],
            [0, 0,betas_nelder[4],betas_nelder[5],
            betas_nelder[6], betas_nelder[7]]]
            
betas = [betas_nelder[8], betas_nelder[9], betas_nelder[10],betas_nelder[11],betas_nelder[12],betas_nelder[13]]
gammas = [betas_nelder[14],betas_nelder[15],betas_nelder[16]]

dolar= 600
value = [14403, 15155]
hw = [value[0]/dolar,value[1]/dolar]
porc = [0.0338, 0.0333]

#inflation adjustemtn: 2012Jan-2020Jan: 1.111
Asig = [50000*1.111, 100000*1.111, 150000*1.111]
AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar] 

# *** This is withouth teaching career ***
# * value professional qualification (pesos)= 72100 *
# * value professional mention (pesos)= 24034 *
# *** This is with teaching career ***
# * value professional qualification (pesos)= 253076 *
# * value professional mention (pesos)= 84360 *

#inflation adjustemtn: 2012Jan-2020Jan: 1.111***
qualiPesos = [72100*1.111, 24034*1.111, 253076, 84360] 
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

pri = [48542,66609,115151]
priori = [pri[0]/dolar, pri[1]/dolar, pri[2]/dolar]

treatment = np.ones(N)*1

initial_p = np.zeros(p1.shape[0])

initial_p[(TrameI=='INICIAL')] = 1
initial_p[(TrameI=='TEMPRANO')] = 2
initial_p[(TrameI=='AVANZADO')] = 3
initial_p[(TrameI=='EXPERTO I')] = 4
initial_p[(TrameI=='EXPERTO II')] = 5


HvalueE = hw[0]
HvalueS = hw[1]
    
initial_p_2 = initial_p.copy()
#initial_p_aep = initial_p[1].copy()
   
RBMNElemt2 = np.zeros(initial_p_2.shape[0])
RBMNSecond2 = np.zeros(initial_p_2.shape[0])
ExpTrameE2 = np.zeros(initial_p_2.shape[0])
ExpTrameS2 = np.zeros(initial_p_2.shape[0])
BRP2 = np.zeros(initial_p_2.shape[0])
BRPWithout2 = np.zeros(initial_p_2.shape[0])
ATDPinitial2 = np.zeros(initial_p_2.shape[0])
ATDPearly2 = np.zeros(initial_p_2.shape[0])
ATDPadvanced2 = np.zeros(initial_p_2.shape[0])
ATDPadvancedfixed2 = np.zeros(initial_p_2.shape[0])
ATDPexpert12 = np.zeros(initial_p_2.shape[0])
ATDPexpert1fixed2 = np.zeros(initial_p_2.shape[0])
ATDPexpert22 = np.zeros(initial_p_2.shape[0])
ATDPexpert2fixed2 = np.zeros(initial_p_2.shape[0])
AsigElemt2 = np.zeros(initial_p_2.shape[0])
AsigSecond2 = np.zeros(initial_p_2.shape[0])
salary2d = np.zeros(initial_p_2.shape[0])
salary3d = np.zeros(initial_p_2.shape[0])
prioirtyap11 = np.zeros(initial_p_2.shape[0])
prioirtyap22 = np.zeros(initial_p_2.shape[0])
prioirtyap33 = np.zeros(initial_p_2.shape[0])
prioirtyap44 = np.zeros(initial_p_2.shape[0])
prioirtyap55 = np.zeros(initial_p_2.shape[0])
prioirtyap66 = np.zeros(initial_p_2.shape[0])
prioirtyap661 = np.zeros(initial_p_2.shape[0])
prioirtyap771 = np.zeros(initial_p_2.shape[0])
prioirtyap77 = np.zeros(initial_p_2.shape[0])
prioirtyap88 = np.zeros(initial_p_2.shape[0])
prioirtyap991 = np.zeros(initial_p_2.shape[0])
prioirtyap992 = np.zeros(initial_p_2.shape[0])
prioirtyap993 = np.zeros(initial_p_2.shape[0])
prioirtyap994 = np.zeros(initial_p_2.shape[0])
prioirtyap995 = np.zeros(initial_p_2.shape[0])
localAssig2 = np.zeros(initial_p_2.shape[0])
localAssig3 = np.zeros(initial_p_2.shape[0])
prioirtyap_aep11 = np.zeros(initial_p_2.shape[0])
prioirtyap_aep22 = np.zeros(initial_p_2.shape[0])
prioirtyap_aep33 = np.zeros(initial_p_2.shape[0])



# " RENTA BASE MÍNIMA NACIONAL per year"

RBMNElemt = np.where((typeSchool==1),HvalueE*HOURS, RBMNElemt2)
RBMNSecond = np.where((typeSchool==0),HvalueS*HOURS, RBMNSecond2)

        
# " EXPERIENCE (4 years)"
bienniumtwoFalse = years/2
biennium = np.floor(bienniumtwoFalse)
biennium[biennium>15]=15

porc1 = porc[0] 
porc2 = porc[1]

ExpTrameE = np.where((typeSchool==1) & (years > 2), (porc1 + (porc2 * (biennium - 1))) * RBMNElemt, ExpTrameE2)
ExpTrameS = np.where((typeSchool==0) & (years > 2), (porc1 + (porc2 * (biennium - 1))) * RBMNSecond, ExpTrameS2)



# " VOCATIONAL RECOGNITION BONUS (BRP) (4 years) "
# We assume that every teacher have
# degree and mention.

professional_qualificationW = pro[0] 
professional_mentionW = pro[1]
professional_qualification = pro[2]
professional_mention = pro[3]
full_contract = 44

BRP = np.where((typeSchool==1) | (typeSchool==0),(professional_qualification + professional_mention)*(HOURS/full_contract),BRP2)
BRPWithout = np.where((typeSchool==1) | (typeSchool==0),(professional_qualificationW + professional_mentionW)*(HOURS/full_contract),BRPWithout2)


# " PROGRESSION COMPONENT BY TRANCHE "
# " COMPONENTE DE FIJO TRAMO "

Proinitial = pol[0] 
ProEarly = pol[1] 
Proadvanced = pol[2] 
Proadvancedfixed = pol[3] 
Proexpert1 = pol[4] 
Proexpert1fixed = pol[5] 
Proexpert2 = pol[6] 
Proexpert2fixed = pol[7]

ATDPinitial = (Proinitial/15)*(HOURS/full_contract)*biennium
ATDPearly = (ProEarly/15)*(HOURS/full_contract)*biennium
ATDPadvanced = (Proadvanced/15)*(HOURS/full_contract)*biennium
ATDPadvancedfixed = (Proadvancedfixed)*(HOURS/full_contract)
ATDPexpert1 = (Proexpert1/15)*(HOURS/44)*biennium
ATDPexpert1fixed = (Proexpert1fixed)*(HOURS/full_contract)
ATDPexpert2 = (Proexpert2/15)*(HOURS/full_contract)*biennium
ATDPexpert2fixed = (Proexpert2fixed)*(HOURS/full_contract)



# " AEP (Teaching excellence) (4 years)

AcreditaTramoI = AEP[0]
AcreditaTramoII = AEP[1]
AcreditaTramoIII = AEP[2]

    
# " Asignación de perfeccionamiento
# " This is the new asignment
AsigElemt = np.where(((typeSchool==1)), RBMNElemt*0.4*(biennium/15), AsigElemt2)
AsigSecond = np.where((typeSchool==0), RBMNSecond*0.4*(biennium/15), AsigSecond2)


# " Asignación por docencia
# "Post-reform Priority allocation

        # "Post-reform Priority allocation 

prioirtyap1 = np.where(((priotity >= 0.6) & (priotity < 0.8)) & (initial_p_2==1), ((ATDPinitial+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[0]), prioirtyap11)
prioirtyap2 = np.where(((priotity >= 0.6) & (priotity < 0.8)) & (initial_p_2==2), (((ATDPearly+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[0])), prioirtyap22)
prioirtyap3 = np.where(((priotity >= 0.6) & (priotity < 0.8)) & (initial_p_2==3), ((ATDPadvanced+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[0]), prioirtyap33)
prioirtyap4 = np.where(((priotity >= 0.6) & (priotity < 0.8)) & (initial_p_2==4), ((ATDPexpert1+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[0]), prioirtyap44)
prioirtyap5 = np.where(((priotity >= 0.6) & (priotity < 0.8)) & (initial_p_2==5), ((ATDPexpert2+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[0]), prioirtyap55)
prioirtyap51 = np.where((priotity >= 0.8) & (initial_p_2==1), ((ATDPinitial+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[1]), prioirtyap661)
prioirtyap52 = np.where((priotity >= 0.8) & (initial_p_2==2), ((ATDPearly+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[1]), prioirtyap771)
prioirtyap6 = np.where((priotity >= 0.8) & (initial_p_2==3), ((ATDPadvanced+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[2]), prioirtyap66)
prioirtyap7 = np.where((priotity >= 0.8) & (initial_p_2==4), ((ATDPexpert1+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[2]), prioirtyap77)
prioirtyap8 = np.where((priotity >= 0.8) & (initial_p_2==5), ((ATDPexpert2+ExpTrameE)*0.2)+((HOURS/full_contract)*priori[2]), prioirtyap88)
prioirtyap91 = np.where(((priotity >= 0.45) & (priotity < 0.6))  & (rural_rbd==1) & (initial_p_2==1), (ATDPinitial*0.1), prioirtyap991)
prioirtyap92 = np.where(((priotity >= 0.45) & (priotity < 0.6))  & (rural_rbd==1) & (initial_p_2==2), (ATDPearly*0.1), prioirtyap992)
prioirtyap93 = np.where(((priotity >= 0.45) & (priotity < 0.6))  & (rural_rbd==1) & (initial_p_2==3), (ATDPadvanced*0.1), prioirtyap993)
prioirtyap94 = np.where(((priotity >= 0.45) & (priotity < 0.6))  & (rural_rbd==1) & (initial_p_2==4), (ATDPexpert1*0.1), prioirtyap994)
prioirtyap95 = np.where(((priotity >= 0.45) & (priotity < 0.6))  & (rural_rbd==1) & (initial_p_2==5),(ATDPexpert2*0.1), prioirtyap995)
prioirtyap = sum([prioirtyap1,prioirtyap2,prioirtyap3,prioirtyap4,prioirtyap5,prioirtyap51,prioirtyap52\
                  ,prioirtyap6,prioirtyap7,prioirtyap8,prioirtyap91,prioirtyap92,prioirtyap93,prioirtyap94,prioirtyap95])

# "Pre-reform Priority allocation

# "Pre-reform Priority allocation (4 years)

#prioirtyap_aep1 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==7), (AcreditaTramoI*0.4), prioirtyap_aep11)
#prioirtyap_aep2 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==8), (AcreditaTramoII*0.4), prioirtyap_aep22)
#prioirtyap_aep3 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==9), (AcreditaTramoIII*0.4), prioirtyap_aep33)
#priorityaep = sum([prioirtyap_aep1,prioirtyap_aep2,prioirtyap_aep3])


# " Locality assignation

localAssig_1 = np.where((typeSchool == 1), (locality*RBMNElemt/100), localAssig2)
localAssig_0 = np.where((typeSchool == 0), (locality*RBMNSecond/100), localAssig3)


    
# " SUM OF TOTAL SALARY "


salary1 = np.where((initial_p_2==1) & (treatment == 1) & (typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPinitial,prioirtyap,localAssig_1]), salary2d)
salary3 = np.where((initial_p_2==1) & (treatment == 1) & (typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ATDPinitial,prioirtyap,localAssig_0]), salary2d)

salary5 = np.where((initial_p_2==2) & (treatment == 1) & (typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPearly,prioirtyap,localAssig_1]), salary2d)
salary7 = np.where((initial_p_2==2) & (treatment == 1) & (typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPearly,prioirtyap,localAssig_0]), salary2d)


salary9 = np.where((initial_p_2==3) & (treatment == 1) & (typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_1]), salary2d)
salary11 = np.where((initial_p_2==3) & (treatment == 1) & (typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_0]), salary2d)

salary13 = np.where((initial_p_2==4) & (treatment == 1) & (typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_1]), salary2d)
salary15 = np.where((initial_p_2==4) & (treatment == 1) & (typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_0]), salary2d)

salary17 = np.where((initial_p_2==5) & (treatment == 1) & (typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_1]), salary2d)
salary19 = np.where((initial_p_2==5) & (treatment == 1) & (typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_0]), salary2d)

#Control: following initial placement


initial_p = self.initial()

salary21 = np.where((initial_p==1) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPinitial,prioirtyap,localAssig_1]), salary2d)
salary22 = np.where((initial_p==1) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ATDPinitial,prioirtyap,localAssig_0]), salary2d)

salary23 = np.where((initial_p==2) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPearly,prioirtyap,localAssig_1]), salary2d)
salary24 = np.where((initial_p==2) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPearly,prioirtyap,localAssig_0]), salary2d)

salary25 = np.where((initial_p==3) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_1]), salary2d)
salary26 = np.where((initial_p==3) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_0]), salary2d)

salary27 = np.where((initial_p==4) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_1]), salary2d)
salary28 = np.where((initial_p==4) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_0]), salary2d)

salary29 = np.where((initial_p==5) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_1]), salary2d)
salary30 = np.where((initial_p==5) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_0]), salary2d)


"""
#pre-reform
salary21 = np.where((initial_p_aep==6) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt]), salary3d)
salary22 = np.where((initial_p_aep==6) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond]), salary3d)

salary23 = np.where((initial_p_aep==7) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,priorityaep,AcreditaTramoI]), salary3d)
salary24 = np.where((initial_p_aep==7) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,priorityaep,AcreditaTramoI]), salary3d)

salary25 = np.where((initial_p_aep==8) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,priorityaep,AcreditaTramoII]), salary3d)
salary26 = np.where((initial_p_aep==8) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,priorityaep,AcreditaTramoII]), salary3d)

salary27 = np.where((initial_p_aep==9) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,priorityaep,AcreditaTramoIII]), salary3d)
salary28 = np.where((initial_p_aep==9) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,priorityaep,AcreditaTramoIII]), salary3d)
"""

    
salary = sum([salary1,salary3,salary5,salary7,salary9,salary11,salary13,salary15,salary17,salary19])
    
salary_pr = sum([salary21,salary22,salary23,salary24,salary25,salary26,salary27,salary28,salary29,salary30])
    