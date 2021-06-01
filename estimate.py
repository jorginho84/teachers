# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:22:05 2020

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




class estimate:
    "This class generate descriptive statiscals"
    
    def __init__(self,N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, w_matrix,moments_vector):
        "Initial class"
        
        self.N = N
        #self.modelSD = modelSD
        self.years = years
        self.treatment = treatment
        self.param0 = param0
        self.p1_0 = p1_0
        self.p2_0 = p2_0
        self.typeSchool = typeSchool
        self.HOURS = HOURS
        self.p1 = p1
        self.p2 = p2
        self.catPort = catPort
        self.catPrueba = catPrueba
        self.TrameI = TrameI
        self.moments_vector,self.w_matrix=moments_vector,w_matrix
        self.p1_0 = p1_0
        self.p2_0 = p2_0
        
        
    
    def simulation(self,times,modelSD):
        "Function that simulate x times."
        
        est_corrSPort = np.zeros(times)
        est_corrSPrue = np.zeros(times)
        #est_corrPP = np.zeros(times)
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
            opt = modelSD.choice(self.treatment)
            dataf = {'SIMCE': opt['Opt Simce'], 'PORTFOLIO': opt['Opt Teacher'][0], 'TEST': opt['Opt Teacher'][1], 
                     'EXP': self.years, 'TREATMENT': opt['Treatment'], 'PLACEMENT': opt['Opt Placement'], 
                     'PORTPAST': self.p1_0, 'TESTPAST': self.p2_0}
            # Here we consider the database complete
            datadfT = pd.DataFrame(dataf, columns=['SIMCE','PORTFOLIO','TEST', 'EXP', 'TREATMENT', 'PLACEMENT', 'PORTPAST', 'TESTPAST'])
            #1 Portfolio mean
            est_mean_Port[i] = np.mean(np.array(datadfT['PORTFOLIO']))
            #2 Portfolio var
            est_var_Port[i] = np.var(np.array(datadfT['PORTFOLIO']))
            #3 SIMCE mean
            est_mean_SIMCE[i] = np.mean(np.array(datadfT['SIMCE']))
            #4 SIMCE var
            est_var_SIMCE[i] = np.var(np.array(datadfT['SIMCE']))
            #5 Test mean
            est_mean_Pru[i] = np.mean(np.array(datadfT['TEST']))
            #6 Test var
            est_var_Pru[i] = np.var(np.array(datadfT['TEST']))
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

            # Here we consider the data for treatmetn group
            datav = datadfT[datadfT['TREATMENT']==1]
            #8
            perc_init[i] = np.mean(datav['PLACEMENT']==1)
            #9
            perc_inter[i] = np.mean(datav['PLACEMENT']==2) 
            #10
            perc_advan[i] = np.mean(datav['PLACEMENT']==3) 
            #11
            perc_expert[i] = np.mean((datav['PLACEMENT']==4)| (datav['PLACEMENT']==5))
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
            perc_inter_c[i] = np.mean((datav_2['PLACEMENT']==2))
            perc_avanexpet_c[i] = np.mean((datav['PLACEMENT']==3) | (datav['PLACEMENT']==4)| (datav['PLACEMENT']==5))
            p1 = np.array(datav_2['PORTFOLIO'])
            p2 = np.array(datav_2['TEST'])
            p1v1 = np.where(np.isnan(p1), 0, p1)
            p2v1 = np.where(np.isnan(p2), 0, p2)
            p0 = np.zeros(p1.shape)
            p0 = np.where((p1v1 == 0),p2v1, p0)
            p0 = np.where((p2v1 == 0),p1v1, p0)
            p0 = np.where((p1v1 != 0) & (p2v1 != 0) ,(p1 + p2)/2, p0)
            est_mean_PortTest[i] = np.mean(p0)
        
        est_bootsSPort = np.mean(est_corrSPort)
        est_bootsSPrue = np.mean(est_corrSPrue)
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
    
    
    def objfunction(self,beta):
        """Define objective function how a substraction of real and
        simulate data """
        
        self.param0.alphas[0][0] = beta[0]
        self.param0.alphas[0][1] = beta[1]
        #self.param0.alphas[0][2] = beta[2]
        self.param0.alphas[0][3] = beta[2]
        self.param0.alphas[0][4] = beta[3]
        self.param0.alphas[0][5] = beta[4]
        self.param0.alphas[1][0] = beta[5]
        #self.param0.alphas[1][1] = beta[7]
        self.param0.alphas[1][2] = beta[6]
        self.param0.alphas[1][3] = beta[7]
        self.param0.alphas[1][4] = beta[8]
        self.param0.alphas[1][5] = beta[9]
        self.param0.betas[0] = beta[10]
        self.param0.betas[1] = beta[11]
        self.param0.betas[2] = beta[12]
        self.param0.betas[3] = beta[13]
        self.param0.gammas[0] = beta[14]
        self.param0.gammas[1] = beta[15]
        self.param0.gammas[2] = beta[16]
        
        model = util.Utility(self.param0,self.N,self.p1_0,self.p2_0,self.years,self.treatment, \
                             self.typeSchool,self.HOURS,self.p1,self.p2,self.catPort,self.catPrueba,self.TrameI)
            
        modelSD = sd.SimData(self.N,model,self.treatment)
            
        #modelestimate = est.estimate(self.N,modelSD,self.years,self.treatment,self.param0)
        
        #result = modelestimate.simulation(50)
        
        result = self.simulation(50,modelSD)
        
        beta_mport = result['Mean Portfolio'] 
        beta_vport = result['Var Port']
        beta_msimce = result['Mean SIMCE']
        beta_vsimce = result['Var SIMCE']
        beta_mtest = result['Mean Test']
        beta_vtest = result['Var Test']
        beta_mporttest = result['Mean PortTest']
        beta_pinit = result['perc init']
        beta_pinter = result['perc inter']
        beta_padv = result['perc advanced']
        beta_pexpert = result['perc expert']
        beta_sport = result['Estimation SIMCE vs Portfolio']
        beta_spru = result['Estimation SIMCE vs Prueba']       
        beta_expport = result['Estimation EXP vs Portfolio']
        beta_exptest = result['Estimation EXP vs Prueba']
        beta_inter_c = result['perc inter control']
        beta_advexp_c = result['perc adv/exp control']
        beta_testp = result['Estimation Test vs p']
        beta_portp = result['Estimation Portfolio vs p']
        
        
        #Number of moments to match
        num_par = beta_mport.size + beta_vport.size + beta_msimce.size + beta_vsimce.size + beta_mtest.size + \
            beta_vtest.size + beta_mporttest.size + beta_pinit.size + beta_pinter.size + beta_padv.size + \
                beta_pexpert.size + beta_sport.size + beta_spru.size + beta_expport.size + beta_exptest.size + \
                    beta_inter_c.size + beta_advexp_c.size + beta_testp.size + beta_portp.size
        
        #Outer matrix
        x_vector=np.zeros((num_par,1))
        x_vector[0,0] = beta_mport - self.moments_vector[0]
        x_vector[1,0] = beta_vport - self.moments_vector[1]
        x_vector[2,0] = beta_msimce - self.moments_vector[2]
        x_vector[3,0] = beta_vsimce - self.moments_vector[3]
        x_vector[4,0] = beta_mtest - self.moments_vector[4]
        x_vector[5,0] = beta_vtest - self.moments_vector[5]
        x_vector[6,0] = beta_mporttest - self.moments_vector[6]
        x_vector[7,0] = beta_pinit - self.moments_vector[7]
        x_vector[8,0] = beta_pinter - self.moments_vector[8]
        x_vector[9,0] = beta_padv - self.moments_vector[9]
        x_vector[10,0] = beta_pexpert - self.moments_vector[10]
        x_vector[11,0] = beta_sport - self.moments_vector[11]
        x_vector[12,0] = beta_spru - self.moments_vector[12]
        x_vector[13,0] = beta_expport - self.moments_vector[13]
        x_vector[14,0] = beta_exptest - self.moments_vector[14]
        x_vector[15,0] = beta_inter_c - self.moments_vector[15]
        x_vector[16,0] = beta_advexp_c - self.moments_vector[16]
        x_vector[17,0] = beta_testp - self.moments_vector[17]
        x_vector[18,0] = beta_portp - self.moments_vector[18]
        
        
        #The Q metric
        q_w = np.dot(np.dot(np.transpose(x_vector),self.w_matrix),x_vector)
                   
        print ('')
        print ('The objetive function value equals ', q_w)
        print ('')

        
    
                 
    
        return q_w
    
    
    def optimizer(self):
        
        beta0 = np.array([self.param0.alphas[0][0],
                          self.param0.alphas[0][1],
                          #self.param0.alphas[0][2],
                          self.param0.alphas[0][3],
                          self.param0.alphas[0][4],
                          self.param0.alphas[0][5],
                          self.param0.alphas[1][0],
                          #self.param0.alphas[1][1],
                          self.param0.alphas[1][2],
                          self.param0.alphas[1][3],
                          self.param0.alphas[1][4],
                          self.param0.alphas[1][5],
                          self.param0.betas[0],
                          self.param0.betas[1],
                          self.param0.betas[2],
                          self.param0.betas[3],
                          self.param0.gammas[0],
                          self.param0.gammas[1],
                          self.param0.gammas[2]])
      
        #Here we go
        opt = minimize(self.objfunction, beta0,  method='Nelder-Mead', options={'maxiter':5000, 'maxfev': 90000, 'ftol': 1e-3, 'disp': True})
        #opt = minimize(self.objfunction, beta0,  method='BFGS', options={'maxiter':5000, 'gtol': 1e-3, 'disp': True})
		#opt = pybobyqa.solve(self.ll, beta0)
        
        return opt
        

        
        
        
        
        