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
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est




class estimate:
    "This class estimates the parameters of the model"
    
    def __init__(self,N, years,param0, p1_0,p2_0,treatment, \
                 typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,priority,rural_rbd,locality, AEP_priority, \
                 w_matrix,moments_vector):
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
        self.moments_vector = moments_vector
        self.w_matrix = w_matrix
        self.p1_0 = p1_0
        self.p2_0 = p2_0
        self.priority = priority
        self.rural_rbd = rural_rbd
        self.locality = locality
        self.AEP_priority = AEP_priority
        
        
    
    def simulation(self,times,modelSD):
        "Simulates samples given set of parameters."
        
        #Full Sample
        est_corrSEXP = np.zeros(times)
        est_corr_EXPPort = np.zeros(times)
        est_corr_EXPPru = np.zeros(times)
        est_corrSPort = np.zeros(times)
        est_corrSPrue = np.zeros(times)

        #Treated sample
        est_mean_SIMCE_treated = np.zeros(times)
        est_var_SIMCE_treated = np.zeros(times)
        est_mean_Port_treated = np.zeros(times)
        est_mean_Pru_treated = np.zeros(times)
        est_var_Port_treated = np.zeros(times)
        est_var_Pru_treated = np.zeros(times)
        est_corrSPast = np.zeros(times)
        est_corrPortp = np.zeros(times)
        est_corrTestp = np.zeros(times)
        est_share_port_treated = np.zeros(times)
        est_share_stei_treated = np.zeros(times)
        
        #Control sample
        est_mean_SIMCE_control = np.zeros(times)
        

        for i in range(1,times):
           
            np.random.seed(i+100)
            opt = modelSD.choice()
        
        
            p1v1 = np.where(np.isnan(self.p1_0), 0, self.p1_0)
            p2v1 = np.where(np.isnan(self.p2_0), 0, self.p2_0)
            p0_past = np.zeros(self.p1_0.shape)
            p0_past = np.where((p1v1 == 0),p2v1, p0_past)
            p0_past = np.where((p2v1 == 0),p1v1, p0_past)
            p0_past = np.where((p1v1 != 0) & (p2v1 != 0) ,(self.p1_0 + self.p2_0)/2, p0_past)
            simce = opt['Opt Simce']
            portfolio = opt['Opt Teacher'][0]
            stei = opt['Opt Teacher'][1]
            
            dataf = {'SIMCE': simce, 
                    'PORTFOLIO': portfolio,
                    'STEI': stei,
                    'EXP': self.years, 'PLACEMENT treated': opt['Opt Placement'][0], 'PLACEMENT control': opt['Opt Placement'][1],
                     'PORTPAST': self.p1_0, 'TESTPAST': self.p2_0,'P_past': p0_past, 'Initial Placement': self.TrameI}
            
            
            #### A. Full sample###
            datadfT = pd.DataFrame(dataf, columns=['SIMCE','PORTFOLIO','STEI','EXP', 'PLACEMENT treated', 'PLACEMENT control' ,'PORTPAST', 'TESTPAST','P_past','Initial Placement'])

            #### B. Treated sample###
            data_treated =  datadfT[self.treatment == 1]
            corrM = data_treated.corr()
            #Moments: corr experience and SIMCE
            est_corrSEXP[i] = corrM.iloc[0]['EXP']

            #Moments: Corr experience and teacher test scores
            est_corr_EXPPort[i] = corrM.iloc[1]['EXP']
            est_corr_EXPPru[i] = corrM.iloc[2]['EXP']

            #Moments: Corr teacher test scores and SIMCE
            est_corrSPort[i] = corrM.iloc[1]['SIMCE']
            est_corrSPrue[i] = corrM.iloc[2]['SIMCE']

            #Moments: means and vars of SIMCE, Portfolio and STEI
            est_mean_SIMCE_treated[i] = np.mean(data_treated['SIMCE'])
            est_var_SIMCE_treated[i] = np.var(data_treated['SIMCE'])
            est_mean_Port_treated[i] = np.mean(data_treated['PORTFOLIO'])
            est_mean_Pru_treated[i] = np.mean(data_treated['STEI'])
            est_var_Port_treated[i] = np.var(data_treated['PORTFOLIO'])
            est_var_Pru_treated[i] = np.var(data_treated['STEI'])          

            
            #Moments: corr of SIMCE, Portfolio and STEU with past test scores
            est_corrSPast[i] = corrM.iloc[0]['P_past']
            est_corrPortp[i] = corrM.iloc[1]['P_past']
            est_corrTestp[i] = corrM.iloc[2]['P_past']

            #Moments: share of teachers with Portfolio > 2.5
            boo_port = data_treated['PORTFOLIO'] >= 2.5
            est_share_port_treated[i] = np.mean(boo_port)
            
            #Moments: share of teachers with STEI > 2.74
            boo_stei = data_treated['STEI'] >= 2.74
            est_share_stei_treated[i] = np.mean(boo_stei)
            

            ### C. Control Sample ####
            data_control = datadfT[self.treatment == 0]


            #Moments: means and vars of SIMCE, Port and STEI (control groups)
            est_mean_SIMCE_control[i] = np.mean(data_control['SIMCE'])
                                    
        
        ####MEANS####
        #Treated sample
        corrSEXP = np.mean(est_corrSEXP)
        corr_EXPPort = np.mean(est_corr_EXPPort)
        corr_EXPPru = np.mean(est_corr_EXPPru)
        corrSPort = np.mean(est_corrSPort)
        corrSPrue = np.mean(est_corrSPrue)
    
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
            
            'SIMCE Mean (control)': mean_SIMCE_control}
    
    
    def objfunction(self,beta):
        "Computes value function given a set of parameters"
        
        self.param0.alphas[0][1] = beta[0]
        self.param0.alphas[0][3] = beta[1]
        self.param0.alphas[0][4] = np.exp(beta[2])
        self.param0.alphas[0][5] = beta[3]
        self.param0.alphas[1][2] = beta[4]
        self.param0.alphas[1][3] = beta[5]
        self.param0.alphas[1][4] = np.exp(beta[6])
        self.param0.alphas[1][5] = beta[7]
        self.param0.betas[0] = beta[8]
        self.param0.betas[1] = beta[9]
        self.param0.betas[2] = beta[10]
        self.param0.betas[3] = np.exp(beta[11])
        self.param0.betas[4] = beta[12]
        self.param0.betas[5] = beta[13]
        self.param0.gammas[0] = beta[14]
        self.param0.gammas[1] = beta[15]
        self.param0.gammas[2] = beta[16]

        
        model = util.Utility(self.param0,self.N,self.p1_0,self.p2_0,self.years,self.treatment, \
                             self.typeSchool,self.HOURS,self.p1,self.p2,self.catPort,self.catPrueba,self.TrameI, \
                             self.priority,self.rural_rbd,self.locality,self.AEP_priority)

            
        modelSD = sd.SimData(self.N,model)
            
        result = self.simulation(50,modelSD)
        
        moments_model = np.array([result['Corr Simce and experience'],
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
            result['SIMCE Mean (control)']])
        
        
        
        num_par = self.moments_vector.shape[0]
        #Outer matrix
        x_vector = np.zeros((num_par,1))
        x_vector[:,0] = self.moments_vector - moments_model  
        
        
        
        #The Q metric
        q_w = np.dot(np.dot(np.transpose(x_vector),self.w_matrix),x_vector)
        
        
                   
        print ('')
        print ('The objetive function value equals ', q_w)
        print ('')

    
        return q_w
    
    
    def optimizer(self):
        "Uses Nelder-Mead to optimize"
        
        beta0 = np.array([self.param0.alphas[0][1],
                          self.param0.alphas[0][3],
                          np.log(self.param0.alphas[0][4]),
                          self.param0.alphas[0][5],
                          self.param0.alphas[1][2],
                          self.param0.alphas[1][3],
                          np.log(self.param0.alphas[1][4]),
                          self.param0.alphas[1][5],
                          self.param0.betas[0],
                          self.param0.betas[1],
                          self.param0.betas[2],
                          np.log(self.param0.betas[3]),
                          self.param0.betas[4],
                          self.param0.betas[5],
                          self.param0.gammas[0],
                          self.param0.gammas[1],
                          self.param0.gammas[2]])
      
        #Here we go
        opt = minimize(self.objfunction, beta0,  method='Nelder-Mead', options={'maxiter':5000, 'maxfev': 90000, 'ftol': 1e-3, 'disp': True})
        #opt = minimize(self.objfunction, beta0,  method='BFGS', options={'maxiter':5000, 'gtol': 1e-3, 'disp': True})
		#opt = pybobyqa.solve(self.ll, beta0)
        
        return opt
        

        
        
        
        
        