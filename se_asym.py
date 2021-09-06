#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code defines a class to compute analytical asymptoticsvar-cov matrix
"""


#from __future__ import division #omit for python 3.x
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
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
#sys.path.append("D:\Git\WageError")
#import gridemax
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
#import pybobyqa
#import xlsxwriter
from openpyxl import load_workbook



class SEs:
    def __init__(self,output_ins,var_cov,psi):
        """
		output_ins: instance of the estimate class
		var_cov: var-cov matrix of auxiliary estimates
		psi: structural parameters (estimated in II)
		"""
        self.output_ins = output_ins
        self.var_cov = var_cov
        self.psi = psi
        
    
    def betas_struct(self,bs):
        """
		Takes structural parameters and update the parameter instance
		bs: structural parameters

		"""

		#these are defined inside the output_ins instance and are held fixed
        dolar= 600
        value = [14403, 15155]        
        hw = [value[0]/dolar,value[1]/dolar]
        
        porc = [0.0338, 0.0333]
        
        qualiPesos = [72100*1.266, 24034*1.266, 253076, 84360]
        pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]
        
        progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]
        pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,  
           progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]
        
        Asig = [150000*1.111,100000*1.111,50000*1.111]
        AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar]

		
        alphas = [[bs[0], bs[1],0,bs[2],
              bs[3], bs[4]],
             [bs[5], 0,bs[6],bs[7],
              bs[8], bs[9]]]
        
        betas = [bs[10], bs[11], bs[12],bs[13],bs[14]]
        
        gammas = [bs[15],bs[16],bs[17]]


		#Re-defines the instance with parameters
        param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP)
        return param0
    
    
    def sim_moments(self,modelSD):
        """
        this function computes samples
        """
        result = self.output_ins.simulation(50,modelSD)
        
        beta_mport = result['Mean Portfolio'] 
        beta_vport = result['Var Port']
        beta_msimce = result['Mean SIMCE']
        beta_vsimce = result['Var SIMCE']
        beta_mtest = result['Mean Test']
        beta_vtest = result['Var Test']
        beta_mporttest = result['Mean PortTest']
        #beta_pinit = result['perc init']
        beta_pinter = result['perc inter']
        beta_padv = result['perc advanced']
        beta_pexpert = result['perc expert']
        beta_sport = result['Estimation SIMCE vs Portfolio']
        beta_spru = result['Estimation SIMCE vs Prueba']       
        beta_expport = result['Estimation EXP vs Portfolio']
        beta_exptest = result['Estimation EXP vs Prueba']
        beta_sexp = result['Estimation SIMCE vs Experience']
        beta_advexp_c = result['perc adv/exp control']
        beta_testp = result['Estimation Test vs p']
        beta_portp = result['Estimation Portfolio vs p']
        beta_portp = result['Estimation Portfolio vs p']
        
        
        return [beta_mport,beta_vport,beta_msimce,beta_vsimce,beta_mtest,
                beta_vtest,beta_mporttest,beta_pinter,beta_padv,beta_pexpert,
                beta_sport,beta_spru,beta_expport,beta_exptest,beta_sexp,
                beta_advexp_c,beta_testp,beta_portp]
    
    def binding(self,psi):
        
        param0 = self.betas_struct(psi)
        
        N = self.output_ins.__dict__['N']
        p1_0 = self.output_ins.__dict__['p1_0']
        p2_0 = self.output_ins.__dict__['p2_0']
        years = self.output_ins.__dict__['years']
        treatment = self.output_ins.__dict__['treatment']
        typeSchool = self.output_ins.__dict__['typeSchool']
        HOURS = self.output_ins.__dict__['HOURS']
        p1 = self.output_ins.__dict__['p1']
        p2 = self.output_ins.__dict__['p2']
        catPort = self.output_ins.__dict__['catPort']
        catPrueba = self.output_ins.__dict__['catPrueba']
        TrameI = self.output_ins.__dict__['TrameI']
                
        
        model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI)
        modelSD = sd.SimData(N,model)
        
        betas = self.sim_moments(modelSD)
        
        return betas

    def db_dtheta(self,psi,eps,K,S):
        
        db_dt = np.zeros((K,S))
        
        for s in range(S): # loop across parameters
            
            #evaluating at optimum
            psi_low = psi.copy()
            psi_high = psi.copy()

			#changing only relevant parameter, one at a time
            h = eps*abs(psi[s])
            psi_low[s] = psi[s] - h
            psi_high[s] = psi[s] + h


			#Computing betas
            betas_low = self.binding(psi_low)
            betas_high = self.binding(psi_high)
			
			#From list to numpy array
            betas_low_array = np.array([[betas_low[0]]])
            betas_high_array = np.array([[betas_high[0]]])
            
            for l in range(1,len(betas_low)):
                if type(betas_low[l]) is np.float64:
                    betas_low_array = np.concatenate( (betas_low_array,np.array([[betas_low[l]]])),axis=0 )
                    betas_high_array = np.concatenate( (betas_high_array,np.array([[betas_high[l]]])),axis=0 )
                else:
                    betas_low_array = np.concatenate( (betas_low_array,betas_low[l].reshape(betas_low[l].shape[0],1)),axis=0 )
                    betas_high_array = np.concatenate( (betas_high_array,betas_high[l].reshape(betas_high[l].shape[0],1)),axis=0 )
                    db_dt[:,s] = (betas_high_array[:,0] - betas_low_array[:,0]) / (psi_high[s]-psi_low[s])
                    
        return db_dt
    
    def big_sand(self,h,nmoments,npar):
        dbdt = self.db_dtheta(self.psi,h,nmoments,npar)
        
        print('dbdt', dbdt)
        V1_1 = np.dot(np.transpose(dbdt),self.output_ins.__dict__['w_matrix'])
        V1 = np.linalg.inv(np.dot(V1_1,dbdt))
        V2 = np.dot(np.dot(V1_1,self.var_cov),np.transpose(V1_1))
        
        return np.dot(np.dot(V1,V2),V1)
        
    




   