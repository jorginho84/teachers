"""
Utility class: takes parameters, X's, and given choices
computes utility. It modifies the original utility by
forcing all agents to go through the same production
functions
"""
# from __future__ import division #omit for python 3.x
import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
import math
from math import *
sys.path.append("/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers")
from utility import Utility


class Count_att_2(Utility):
    """ 

    This class modifies the economic environment of the agent

    """

    def __init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, 
                 catPort, catPrueba, TrameI,priotity, rural_rbd, locality, AEP_priority):
        """
        Calling baseline model

        """
        
        Utility.__init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, 
                 catPort, catPrueba, TrameI,priotity, rural_rbd, locality, AEP_priority)



    def student_h(self, effort):
        """
        takes student initial HC and teacher effort to compute achievement

        return: student test score, where effort_low = 0

        """
        d_effort_t1 = effort == 1
        d_effort_t2 = effort == 2
        d_effort_t3 = effort == 3
        
        effort_m = d_effort_t1 + d_effort_t3
        effort_h = d_effort_t2 + d_effort_t3
        
        p1v1_past = np.where(np.isnan(self.p1_0), 0, self.p1_0)
        p2v1_past = np.where(np.isnan(self.p2_0), 0, self.p2_0)
        
     
        p0_past = np.zeros(p1v1_past.shape)
        p0_past = np.where((p1v1_past == 0),p2v1_past, p0_past)
        p0_past = np.where((p2v1_past == 0),p1v1_past, p0_past)
        p0_past = np.where((p1v1_past != 0) & (p2v1_past != 0) ,(self.p1_0 + self.p2_0)/2, p0_past)
        p0_past = (p0_past-np.mean(p0_past))/np.std(p0_past)
        

    
        eps = np.random.randn(self.N)*self.param.betas[3]
        
        h_treated =  self.param.betas[0] + self.param.betas[1]*effort_m + self.param.betas[2]*effort_h + \
            self.param.betas[4]*self.years/10 + self.param.betas[5]*p0_past + eps

                

        return [h_treated,h_treated]
    
    def t_test(self,effort):
        """
        takes initial scores, effort and experience

        returns: test scores and portfolio

        """
        
 
        p1v1_past = np.where(np.isnan(self.p1_0), 0, self.p1_0)
        p2v1_past = np.where(np.isnan(self.p2_0), 0, self.p2_0)
        
     
        p0_past = np.zeros(p1v1_past.shape)
        p0_past = np.where((p1v1_past == 0),p2v1_past, p0_past)
        p0_past = np.where((p2v1_past == 0),p1v1_past, p0_past)
        p0_past = np.where((p1v1_past != 0) & (p2v1_past != 0) ,(self.p1_0 + self.p2_0)/2, p0_past)
        p0_past = (p0_past-np.mean(p0_past))/np.std(p0_past)
        
        d_effort_t1 = effort == 1
        d_effort_t2 = effort == 2
        d_effort_t3 = effort == 3
        
        effort_m = d_effort_t1 + d_effort_t3
        effort_h = d_effort_t2 + d_effort_t3
        
       
        pb_treated = []
                   
        for j in range(2):
            
            shock = np.random.normal(0, self.param.alphas[j][4], p1v1_past.shape)
            
            pb_treated.append(self.param.alphas[j][0] + \
                     self.param.alphas[j][1]*effort_m + self.param.alphas[j][2]*effort_h + \
                         self.param.alphas[j][3]*self.years/10 + self.param.alphas[j][5]*p0_past  + \
                             shock)
           

        p_treated = [((1/(1+np.exp(-pb_treated[0]))) + (1/3))*3, ((1/(1+np.exp(-pb_treated[1]))) + (1/3))*3]
        
        

                
        return [p_treated,p_treated]


