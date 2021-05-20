#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:39:56 2021

@author: jorge-home

This class modifies SIMDATA

"""

import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
import math
from math import *
from scipy.optimize import minimize

class SimDataC:
    """
    eitc: eitc function
    emax_function: interpolating instance
    The rest are state variables at period 0
    """
    
    def __init__(self,N,model,treatment):
        """
        model: a utility instance (with arbitrary parameters)
        """
        self.N = N
        self.model = model
        self.treatment = treatment
        
    def util(self,effort):
        """
        This function takes effort and computes utils
        """
        

        
        initial_p = self.model.initial()
        
        teacher_scores = self.model.t_test(effort)
        
        placement = self.model.placement(teacher_scores)
        
        income = self.model.income(placement,teacher_scores)
        
        simce = self.model.student_h(effort)
        
                
        return self.model.utility(income, effort, simce)
    
    
    def choice(self,treatment):
        """
        computes optimal effort values. Maximizes util(effort).
        """
        
        effort0 = np.zeros(self.N)
        effort1 = np.ones(self.N)
        effort2 = np.ones(self.N)*2
        effort3 = np.ones(self.N)*3
        
        u_0 = self.util(effort0)
        u_1 = self.util(effort1)
        u_2 = self.util(effort2)
        u_both = self.util(effort3)
        
        u_v2 = np.array([u_0, u_1, u_2, u_both]).T
        
        effort_v1 = np.argmax(u_v2, axis=1)
        
        teacher_scores = self.model.t_test(effort_v1)
        
        placement = self.model.placement(teacher_scores)
        
        income = self.model.income(placement,teacher_scores)
        
        simce = self.model.student_h(effort_v1)
        
        #updating
        effort_opt = effort_v1
        teacher_opt = teacher_scores
        simce_opt = simce
        placement_opt = placement
        income_opt = income
        treatment = treatment
        
                        
        return {'Opt Effort': effort_opt, 'Opt Teacher': teacher_opt, 'Opt Simce': simce_opt,
                'Opt Placement': placement_opt, 'Opt Income': income_opt, 'Treatment': treatment}