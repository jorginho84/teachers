"""
This code simulates solves the teacher's problem and computes utility

"""

import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
import math
from math import *
from scipy.optimize import minimize


class SimData:
    """
    eitc: eitc function
    emax_function: interpolating instance
    The rest are state variables at period 0
    """
    
    def __init__(self,N,model):
        """
        model: a utility instance (with arbitrary parameters)
        """
        self.N = N
        self.model = model
        
        
    def util(self,effort):
        """
        This function takes effort and computes utils
        """
        

        
        teacher_scores = self.model.t_test(effort)
        
        placement = self.model.placement(teacher_scores[0])
        
        income = self.model.income(placement)
        
        
        student_h = self.model.student_h(effort)
        
        
        return self.model.utility(income, effort, student_h)
    
    
    def choice(self):
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
        
        placement = self.model.placement(teacher_scores[0])
        
        income = self.model.income(placement)
        
        student_h = self.model.student_h(effort_v1)
        
        simce = self.model.measure_simce(student_h)
        

                                
        return {'Opt Effort': effort_v1, 'Opt Simce': simce,
                'Opt Placement': placement, 'Opt Income': income,
                'Opt Teacher': teacher_scores[0], 'Potential scores': teacher_scores[1]}
