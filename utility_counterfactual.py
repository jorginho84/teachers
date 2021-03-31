#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:31:28 2021

@author: jorge-home
"""


# -*- coding: utf-8 -*-
"""


-Utility class: takes parameters, X's, and given choices
computes utility


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


class Count_1(Utility):
    """ 

    This class modifies the economic environment of the agent

    """

    def __init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, 
                 catPort, catPrueba, TrameI):
        """
        Calling baseline m odel

        """
        
        Utility.__init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, 
                 catPort, catPrueba, TrameI)
        
    
    def income(self, initial_p,tscores):
        """
        Defines the new budget set

        """
        # " WE SIMULATE A PROFESSOR WITH 44 HOURS "
        # " Values in dolars "
        HvalueE = self.param.hw[0]
        HvalueS = self.param.hw[1]
        
        
        
        RBMNElemt2 = np.zeros(initial_p.shape[0])
        RBMNSecond2 = np.zeros(initial_p.shape[0])
        ExpTrameE2 = np.zeros(initial_p.shape[0])
        ExpTrameS2 = np.zeros(initial_p.shape[0])
        BRP2 = np.zeros(initial_p.shape[0])
        BRPWithout2 = np.zeros(initial_p.shape[0])
        ATDPinitial2 = np.zeros(initial_p.shape[0])
        ATDPearly2 = np.zeros(initial_p.shape[0])
        ATDPadvanced2 = np.zeros(initial_p.shape[0])
        ATDPadvancedfixed2 = np.zeros(initial_p.shape[0])
        ATDPexpert12 = np.zeros(initial_p.shape[0])
        ATDPexpert1fixed2 = np.zeros(initial_p.shape[0])
        ATDPexpert22 = np.zeros(initial_p.shape[0])
        ATDPexpert2fixed2 = np.zeros(initial_p.shape[0])
        salary2d = np.zeros(initial_p.shape[0])
        
        
        
        # " RENTA BASE MÍNIMA NACIONAL "
        
        RBMNElemt = np.where((self.typeSchool==1),HvalueE*self.HOURS, RBMNElemt2)
        RBMNSecond = np.where((self.typeSchool==0),HvalueS*self.HOURS, RBMNSecond2)
        
                
        # " EXPERIENCE "
        
        bienniumtwoFalse = self.years/2
        biennium = np.floor(bienniumtwoFalse)
        
        porc1 = self.param.porc[0] 
        porc2 = self.param.porc[1] 
        
        ExpTrameE = np.where((self.typeSchool==1) & (self.years > 2), (porc1 + (porc2 * (biennium - 1))) * RBMNElemt, ExpTrameE2)
        ExpTrameS = np.where((self.typeSchool==0) & (self.years > 2), (porc1 + (porc2 * (biennium - 1))) * RBMNSecond, ExpTrameS2)
    
        
        # " VOCATIONAL RECOGNITION BONUS (BRP) "
        # We assume that every teacher have
        # degree and mention.
    
        professional_qualificationW = self.param.pro[0] 
        professional_mentionW = self.param.pro[1]
        professional_qualification = self.param.pro[2]
        professional_mention = self.param.pro[3]
        full_contract = 44
        
        BRP = np.where((self.typeSchool==1),(professional_qualification + professional_mention)*(self.HOURS/full_contract),BRP2)
        BRPWithout = np.where((self.typeSchool==1),(professional_qualificationW + professional_mentionW)*(self.HOURS/full_contract),BRPWithout2)
    
    
        # " PROGRESSION COMPONENT BY TRANCHE "
        # " COMPONENTE DE FIJO TRAMO "
    
        Proinitial = self.param.pol[0] 
        ProEarly = self.param.pol[1] 
        Proadvanced = self.param.pol[2] 
        Proadvancedfixed = self.param.pol[3] 
        Proexpert1 = self.param.pol[4] 
        Proexpert1fixed = self.param.pol[5] 
        Proexpert2 = self.param.pol[6] 
        Proexpert2fixed = self.param.pol[7] 
        
        
        ATDPinitial = np.where((initial_p==1) & (self.years > 1), (Proinitial/15)*(self.HOURS/full_contract)*biennium, ATDPinitial2)
        ATDPearly = np.where((initial_p==2) & (self.years > 3), (ProEarly/15)*(self.HOURS/full_contract)*biennium, ATDPearly2)
        ATDPadvanced = np.where((initial_p==3) & (self.years > 3), (Proadvanced/15)*(self.HOURS/full_contract)*biennium, ATDPadvanced2)
        ATDPadvancedfixed = np.where((initial_p==3) & (self.years > 3), (Proadvancedfixed/15)*(self.HOURS/full_contract)*biennium, ATDPadvancedfixed2)
        ATDPexpert1 = np.where((initial_p==4) & (self.years > 7), (Proexpert1/15)*(self.HOURS/44)*biennium, ATDPexpert12)
        ATDPexpert1fixed = np.where((initial_p==4) & (self.years > 7), (Proexpert1fixed/15)*(self.HOURS/full_contract)*biennium, ATDPexpert1fixed2)
        ATDPexpert2 = np.where((initial_p==5) & (self.years > 11), (Proexpert2/15)*(self.HOURS/full_contract)*biennium, ATDPexpert22)
        ATDPexpert2fixed = np.where((initial_p==5) & (self.years > 11), (Proexpert2fixed/15)*(self.HOURS/full_contract)*biennium, ATDPexpert2fixed2)
        
        # " AEP (Teaching excellence)
        
        AcreditaTramoI = self.param.AEP[0]
        AcreditaTramoII = self.param.AEP[1]
        AcreditaTramoIII = self.param.AEP[2]
    
            
        # " SUM OF TOTAL SALARY "
        
        salary1 = np.where((initial_p==1) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPinitial]), salary2d)
        salary2 = np.where((initial_p==1) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPinitial]), salary2d)
        salary3 = np.where((initial_p==1) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout]), salary2d)
        salary4 = np.where((initial_p==1) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout]), salary2d)
        
        salary5 = np.where((initial_p==2) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPearly]), salary2d)
        salary6 = np.where((initial_p==2) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPearly]), salary2d)
        salary7 = np.where((initial_p==2) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout]), salary2d)
        salary8 = np.where((initial_p==2) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout]), salary2d)
        
        salary9 = np.where((initial_p==3) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPadvanced,ATDPadvancedfixed]), salary2d)
        salary10 = np.where((initial_p==3) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPadvanced,ATDPadvancedfixed]), salary2d)
        salary11 = np.where((initial_p==3) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout,AcreditaTramoIII]), salary2d)
        salary12 = np.where((initial_p==3) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout,AcreditaTramoIII]), salary2d)
        
        salary13 = np.where((initial_p==4) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert1,ATDPexpert1fixed]), salary2d)
        salary14 = np.where((initial_p==4) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert1,ATDPexpert1fixed]), salary2d)
        salary15 = np.where((initial_p==4) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout,AcreditaTramoII]), salary2d)
        salary16 = np.where((initial_p==4) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout,AcreditaTramoII]), salary2d)
        
        salary17 = np.where((initial_p==5) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert2,ATDPexpert2fixed]), salary2d)
        salary18 = np.where((initial_p==5) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert2,ATDPexpert2fixed]), salary2d)
        salary19 = np.where((initial_p==5) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout,AcreditaTramoI]), salary2d)
        salary20 = np.where((initial_p==5) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout,AcreditaTramoI]), salary2d)
        
        salary = sum([salary1,salary2,salary3,salary4,salary5,salary6,salary7,salary8,salary9,salary10, \
                      salary11,salary12,salary13,salary14,salary15,salary16,salary17,salary18,salary19,salary20])
        
                    #taking the minimum/max salary to fit a line
        b = (3250-1500)/3
        a = 1500 - b
        
        salary[(self.treatment == 1)] = a + b*(tscores[0] + tscores[1])/2

        return salary


