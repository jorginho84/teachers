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


class Utility(object):
    """ 

    This class defines the economic environment of the agent

    """

    def __init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, catPort, catPrueba, TrameI):
        """
        Set up model's data and paramaters

        treatment: 1 if we simulate carrera docente. 0 otherwise.

        """

        self.param = param
        self.N = N
        self.p1_0,self.p2_0 = p1_0,p2_0
        self.years = years
        self.treatment = treatment
        self.typeSchool = typeSchool
        self.HOURS = HOURS
        self.p1 = p1
        self.p2 = p2
        self.catPort = catPort
        self.catPrueba = catPrueba
        self.TrameI = TrameI
        
    
    def initial(self):
        
        initial_p = np.zeros(self.p1.shape[0])
        
        initial_p[(self.TrameI=='INICIAL')] = 1
        initial_p[(self.TrameI=='TEMPRANO')] = 2
        initial_p[(self.TrameI=='AVANZADO')] = 3
        initial_p[(self.TrameI=='EXPERTO I')] = 4
        initial_p[(self.TrameI=='EXPERTO II')] = 5
        
        return initial_p
        
              
                
        
    def placement(self,tscores):

        # *I want to replicate the typecasting of the teachers to the tramo
        # puntajeportafolio := p1
        # Puntajepruebaconocimientosespecí := self.p2
        # The tranche nameXY is defined how X: row nad Y:column
        # We have {'INICIAL': 1, 'TEMPRANO': 2, 'AVANZADO': 3 , 'EXPERTO I': 4, 'EXPERTO II': 5}

        # Inicializar vector initial_p
        #initial_p = np.array(['']*(len(self.p1)))
        placementF = np.zeros(self.p1.shape[0])
        
        placementF[(self.years < 4)]=1
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2 
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=3
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & (tscores[0]<2) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=1
        placementF[((self.years >= 4) & (self.years < 8)) & (tscores[0]<2) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=1
        placementF[((self.years >= 4) & (self.years < 8)) & (tscores[0]<2) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=1
        placementF[((self.years >= 4) & (self.years < 8)) & (tscores[0]<2) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=1
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=3
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=3
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=2
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=3
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=3
        placementF[((self.years >= 4) & (self.years < 8)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=3
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=4
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=4
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=4
        placementF[((self.years >= 8) & (self.years < 12)) & (tscores[0]<2) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=1
        placementF[((self.years >= 8) & (self.years < 12)) & (tscores[0]<2) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=1
        placementF[((self.years >= 8) & (self.years < 12)) & (tscores[0]<2) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=1
        placementF[((self.years >= 8) & (self.years < 12)) & (tscores[0]<2) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=1
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=2
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=2
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=3
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=2
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=3
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=3
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=4
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=4
        placementF[((self.years >= 8) & (self.years < 12)) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=4
        placementF[(self.years >= 12) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=5
        placementF[(self.years >= 12) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=5
        placementF[(self.years >= 12) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=5
        placementF[(self.years >= 12) & (tscores[0]<2) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=1
        placementF[(self.years >= 12) & (tscores[0]<2) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=1
        placementF[(self.years >= 12) & (tscores[0]<2) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=1
        placementF[(self.years >= 12) & (tscores[0]<2) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=1
        placementF[(self.years >= 12) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[(self.years >= 12) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=2
        placementF[(self.years >= 12) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=2
        placementF[(self.years >= 12) & ((tscores[0]>=2) & (tscores[0]<=2.25)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=3
        placementF[(self.years >= 12) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[(self.years >= 12) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=2
        placementF[(self.years >= 12) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=3
        placementF[(self.years >= 12) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[(self.years >= 12) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=3
        placementF[(self.years >= 12) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]>=1) & (tscores[1] <= 1.87))]=2
        placementF[(self.years >= 12) & ((tscores[0]>2.25) & (tscores[0]<=2.5)) & ((tscores[1]> 3.38) & (tscores[1] <= 4))]=4
        placementF[(self.years >= 12) & ((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38))]=4
        placementF[(self.years >= 12) & ((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74))]=4
        np.warnings.filterwarnings('ignore')
        

        return placementF
    

    def income(self, initial_p):
        """
        takes treatment dummy, teach test scores, 
        initial placement, and self.years of experience

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
        salary11 = np.where((initial_p==3) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout]), salary2d)
        salary12 = np.where((initial_p==3) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout]), salary2d)
        
        salary13 = np.where((initial_p==4) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert1,ATDPexpert1fixed]), salary2d)
        salary14 = np.where((initial_p==4) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert1,ATDPexpert1fixed]), salary2d)
        salary15 = np.where((initial_p==4) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout]), salary2d)
        salary16 = np.where((initial_p==4) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout]), salary2d)
        
        salary17 = np.where((initial_p==5) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert2,ATDPexpert2fixed]), salary2d)
        salary18 = np.where((initial_p==5) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert2,ATDPexpert2fixed]), salary2d)
        salary19 = np.where((initial_p==5) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRPWithout]), salary2d)
        salary20 = np.where((initial_p==5) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRPWithout]), salary2d)
        
        salary = sum([salary1,salary2,salary3,salary4,salary5,salary6,salary7,salary8,salary9,salary10, \
                      salary11,salary12,salary13,salary14,salary15,salary16,salary17,salary18,salary19,salary20])


        return salary
    
    def distance(self, initial_pF):
        """
        Take the evaluation in portfolio and test.
            
        Returns
        -------
        distancetrame : Vector of next tramo to reach.
    
        """        
        
        concat = self.catPort +  self.catPrueba
        concat = concat.reshape(-1)
        concat2fg = [str(i) for i in initial_pF]
        arr = np.array(concat2fg)
        concatlast = arr + concat
        
        #This is Next trame
        
        tramolet = np.zeros(self.catPort.shape)
        
        #** Initial **
        tramolet[(concatlast =="0.0A") | (concatlast =="0.0B") | (concatlast =="0.0C") | (concatlast =="0.0D") | \
                 (concatlast =="0.0E")]= 154 #"INICIALED"
        #** Early **
        tramolet[(concatlast =="1.0EA") | (concatlast =="1.0DB")]= 241 #"TEMPRANODA"    
        tramolet[(concatlast =="1.0EB") | (concatlast =="1.0DC") | (concatlast =="1.0D")]= 242 #"TEMPRANODB"
        tramolet[(concatlast =="1.0EC") | (concatlast =="1.0DD")]= 243 #"TEMPRANODC"
        tramolet[(concatlast =="1.0ED") | (concatlast =="1.0E")]= 244 #"TEMPRANODD"
        tramolet[(concatlast =="1.0A")]= 214 #"TEMPRANOAD"
        #** Advanced **
        tramolet[(concatlast =="1.0CD") |(concatlast =="2.0DB") | (concatlast =="2.0CC") | (concatlast =="2.0D") | \
                 (concatlast =="2.0C") | (concatlast =="1.0CC") | (concatlast =="1.0DA") | (concatlast =="1.0C") | \
                     (concatlast =="2.0ED") | (concatlast =="2.0EC") | (concatlast =="2.0EB") | (concatlast =="2.0DD") | \
                         (concatlast =="2.0DC") | (concatlast =="2.0CD")]= 332 #"AVANZADOCB"
        tramolet[(concatlast =="2.0BD") | (concatlast =="2.0B") | (concatlast =="1.0B") | (concatlast =="1.0B")]= 323 #"AVANZADOBC"
        tramolet[(concatlast =="2.0BD") | (concatlast =="2.0B") | (concatlast =="1.0B") | (concatlast =="1.0B") | \
                 (concatlast =="1.0BD")]= 323 #"AVANZADOBC"
        #** Expert I **
        tramolet[(concatlast =="2.0DA") | (concatlast =="3.0CB") | (concatlast =="3.0D") | (concatlast =="3.0CC") | \
                 (concatlast =="3.0CD") | (concatlast =="3.0DA") | (concatlast =="3.0DB") | (concatlast =="3.0DC") | \
                     (concatlast =="3.0DD") | (concatlast =="3.0EA") | (concatlast =="3.0EB") | (concatlast =="3.0C") | \
                         (concatlast =="3.0E") | (concatlast =="3.0EC") | (concatlast =="3.0ED") | (concatlast =="2.0CB")]= 431 #"EXPERTO ICA"
        tramolet[(concatlast =="2.0AD") | (concatlast =="3.0A") | (concatlast =="3.0AD")]= 413 #"EXPERTO IAC"
        tramolet[(concatlast =="3.0BC") | (concatlast =="3.0B") | (concatlast =="3.0BD") | (concatlast =="3.0CA")]= 422 #"EXPERTO IBB"
        #** Expert II **
        tramolet[(concatlast =="4.0CA") | (concatlast =="4.0BB") | (concatlast =="4.0BC") | (concatlast =="4.0CB") | \
                 (concatlast =="4.0DA") | (concatlast =="4.0CC") | (concatlast =="4.0DB") | (concatlast =="4.0EA")]= 521 #"EXPERTO IIBA"
        tramolet[(concatlast =="4.0AC")]= 512 #"EXPERTO IIAB"
        tramolet[(concatlast =="4.0AB")]= 512 #"EXPERTO IIAA"
        #** Top Teacher **
        tramolet[(concatlast =="4.0BA") | (concatlast =="4.0AA") | (concatlast =="3.0AA") | (concatlast =="3.0AB") | \
                 (concatlast =="3.0BA") | (concatlast =="1.0AA") | (concatlast =="1.0AB") | (concatlast =="1.0AC") | \
                     (concatlast =="1.0AD") | (concatlast =="1.0BA") | (concatlast =="1.0BB") | (concatlast =="1.0BC") | \
                         (concatlast =="1.0CA") | (concatlast =="1.0CB") | (concatlast =="5.0AA") | (concatlast =="5.0AB") | \
                             (concatlast =="5.0BA") | (concatlast =="3.0BB") | (concatlast =="3.0AC") | (concatlast =="5.0BB") | \
                                 (concatlast =="5.0CA") | (concatlast =="5.0DB")]= 6 #"TOP TEACHER"
            
        nexttramo = tramolet
        
        #This is distance
        
        row = np.zeros(self.catPort.shape)
        
        row[(concatlast =="0.0A") | (concatlast =="0.0B") | (concatlast =="0.0C") | (concatlast =="0.0D") | \
            (concatlast =="1.0E")]= 1
        row[(concatlast =="1.0ED") | (concatlast =="1.0EC") | (concatlast =="1.0EB") | (concatlast =="1.0EA")]= 2
        row[(concatlast =="2.0DA") | (concatlast =="2.0DB") | (concatlast =="2.0D") | (concatlast =="1.0DA") | \
            (concatlast =="3.0D") | (concatlast =="2.0D") | (concatlast =="2.0ED") | (concatlast =="2.0EC") | \
                (concatlast =="2.0EB") | (concatlast =="2.0DD") | (concatlast =="2.0DC") | (concatlast =="3.0DA") | \
                    (concatlast =="3.0E") | (concatlast =="3.0DB") | (concatlast =="3.0EC") | (concatlast =="3.0DC") | \
                        (concatlast =="3.0ED") | (concatlast =="3.0DD") | (concatlast =="3.0EA") | (concatlast =="3.0EB")]= 2.26
        row[(concatlast =="3.0CA") | (concatlast =="4.0CA") | (concatlast =="4.0CB") | (concatlast =="4.0DA") | \
            (concatlast =="4.0CC") | (concatlast =="4.0DB") | (concatlast =="4.0EA")]= 2.51
    
        
        col = np.zeros(self.catPort.shape)
        
        col[(concatlast =="1.0A")]= 1
        col[(concatlast =="2.0BD") | (concatlast =="1.0DD") | (concatlast =="2.0B") | (concatlast =="1.0B") | \
            (concatlast =="2.0AD") | (concatlast =="1.0BD") | (concatlast =="3.0A") | (concatlast =="3.0AD") | \
                (concatlast =="1.0BD")]= 1.88
        col[(concatlast =="1.0ED") | (concatlast =="1.0EC") | (concatlast =="1.0DD") | (concatlast =="1.0DC") | \
            (concatlast =="1.0CD") | (concatlast =="2.0CC") | (concatlast =="1.0D") | (concatlast =="3.0BC") | \
                (concatlast =="2.0C") | (concatlast =="3.0BD") | (concatlast =="4.0AC") | (concatlast =="1.0C") | \
                    (concatlast =="3.0B") | (concatlast =="1.0CC") | (concatlast =="2.0ED") | (concatlast =="2.0EC") | \
                        (concatlast =="2.0DD") | (concatlast =="2.0DC") | (concatlast =="2.0CD")]= 2.75
        col[(concatlast =="4.0AB") | (concatlast =="4.0BB") | (concatlast =="3.0CB") | (concatlast =="1.0DB") | \
            (concatlast =="3.0CC") | (concatlast =="3.0C") | (concatlast =="3.0CD") | (concatlast =="3.0EC") | \
                (concatlast =="3.0DB") | (concatlast =="3.0ED") | (concatlast =="3.0DC") | (concatlast =="3.0DD") | \
                    (concatlast =="3.0EB") | (concatlast =="4.0BC") | (concatlast =="4.0CB") | (concatlast =="4.0CC") | \
                        (concatlast =="4.0DB")]= 3.38
        
        susX2 = np.zeros(self.catPort.shape)
        susX2 = susX2.reshape(-1)
        susY2 = np.zeros(self.catPort.shape)
        susY2 = susY2.reshape(-1)
            
        row = row.reshape(-1)
        p1 = self.p1.reshape(-1)
        susX = np.where((row != 0 ), p1 - row, susX2)
        col = col.reshape(-1)
        p2 = self.p2.reshape(-1)
        susY = np.where((col != 0), p2 - col, susY2) 
        squareX = np.square(susX)
        squareY = np.square(susY)
        sumsquare = squareX + squareY
        finalsquare = np.sqrt(sumsquare)
        
        distancetrame = finalsquare
        
        return [nexttramo, distancetrame]
    
    """
    def distance(self, initial_pF):
        
        #Returns
        #-------
        #distancetrame : Distance to next trame
    
             
        
        concat = self.catPort +  self.catPrueba
        concat = concat.reshape(-1)
        concat2fg = [str(i) for i in initial_pF]
        arr = np.array(concat2fg)
        concatlast = arr + concat
        
        row = np.zeros(self.catPort.shape)
        
        row[(concatlast =="0.0A") | (concatlast =="0.0B") | (concatlast =="0.0C") | (concatlast =="0.0D") | \
            (concatlast =="1.0E")]= 1
        row[(concatlast =="1.0ED") | (concatlast =="1.0EC") | (concatlast =="1.0EB") | (concatlast =="1.0EA")]= 2
        row[(concatlast =="2.0DA") | (concatlast =="2.0DB") | (concatlast =="2.0D") | (concatlast =="1.0DA") | \
            (concatlast =="3.0D") | (concatlast =="2.0D") | (concatlast =="2.0ED") | (concatlast =="2.0EC") | \
                (concatlast =="2.0EB") | (concatlast =="2.0DD") | (concatlast =="2.0DC") | (concatlast =="3.0DA") | \
                    (concatlast =="3.0E") | (concatlast =="3.0DB") | (concatlast =="3.0EC") | (concatlast =="3.0DC") | \
                        (concatlast =="3.0ED") | (concatlast =="3.0DD") | (concatlast =="3.0EA") | (concatlast =="3.0EB")]= 2.26
        row[(concatlast =="3.0CA") | (concatlast =="4.0CA") | (concatlast =="4.0CB") | (concatlast =="4.0DA") | \
            (concatlast =="4.0CC") | (concatlast =="4.0DB") | (concatlast =="4.0EA")]= 2.51
    
        
        col = np.zeros(self.catPort.shape)
        
        col[(concatlast =="1.0A")]= 1
        col[(concatlast =="2.0BD") | (concatlast =="1.0DD") | (concatlast =="2.0B") | (concatlast =="1.0B") | \
            (concatlast =="2.0AD") | (concatlast =="1.0BD") | (concatlast =="3.0A") | (concatlast =="3.0AD") | \
                (concatlast =="1.0BD")]= 1.88
        col[(concatlast =="1.0ED") | (concatlast =="1.0EC") | (concatlast =="1.0DD") | (concatlast =="1.0DC") | \
            (concatlast =="1.0CD") | (concatlast =="2.0CC") | (concatlast =="1.0D") | (concatlast =="3.0BC") | \
                (concatlast =="2.0C") | (concatlast =="3.0BD") | (concatlast =="4.0AC") | (concatlast =="1.0C") | \
                    (concatlast =="3.0B") | (concatlast =="1.0CC") | (concatlast =="2.0ED") | (concatlast =="2.0EC") | \
                        (concatlast =="2.0DD") | (concatlast =="2.0DC") | (concatlast =="2.0CD")]= 2.75
        col[(concatlast =="4.0AB") | (concatlast =="4.0BB") | (concatlast =="3.0CB") | (concatlast =="1.0DB") | \
            (concatlast =="3.0CC") | (concatlast =="3.0C") | (concatlast =="3.0CD") | (concatlast =="3.0EC") | \
                (concatlast =="3.0DB") | (concatlast =="3.0ED") | (concatlast =="3.0DC") | (concatlast =="3.0DD") | \
                    (concatlast =="3.0EB") | (concatlast =="4.0BC") | (concatlast =="4.0CB") | (concatlast =="4.0CC") | \
                        (concatlast =="4.0DB")]= 3.38
        
        susX2 = np.zeros(self.catPort.shape)
        susX2 = susX2.reshape(-1)
        susY2 = np.zeros(self.catPort.shape)
        susY2 = susY2.reshape(-1)
            
        row = row.reshape(-1)
        p1 = self.p1.reshape(-1)
        susX = np.where((row != 0 ), p1 - row, susX2)
        col = col.reshape(-1)
        p2 = self.p2.reshape(-1)
        susY = np.where((col != 0), p2 - col, susY2) 
        squareX = np.square(susX)
        squareY = np.square(susY)
        sumsquare = squareX + squareY
        finalsquare = np.sqrt(sumsquare)
        
        distancetrame = finalsquare
        
        return distancetrame
    """


    def student_h(self, effort):
        """
        takes student initial HC and teacher effort to compute achievement

        return: student test score, where effort_low = 0

        """
        effort_m1 = np.zeros(effort.shape)
        effort_m = np.where(effort==1, 1, effort_m1)
        effort_h1 = np.zeros(effort.shape)
        effort_h = np.where((effort==2), 1, effort_h1)
    
        eps = np.random.randn(self.N)*self.param.betas[3]
        
        h = self.param.betas[0] + self.param.betas[1]*effort_m + self.param.betas[2]*effort_h + eps
        
        
        """
        h = self.param.betas[0] + self.param.betas[1]*effort[0] + \
                    self.param.betas[2]*effort[1] + \
                    self.param.betas[3]*np.square(effort[0]) + self.param.betas[4]*np.square(effort[1])+ eps
        
        
        h = self.param.betas[0] + self.param.betas[1]*effort + \
            self.param.betas[2]*np.square(effort) + eps
            
        """

        return h

    def t_test(self,effort):
        """
        takes initial scores, effort and experiencie

        returns: test scores and portfolio

        """
        
 
        p1v1_past = np.where(np.isnan(self.p1_0), 0, self.p1_0)
        p2v1_past = np.where(np.isnan(self.p2_0), 0, self.p2_0)
        
     
        p0_past = np.zeros(p1v1_past.shape)
        p0_past = np.where((p1v1_past == 0),p2v1_past, p0_past)
        p0_past = np.where((p2v1_past == 0),p1v1_past, p0_past)
        p0_past = np.where((p1v1_past != 0) & (p2v1_past != 0) ,(self.p1_0 + self.p2_0)/2, p0_past)
        p0_past = (p0_past-np.mean(p0_past))/np.std(p0_past)
        
        effort_m1 = np.zeros(effort.shape)
        effort_m = np.where(effort==1, 1, effort_m1)
        effort_h1 = np.zeros(effort.shape)
        effort_h = np.where((effort==2), 1, effort_h1)
        
        pb = []

           
        for j in range(2):
            
            pb.append(self.param.alphas[j][0] + \
                     self.param.alphas[j][1]*effort_m + self.param.alphas[j][2]*effort_h + \
                         self.param.alphas[j][3]*self.years/10 + self.param.alphas[j][5]*p0_past  + \
                             np.random.normal(0, self.param.alphas[j][4], p1v1_past.shape)) 
        
        
        pv1 = ((1/(1+np.exp(-pb[0]))) + (1/3))*3
        pv2 = ((1/(1+np.exp(-pb[1]))) + (1/3))*3

        p = [pv1, pv2]
                
        return p

    def utility(self, income, effort, h):
        """
        Takes states income, student achievement, and effort

        returns: utility in log income terms

        """
        
        effort_m1 = np.zeros(effort.shape)
        effort_m = np.where(effort==1, 1, effort_m1)
        effort_h1 = np.zeros(effort.shape)
        effort_h = np.where((effort==2), 1, effort_h1)
        
        U_rsl = np.log(income) + self.param.gammas[0]*effort_m + self.param.gammas[1]*effort_h + self.param.gammas[2]*np.square(h)/2

        return U_rsl



