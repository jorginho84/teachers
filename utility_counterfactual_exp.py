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


class Count_2(Utility):
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
    def placement(self,tscores):

        # *I want to replicate the typecasting of the teachers to the tramo
        # puntajeportafolio := p1
        # Puntajepruebaconocimientosespecí := self.p2
        # The tranche nameXY is defined how X: row nad Y:column
        # We have {'INICIAL': 1, 'TEMPRANO': 2, 'AVANZADO': 3 , 'EXPERTO I': 4, 'EXPERTO II': 5}

        # Inicializar vector initial_p
        #initial_p = np.array(['']*(len(self.p1)))
        placementF = np.zeros(self.p1.shape[0])
        placementF_aep = np.zeros(self.p1.shape[0])
        # " Treatment "
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
        
        # " Control: AEP "
        placementF_aep[(tscores[0]<2) & ((tscores[1]>=1) & (tscores[1] <= 1.87)) & (self.treatment == 0)]=6
        placementF_aep[(tscores[0]<2) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74)) & (self.treatment == 0)]=6
        placementF_aep[(tscores[0]<2) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38)) & (self.treatment == 0)]=6
        placementF_aep[(tscores[0]<2) & ((tscores[1]> 3.38) & (tscores[1] <= 4)) & (self.treatment == 0)]=6
        placementF_aep[((tscores[0]>=2) & (tscores[0]<=2.5)) & ((tscores[1]>=1) & (tscores[1] <= 1.87)) & (self.treatment == 0)]=6 
        placementF_aep[((tscores[0]>=2) & (tscores[0]<=2.5)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74)) & (self.treatment == 0)]=6
        placementF_aep[((tscores[0]>=2) & (tscores[0]<=2.5)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38)) & (self.treatment == 0)]=7
        placementF_aep[((tscores[0]>=2) & (tscores[0]<=2.5)) & ((tscores[1]> 3.38) & (tscores[1] <= 4)) & (self.treatment == 0)]=8
        placementF_aep[((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]>=1) & (tscores[1] <= 1.87)) & (self.treatment == 0)]=6
        placementF_aep[((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74)) & (self.treatment == 0)]=7
        placementF_aep[((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38)) & (self.treatment == 0)]=8
        placementF_aep[((tscores[0]>2.5) & (tscores[0]<=3)) & ((tscores[1]> 3.38) & (tscores[1] <= 4)) & (self.treatment == 0)]=9
        placementF_aep[((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]>=1) & (tscores[1] <= 1.87)) & (self.treatment == 0)]=6
        placementF_aep[((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 1.87) & (tscores[1] <= 2.74)) & (self.treatment == 0)]=8
        placementF_aep[((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 2.74) & (tscores[1] <= 3.38)) & (self.treatment == 0)]=9
        placementF_aep[((tscores[0]>3) & (tscores[0]<=4)) & ((tscores[1]> 3.38) & (tscores[1] <= 4)) & (self.treatment == 0)]=9
        np.warnings.filterwarnings('ignore')
        

        return [placementF,placementF_aep]
        


        
    
    def income(self, initial_p,tscores):
        """
        takes treatment dummy, teach test scores, 
        initial placement, and self.years of experience

        """
        # " WE SIMULATE A PROFESSOR WITH 44 HOURS "
        # " Values in dolars "
        HvalueE = self.param.hw[0]
        HvalueS = self.param.hw[1]
        
        initial_p = initial_p[0].copy()
        initial_p_aep = initial_p[1].copy()
        
        
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
        AsigElemt2 = np.zeros(initial_p.shape[0])
        AsigSecond2 = np.zeros(initial_p.shape[0])
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
        
        
        ATDPinitial = np.where((initial_p==1) , (Proinitial/15)*(self.HOURS/full_contract)*biennium, ATDPinitial2)
        ATDPearly = np.where((initial_p==2) , (ProEarly/15)*(self.HOURS/full_contract)*biennium, ATDPearly2)
        ATDPadvanced = np.where((initial_p==3) , (Proadvanced/15)*(self.HOURS/full_contract)*biennium, ATDPadvanced2)
        ATDPadvancedfixed = np.where((initial_p==3), (Proadvancedfixed)*(self.HOURS/full_contract), ATDPadvancedfixed2)
        ATDPexpert1 = np.where((initial_p==4), (Proexpert1/15)*(self.HOURS/44)*biennium, ATDPexpert12)
        ATDPexpert1fixed = np.where((initial_p==4), (Proexpert1fixed)*(self.HOURS/full_contract), ATDPexpert1fixed2)
        ATDPexpert2 = np.where((initial_p==5) , (Proexpert2/15)*(self.HOURS/full_contract)*biennium, ATDPexpert22)
        ATDPexpert2fixed = np.where((initial_p==5) , (Proexpert2fixed)*(self.HOURS/full_contract), ATDPexpert2fixed2)
        
        # " AEP (Teaching excellence)
        
        AcreditaTramoI = self.param.AEP[0]
        AcreditaTramoII = self.param.AEP[1]
        AcreditaTramoIII = self.param.AEP[2]
    
            
        # " Asignación de perfeccionamiento
        # " This is the new asignment
        
        AsigElemt = np.where(((self.typeSchool==1)), RBMNElemt*0.4*(biennium/15), AsigElemt2)
        AsigSecond = np.where((self.typeSchool==0), RBMNSecond*0.4*(biennium/15), AsigSecond2)
        #AsigElemt = np.where(((self.typeSchool==1)), RBMNElemt*0.4*(biennium/15), AsigElemt2)
        
    
            
        # " SUM OF TOTAL SALARY "
        
        salary1 = np.where((initial_p==1) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPinitial]), salary2d)
        salary2 = np.where((initial_p==1) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPinitial]), salary2d)
        
        salary3 = np.where((initial_p==2) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPearly]), salary2d)
        salary4 = np.where((initial_p==2) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPearly]), salary2d)
        
        salary5 = np.where((initial_p==3) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPadvanced,ATDPadvancedfixed]), salary2d)
        salary6 = np.where((initial_p==3) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPadvanced,ATDPadvancedfixed]), salary2d)
        
        salary7 = np.where((initial_p==4) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert1,ATDPexpert1fixed]), salary2d)
        salary8 = np.where((initial_p==4) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert1,ATDPexpert1fixed]), salary2d)
        
        salary9 = np.where((initial_p==5) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert2,ATDPexpert2fixed]), salary2d)
        salary10 = np.where((initial_p==5) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert2,ATDPexpert2fixed]), salary2d)
        
        salary11 = np.where((initial_p_aep==6) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt]), salary2d)
        salary12 = np.where((initial_p_aep==6) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond]), salary2d)
        
        salary13 = np.where((initial_p_aep==7) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,AcreditaTramoI]), salary2d)
        salary14 = np.where((initial_p_aep==7) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,AcreditaTramoI]), salary2d)
        
        salary15 = np.where((initial_p_aep==8) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,AcreditaTramoII]), salary2d)
        salary16 = np.where((initial_p_aep==8) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,AcreditaTramoII]), salary2d)
        
        salary17 = np.where((initial_p_aep==9) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,AcreditaTramoIII]), salary2d)
        salary18 = np.where((initial_p_aep==9) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,AcreditaTramoIII]), salary2d)
        
        
        salary = sum([salary1,salary2,salary3,salary4,salary5,salary6,salary7,salary8,salary9,salary10, \
                      salary11,salary12,salary13,salary14,salary15,salary16,salary17,salary18])


        return salary

