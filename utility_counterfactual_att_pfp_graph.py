# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:24:22 2023

@author: Patricio De Araya
"""

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


class Count_att_2_pfp_2(Utility):
    """ 

    This class modifies the economic environment of the agent

    """

    def __init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, 
                 catPort, catPrueba, TrameI,priotity, rural_rbd, locality, AEP_priority,a,b):
        """
        Calling baseline model

        """
        
        self.a = a
        self.b = b
        
        Utility.__init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, 
                 catPort, catPrueba, TrameI,priotity, rural_rbd, locality, AEP_priority)




    def placement(self,tscores,initial_p):

        # *I want to replicate the typecasting of the teachers to the tramo
        # puntajeportafolio := p1
        # Puntajepruebaconocimientosespecí := self.p2
        # The tranche nameXY is defined how X: row nad Y:column
        # We have {'INICIAL': 1, 'TEMPRANO': 2, 'AVANZADO': 3 , 'EXPERTO I': 4, 'EXPERTO II': 5}

        # Inicializar vector initial_p
        #initial_p = np.array(['']*(len(self.p1)))
        placementF = np.zeros(self.p1.shape[0])
        placementF_aep = np.zeros(self.p1.shape[0])
        #placement_corr = np.zeros(self.p1.shape[0])

            
        #The following will not be used in this counterfactual
        #initial placement 1
        placementF[(initial_p == 1) & (tscores[0] <= 1.99)] = 1
        placementF[(initial_p == 1) & ((tscores[0] > 1.99) & (tscores[0] <= 2.25)) & (tscores[1] <= 2.74) ] = 1
        placementF[(initial_p == 1) & ((tscores[0] > 1.99) & (tscores[0] <= 2.25)) & (tscores[1] > 2.74) ] = 2
        placementF[(initial_p == 1) & ((tscores[0] > 2.25) & (tscores[0] <= 2.5)) & (tscores[1] <= 1.87) ] = 1
        placementF[(initial_p == 1) & ((tscores[0] > 2.25) & (tscores[0] <= 2.5)) & ((tscores[1] > 1.87) & (tscores[1] <= 2.74)) ] = 2
        placementF[(initial_p == 1) & ((tscores[0] > 2.25) & (tscores[0] <= 2.5)) & (tscores[1] > 2.74)  ] = 3
        placementF[(initial_p == 1) & ((tscores[0] > 2.5) & (tscores[0] <= 3)) & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 1) & ((tscores[0] > 2.5) & (tscores[0] <= 3)) & (tscores[1] > 1.87) ] = 3
        placementF[(initial_p == 1) & (tscores[0] > 3)  & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 1) & (tscores[0] > 3)  & (tscores[1] > 1.87) ] = 3
        
        #initial placement 2
        placementF[(initial_p == 2) & (tscores[0] <= 2.25)] = 2
        placementF[(initial_p == 2) & ((tscores[0] > 2.25) & (tscores[0] <= 2.5)) & (tscores[1] <= 2.74)] = 2
        placementF[(initial_p == 2) & ((tscores[0] > 2.25) & (tscores[0] <= 2.5)) & (tscores[1] > 2.74) ] = 3
        placementF[(initial_p == 2) & ((tscores[0] > 2.5) & (tscores[0] <= 3.0)) & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 2) & ((tscores[0] > 2.5) & (tscores[0] <= 3.0)) & (tscores[1] > 1.87) ] = 3
        placementF[(initial_p == 2) & (tscores[0] > 3.0)  & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 2) & (tscores[0] > 3.0)  & (tscores[1] > 1.87) ] = 3
        
        #initial placement 3
        placementF[(initial_p == 3) & (tscores[0] <= 2.25)] = 3
        placementF[(initial_p == 3) & ((tscores[0] > 2.25) & (tscores[0] <= 2.5)) & (tscores[1] <= 3.38)] = 3
        placementF[(initial_p == 3) & ((tscores[0] > 2.25) & (tscores[0] <= 2.5)) & (tscores[1] > 3.38)] = 4
        placementF[(initial_p == 3) & ((tscores[0] > 2.5) & (tscores[0] <= 3.0)) & (tscores[1] <= 2.74)] = 3
        placementF[(initial_p == 3) & ((tscores[0] > 2.5) & (tscores[0] <= 3.0)) & (tscores[1] > 2.74)] = 4
        placementF[(initial_p == 3) & (tscores[0] > 3.0) & (tscores[1] <= 1.87)] = 3
        placementF[(initial_p == 3) & (tscores[0] > 3.0) & (tscores[1] > 1.87)] = 4
        
       #initial placement 4
        placementF[(initial_p == 4) & (tscores[0] <= 2.5)] = 4
        placementF[(initial_p == 4) & ((tscores[0] > 2.5) & (tscores[0] <= 3.0)) & (tscores[1] <= 3.38)] = 4
        placementF[(initial_p == 4) & ((tscores[0] > 2.5) & (tscores[0] <= 3.0)) & (tscores[1] > 3.38)] = 5
        placementF[(initial_p == 4) & (tscores[0] > 3.0) & (tscores[1] <= 2.74)] = 4
        placementF[(initial_p == 4) & (tscores[0] > 3.01) & (tscores[1] > 2.74)] = 5
        
        #initial placement 5
        placementF[(initial_p == 5)] = 5
                
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

        
        

        #return [placementF,placement_corr,placementF_aep]
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
        
        initial_p_2 = initial_p[0].copy()
        initial_p_aep = initial_p[1].copy()
       
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
        
        RBMNElemt = np.where((self.typeSchool==1),HvalueE*self.HOURS, RBMNElemt2)
        RBMNSecond = np.where((self.typeSchool==0),HvalueS*self.HOURS, RBMNSecond2)
        
                
        # " EXPERIENCE (4 years)"
        bienniumtwoFalse = self.years/2
        biennium = np.floor(bienniumtwoFalse)
        biennium[biennium>15]=15
        
        porc1 = self.param.porc[0] 
        porc2 = self.param.porc[1]
        
        ExpTrameE = np.where((self.typeSchool==1) & (self.years > 2), (porc1 + (porc2 * (biennium - 1))) * RBMNElemt, ExpTrameE2)
        ExpTrameS = np.where((self.typeSchool==0) & (self.years > 2), (porc1 + (porc2 * (biennium - 1))) * RBMNSecond, ExpTrameS2)
        
   
        
        # " VOCATIONAL RECOGNITION BONUS (BRP) (4 years) "
        # We assume that every teacher have
        # degree and mention.
    
        professional_qualificationW = self.param.pro[0] 
        professional_mentionW = self.param.pro[1]
        professional_qualification = self.param.pro[2]
        professional_mention = self.param.pro[3]
        full_contract = 44
        
        BRP = np.where((self.typeSchool==1) | (self.typeSchool==0),(professional_qualification + professional_mention)*(self.HOURS/full_contract),BRP2)
        BRPWithout = np.where((self.typeSchool==1) | (self.typeSchool==0),(professional_qualificationW + professional_mentionW)*(self.HOURS/full_contract),BRPWithout2)
    
    
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

        ATDPinitial = (Proinitial/15)*(self.HOURS/full_contract)*biennium
        ATDPearly = (ProEarly/15)*(self.HOURS/full_contract)*biennium
        ATDPadvanced = (Proadvanced/15)*(self.HOURS/full_contract)*biennium
        ATDPadvancedfixed = (Proadvancedfixed)*(self.HOURS/full_contract)
        ATDPexpert1 = (Proexpert1/15)*(self.HOURS/44)*biennium
        ATDPexpert1fixed = (Proexpert1fixed)*(self.HOURS/full_contract)
        ATDPexpert2 = (Proexpert2/15)*(self.HOURS/full_contract)*biennium
        ATDPexpert2fixed = (Proexpert2fixed)*(self.HOURS/full_contract)

        
        
        # " AEP (Teaching excellence) (4 years)
        
        AcreditaTramoI = self.param.AEP[0]
        AcreditaTramoII = self.param.AEP[1]
        AcreditaTramoIII = self.param.AEP[2]
    
            
        # " Asignación de perfeccionamiento
        # " This is the new asignment
        AsigElemt = np.where(((self.typeSchool==1)), RBMNElemt*0.4*(biennium/15), AsigElemt2)
        AsigSecond = np.where((self.typeSchool==0), RBMNSecond*0.4*(biennium/15), AsigSecond2)

        
        # " Asignación por docencia
        # "Post-reform Priority allocation

                # "Post-reform Priority allocation 
        
        prioirtyap1 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_2==1), ((ATDPinitial+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap11)
        prioirtyap2 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_2==2), (((ATDPearly+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0])), prioirtyap22)
        prioirtyap3 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_2==3), ((ATDPadvanced+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap33)
        prioirtyap4 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_2==4), ((ATDPexpert1+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap44)
        prioirtyap5 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_2==5), ((ATDPexpert2+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap55)
        prioirtyap51 = np.where((self.priotity >= 0.8) & (initial_p_2==1), ((ATDPinitial+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[1]), prioirtyap661)
        prioirtyap52 = np.where((self.priotity >= 0.8) & (initial_p_2==2), ((ATDPearly+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[1]), prioirtyap771)
        prioirtyap6 = np.where((self.priotity >= 0.8) & (initial_p_2==3), ((ATDPadvanced+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[2]), prioirtyap66)
        prioirtyap7 = np.where((self.priotity >= 0.8) & (initial_p_2==4), ((ATDPexpert1+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[2]), prioirtyap77)
        prioirtyap8 = np.where((self.priotity >= 0.8) & (initial_p_2==5), ((ATDPexpert2+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[2]), prioirtyap88)
        prioirtyap91 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_2==1), (ATDPinitial*0.1), prioirtyap991)
        prioirtyap92 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_2==2), (ATDPearly*0.1), prioirtyap992)
        prioirtyap93 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_2==3), (ATDPadvanced*0.1), prioirtyap993)
        prioirtyap94 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_2==4), (ATDPexpert1*0.1), prioirtyap994)
        prioirtyap95 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_2==5),(ATDPexpert2*0.1), prioirtyap995)
        prioirtyap = sum([prioirtyap1,prioirtyap2,prioirtyap3,prioirtyap4,prioirtyap5,prioirtyap51,prioirtyap52\
                          ,prioirtyap6,prioirtyap7,prioirtyap8,prioirtyap91,prioirtyap92,prioirtyap93,prioirtyap94,prioirtyap95])
        
        # "Pre-reform Priority allocation
        
        # "Pre-reform Priority allocation (4 years)
        
        prioirtyap_aep1 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==7), (AcreditaTramoI*0.4), prioirtyap_aep11)
        prioirtyap_aep2 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==8), (AcreditaTramoII*0.4), prioirtyap_aep22)
        prioirtyap_aep3 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==9), (AcreditaTramoIII*0.4), prioirtyap_aep33)
        priorityaep = sum([prioirtyap_aep1,prioirtyap_aep2,prioirtyap_aep3])
        
        
        # " Locality assignation
        
        localAssig_1 = np.where((self.typeSchool == 1), (self.locality*RBMNElemt/100), localAssig2)
        localAssig_0 = np.where((self.typeSchool == 0), (self.locality*RBMNSecond/100), localAssig3)

    
            
        #Treated: a linear pay-for-performance salary        

        salary = np.zeros(initial_p_2.shape[0])
        
        salary[(self.treatment == 1)] = self.a + self.b*(tscores[0] + tscores[1])/2

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
        
            
        salary_pr = sum([salary21,salary22,salary23,salary24,salary25,salary26,salary27,salary28,salary29,salary30])
        
        #This is salary post-reform, for -2018 teachers

        #getting 4-year average salary
        return [salary, salary_pr]

