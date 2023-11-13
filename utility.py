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

    def __init__(self, param, N, p1_0, p2_0, years, treatment, typeSchool, HOURS, p1, p2, catPort, catPrueba, 
                 TrameI, priotity, rural_rbd, locality, AEP_priority):
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
        self.priotity = priotity
        self.rural_rbd = rural_rbd
        self.locality = locality
        self.AEP_priority = AEP_priority
    
    def initial(self):
        
        
        initial_p = np.zeros(self.p1.shape[0])
        
        initial_p[(self.TrameI=='INICIAL')] = 1
        initial_p[(self.TrameI=='TEMPRANO')] = 2
        initial_p[(self.TrameI=='AVANZADO')] = 3
        initial_p[(self.TrameI=='EXPERTO I')] = 4
        initial_p[(self.TrameI=='EXPERTO II')] = 5
        
        #concat = self.catPort +  self.catPrueba
        #concat = concat.reshape(-1)
        #concat2fg = [str(i) for i in initial_p]
        #arr = np.array(concat2fg)
        #concatlast = arr + concat
        
        return initial_p
        
              
                
        
    def placement(self,ttscores,initial_p):

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

        tscores = ttscores[0]*self.treatment + ttscores[1]*(1 - self.treatment)
        
        
        #initial placement 1
        placementF[(initial_p == 1) & (tscores[0] <= 1.99)] = 1
        placementF[(initial_p == 1) & ((tscores[0] >= 2.00) & (tscores[0] <= 2.25)) & (tscores[1] <= 2.74) ] = 1
        placementF[(initial_p == 1) & ((tscores[0] >= 2.00) & (tscores[0] <= 2.25)) & (tscores[1] >= 2.75) ] = 2
        placementF[(initial_p == 1) & ((tscores[0] >= 2.26) & (tscores[0] <= 2.5)) & (tscores[1] <= 1.87) ] = 1
        placementF[(initial_p == 1) & ((tscores[0] >= 2.26) & (tscores[0] <= 2.5)) & ((tscores[1] >= 1.88) & (tscores[1] <= 2.74)) ] = 2
        placementF[(initial_p == 1) & ((tscores[0] >= 2.26) & (tscores[0] <= 2.5)) & (tscores[1] > 2.74)  ] = 3
        placementF[(initial_p == 1) & ((tscores[0] >= 2.51) & (tscores[0] <= 3)) & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 1) & ((tscores[0] >= 2.51) & (tscores[0] <= 3)) & (tscores[1] > 1.87) ] = 3
        placementF[(initial_p == 1) & (tscores[0] >= 3)  & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 1) & (tscores[0] >= 3)  & (tscores[1] > 1.87) ] = 3
        
        #initial placement 2
        placementF[(initial_p == 2) & (tscores[0] <= 2.25)] = 2
        placementF[(initial_p == 2) & ((tscores[0] >= 2.26) & (tscores[0] <= 2.5)) & (tscores[1] <= 2.74)] = 2
        placementF[(initial_p == 2) & ((tscores[0] >= 2.26) & (tscores[0] <= 2.5)) & (tscores[1] >= 2.75) ] = 3
        placementF[(initial_p == 2) & ((tscores[0] >= 2.51) & (tscores[0] <= 3.0)) & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 2) & ((tscores[0] >= 2.51) & (tscores[0] <= 3.0)) & (tscores[1] >= 1.88) ] = 3
        placementF[(initial_p == 2) & (tscores[0] >= 3.01)  & (tscores[1] <= 1.87) ] = 2
        placementF[(initial_p == 2) & (tscores[0] >= 3.01)  & (tscores[1] >= 1.88) ] = 3
        
        #initial placement 3
        placementF[(initial_p == 3) & (tscores[0] <= 2.25)] = 3
        placementF[(initial_p == 3) & ((tscores[0] >= 2.26) & (tscores[0] <= 2.5)) & (tscores[1] <= 3.38)] = 3
        placementF[(initial_p == 3) & ((tscores[0] >= 2.26) & (tscores[0] <= 2.5)) & (tscores[1] >= 3.98)] = 4
        placementF[(initial_p == 3) & ((tscores[0] >= 2.51) & (tscores[0] <= 3.0)) & (tscores[1] <= 2.74)] = 3
        placementF[(initial_p == 3) & ((tscores[0] >= 2.51) & (tscores[0] <= 3.0)) & (tscores[1] >= 2.75)] = 4
        placementF[(initial_p == 3) & (tscores[0] >= 3.01) & (tscores[1] <= 1.87)] = 3
        placementF[(initial_p == 3) & (tscores[0] >= 3.01) & (tscores[1] >= 1.88)] = 4
        
       #initial placement 4
        placementF[(initial_p == 4) & (tscores[0] <= 2.5)] = 4
        placementF[(initial_p == 4) & ((tscores[0] >= 2.51) & (tscores[0] <= 3.0)) & (tscores[1] <= 3.38)] = 4
        placementF[(initial_p == 4) & ((tscores[0] >= 2.51) & (tscores[0] <= 3.0)) & (tscores[1] >= 3.39)] = 5
        placementF[(initial_p == 4) & (tscores[0] >= 3.01) & (tscores[1] <= 2.74)] = 4
        placementF[(initial_p == 4) & (tscores[0] >= 3.01) & (tscores[1] >= 2.75)] = 5
        
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

        #Experience requirements
        placementF[(placementF == 2) & (self.years < 4) ] = 1
        placementF[(placementF == 3) & (self.years < 4) ] = 1
        placementF[(placementF == 4) & (self.years < 8) ] = 3
        placementF[(placementF == 5) & (self.years < 12) ] = 4
        

        #return [placementF,placement_corr,placementF_aep]
        return [placementF,placementF_aep]
    

    def income(self, initial_p):
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

    
            
        # " SUM OF TOTAL SALARY "

        
        salary1 = np.where((initial_p_2==1) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPinitial,prioirtyap,localAssig_1]), salary2d)
        salary3 = np.where((initial_p_2==1) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ATDPinitial,prioirtyap,localAssig_0]), salary2d)
        
        salary5 = np.where((initial_p_2==2) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPearly,prioirtyap,localAssig_1]), salary2d)
        salary7 = np.where((initial_p_2==2) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPearly,prioirtyap,localAssig_0]), salary2d)

        
        salary9 = np.where((initial_p_2==3) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_1]), salary2d)
        salary11 = np.where((initial_p_2==3) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_0]), salary2d)

        salary13 = np.where((initial_p_2==4) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_1]), salary2d)
        salary15 = np.where((initial_p_2==4) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_0]), salary2d)

        salary17 = np.where((initial_p_2==5) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRP,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_1]), salary2d)
        salary19 = np.where((initial_p_2==5) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRP,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_0]), salary2d)

        #pre-reform, experiencia only once
        
        salary21 = np.where((initial_p_aep==6) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt]), salary3d)
        salary22 = np.where((initial_p_aep==6) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond]), salary3d)
        
        salary23 = np.where((initial_p_aep==7) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,priorityaep,AcreditaTramoI]), salary3d)
        salary24 = np.where((initial_p_aep==7) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,priorityaep,AcreditaTramoI]), salary3d)
        
        salary25 = np.where((initial_p_aep==8) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,priorityaep,AcreditaTramoII]), salary3d)
        salary26 = np.where((initial_p_aep==8) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,priorityaep,AcreditaTramoII]), salary3d)
        
        salary27 = np.where((initial_p_aep==9) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,2*ExpTrameE,BRPWithout,AsigElemt,priorityaep,AcreditaTramoIII]), salary3d)
        salary28 = np.where((initial_p_aep==9) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,2*ExpTrameS,BRPWithout,AsigSecond,priorityaep,AcreditaTramoIII]), salary3d)
        
        
        #salary = sum([salary1,salary2,salary3,salary4,salary5,salary6,salary7,salary8,salary9,salary10, \
        #              salary11,salary12,salary13,salary14,salary15,salary16,salary17,salary18,salary19,salary20])
            
        salary = sum([salary1,salary3,salary5,salary7,salary9,salary11,salary13,salary15,salary17,salary19])
            
        salary_pr = sum([salary21,salary22,salary23,salary24,salary25,salary26,salary27,salary28])
        
        #This is salary post-reform, for -2018 teachers

        #getting 4-year average salary
        return [salary, salary_pr]
    
    
    def distance(self, initial_pF,tscores):
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
        #tramolet[(concatlast =="0.0A") | (concatlast =="0.0B") | (concatlast =="0.0C") | (concatlast =="0.0D") | \
                 #(concatlast =="0.0E")]= 154 #"INICIALED"
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
        
        #row[(concatlast =="0.0A") | (concatlast =="0.0B") | (concatlast =="0.0C") | (concatlast =="0.0D") | \
            #(concatlast =="1.0E")]= 1
        row[(concatlast =="1.0ED") | (concatlast =="1.0EC") | (concatlast =="1.0EB") | (concatlast =="1.0EA")]= 2
        row[(concatlast =="2.0DA") | (concatlast =="2.0DB") | (concatlast =="2.0D") | (concatlast =="1.0DA") | \
            (concatlast =="3.0D") | (concatlast =="2.0D") | (concatlast =="2.0ED") | (concatlast =="2.0EC") | \
                (concatlast =="2.0EB") | (concatlast =="2.0DD") | (concatlast =="2.0DC") | (concatlast =="3.0DA") | \
                    (concatlast =="3.0E") | (concatlast =="3.0DB") | (concatlast =="3.0EC") | (concatlast =="3.0DC") | \
                        (concatlast =="3.0ED") | (concatlast =="3.0DD") | (concatlast =="3.0EA") | (concatlast =="3.0EB")]= 2.26
        row[(concatlast =="3.0CA") | (concatlast =="4.0CA") | (concatlast =="4.0CB") | (concatlast =="4.0DA") | \
            (concatlast =="4.0CC") | (concatlast =="4.0DB") | (concatlast =="4.0EA") | (concatlast =="2.0CC") | \
                (concatlast =="3.0CB")]= 2.51
        row[(concatlast =="3.0BC") | (concatlast =="4.0BB")]= 3.01
    
        
        col = np.zeros(self.catPort.shape)
        
        #col[(concatlast =="1.0A")]= 1
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
        p1 = tscores[0].reshape(-1)
        susX = np.where((row != 0 ), p1 - row, susX2)
        col = col.reshape(-1)
        p2 = tscores[1].reshape(-1)
        susY = np.where((col != 0), p2 - col, susY2) 
        #squareX = np.square(susX)
        #squareY = np.square(susY)
        XY_distance2 = np.zeros(self.catPort.shape)
        XY_distance = np.where((susX != 0) & (susY != 0), (susX+susY)/2, np.where(susX != 0, susX, np.where(susY != 0, susY, XY_distance2))) 
        #sumsquare = squareX + squareY
        #finalsquare = np.sqrt(sumsquare)
        
        Rsquare = np.empty(self.catPort.shape)
        Rsquare[:] = np.nan
        Rsquare2 = np.where(tramolet != 6, XY_distance, Rsquare)
        
        #Rsquare_5 = pd.qcut(Rsquare2, 5, labels = False)
        
        return [nexttramo, Rsquare2, susX, susY]
    



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

        eps = np.random.randn(self.N)*self.param.betas_control[1]

        h_control =  self.param.betas_control[0] + self.param.betas[1]*effort_m + self.param.betas[2]*effort_h + \
            self.param.betas[4]*self.years/10  + eps
        

        return [h_treated,h_control]
    
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
        pb_control = []

           
        for j in range(2):
            
            shock = np.random.normal(0, self.param.alphas[j][4], p1v1_past.shape)
            
            pb_treated.append(self.param.alphas[j][0] + \
                     self.param.alphas[j][1]*effort_m + self.param.alphas[j][2]*effort_h + \
                         self.param.alphas[j][3]*self.years/10 + self.param.alphas[j][5]*p0_past  + \
                             shock)
            
            shock = np.random.normal(0, self.param.alphas_control[j][1], p1v1_past.shape)
            
            pb_control.append(self.param.alphas_control[j][0] + \
                     self.param.alphas[j][1]*effort_m + self.param.alphas[j][2]*effort_h + \
                         self.param.alphas[j][3]*self.years/10  + \
                             shock)              

          

        p_treated = [((1/(1+np.exp(-pb_treated[0]))) + (1/3))*3, ((1/(1+np.exp(-pb_treated[1]))) + (1/3))*3]
        p_control = [((1/(1+np.exp(-pb_control[0]))) + (1/3))*3, ((1/(1+np.exp(-pb_control[1]))) + (1/3))*3]
        

                
        return [p_treated,p_control]

    def utility(self, income, effort, h):
        """
        Takes states income, student achievement, and effort

        returns: utility in log income terms

        """
        
       
        d_effort_t1 = effort == 1
        d_effort_t2 = effort == 2
        d_effort_t3 = effort == 3
        
        effort_m = d_effort_t1 + d_effort_t3
        effort_h = d_effort_t2 + d_effort_t3
        
        income_aux = income[0]*self.treatment + income[1]*(1-self.treatment)

        simce = h[0]*self.treatment + h[1]*(1-self.treatment)
         
        U_rsl = np.log(income_aux) + self.param.gammas[0]*effort_m + self.param.gammas[1]*effort_h + self.param.gammas[2]*simce
        
        #mu_c = -0.5
        #ut_h = self.param.gammas[0]*effort_m + self.param.gammas[1]*effort_h
        #U_rsl = np.exp(ut_h)*((income_aux)**mu_c)/(mu_c) + self.param.gammas[2]*np.log(h)
        return U_rsl



