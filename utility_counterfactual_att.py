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


class Count_att(Utility):
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
        
        
        
        # " RENTA BASE MÍNIMA NACIONAL "
        
        RBMNElemt = np.where((self.typeSchool==1),HvalueE*self.HOURS, RBMNElemt2)
        RBMNSecond = np.where((self.typeSchool==0),HvalueS*self.HOURS, RBMNSecond2)
        
                
        # " EXPERIENCE "
        
        bienniumtwoFalse = self.years/2
        biennium = np.floor(bienniumtwoFalse)
        biennium[biennium>15]=15
        
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
        
        
        ATDPinitial = np.where((initial_p_2==1) & (self.years > 1), (Proinitial/15)*(self.HOURS/full_contract)*biennium, ATDPinitial2)
        ATDPearly = np.where((initial_p_2==2) & (self.years > 3), (ProEarly/15)*(self.HOURS/full_contract)*biennium, ATDPearly2)
        ATDPadvanced = np.where((initial_p_2==3) & (self.years > 3), (Proadvanced/15)*(self.HOURS/full_contract)*biennium, ATDPadvanced2)
        ATDPadvancedfixed = np.where((initial_p_2==3) & (self.years > 3), (Proadvancedfixed)*(self.HOURS/full_contract), ATDPadvancedfixed2)
        ATDPexpert1 = np.where((initial_p_2==4) & (self.years > 7), (Proexpert1/15)*(self.HOURS/44)*biennium, ATDPexpert12)
        ATDPexpert1fixed = np.where((initial_p_2==4) & (self.years > 7), (Proexpert1fixed)*(self.HOURS/full_contract), ATDPexpert1fixed2)
        ATDPexpert2 = np.where((initial_p_2==5) & (self.years > 11), (Proexpert2/15)*(self.HOURS/full_contract)*biennium, ATDPexpert22)
        ATDPexpert2fixed = np.where((initial_p_2==5) & (self.years > 11), (Proexpert2fixed)*(self.HOURS/full_contract), ATDPexpert2fixed2)
        
        # " AEP (Teaching excellence)
        
        AcreditaTramoI = self.param.AEP[0]
        AcreditaTramoII = self.param.AEP[1]
        AcreditaTramoIII = self.param.AEP[2]
    
            
        # " Asignación de perfeccionamiento
        # " This is the new asignment
        
        AsigElemt = np.where(((self.typeSchool==1)), RBMNElemt*0.4*(biennium/15), AsigElemt2)
        AsigSecond = np.where((self.typeSchool==0), RBMNSecond*0.4*(biennium/15), AsigSecond2)
        #AsigElemt = np.where(((self.typeSchool==1)), RBMNElemt*0.4*(biennium/15), AsigElemt2)
        
        # " Asignación por docencia
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
        
        prioirtyap_aep1 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==7), (AcreditaTramoI*0.4), prioirtyap_aep11)
        prioirtyap_aep2 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==8), (AcreditaTramoII*0.4), prioirtyap_aep22)
        prioirtyap_aep3 = np.where((self.AEP_priority >= 0.6) & (initial_p_aep==9), (AcreditaTramoIII*0.4), prioirtyap_aep33)
        priorityaep = sum([prioirtyap_aep1,prioirtyap_aep2,prioirtyap_aep3])
        
        
        # " Locality assignation
        
        localAssig_1 = np.where((self.typeSchool == 1), (self.locality*RBMNElemt/100), localAssig2)
        localAssig_0 = np.where((self.typeSchool == 0), (self.locality*RBMNSecond/100), localAssig3)

    
            
        # " SUM OF TOTAL SALARY "
        
        salary1 = np.where((initial_p_2==1) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPinitial,prioirtyap,localAssig_1]), salary2d)
        #salary2 = np.where((initial_p_2==1) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPinitial,localAssig_1]), salary2d)
        salary3 = np.where((initial_p_2==1) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPinitial,prioirtyap,localAssig_0]), salary2d)
        #salary4 = np.where((initial_p_2==1) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPinitial,localAssig_0]), salary2d)
        
        salary5 = np.where((initial_p_2==2) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPearly,prioirtyap,localAssig_1]), salary2d)
        #salary6 = np.where((initial_p_2==2) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPearly,localAssig_1]), salary2d)
        salary7 = np.where((initial_p_2==2) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPearly,prioirtyap,localAssig_0]), salary2d)
        #salary8 = np.where((initial_p_2==2) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPearly,localAssig_0]), salary2d)
        
        salary9 = np.where((initial_p_2==3) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_1]), salary2d)
        #salary10 = np.where((initial_p_2==3) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPadvanced,ATDPadvancedfixed,localAssig_1]), salary2d)
        salary11 = np.where((initial_p_2==3) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_0]), salary2d)
        #salary12 = np.where((initial_p_2==3) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPadvanced,ATDPadvancedfixed,localAssig_0]), salary2d)
        
        salary13 = np.where((initial_p_2==4) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_1]), salary2d)
        #salary14 = np.where((initial_p_2==4) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert1,ATDPexpert1fixed,localAssig_1]), salary2d)
        salary15 = np.where((initial_p_2==4) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_0]), salary2d)
        #salary16 = np.where((initial_p_2==4) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert1,ATDPexpert1fixed,localAssig_0]), salary2d)
        
        salary17 = np.where((initial_p_2==5) & (self.treatment == 1) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_1]), salary2d)
        #salary18 = np.where((initial_p_2==5) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert2,ATDPexpert2fixed,localAssig_1]), salary2d)
        salary19 = np.where((initial_p_2==5) & (self.treatment == 1) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_0]), salary2d)

        salary = sum([salary1,salary3,salary5,salary7,salary9,salary11,salary13,salary15,salary17,salary19])

        ####Control group: not evaluated yet, salary is fixed at baseline##
                
        initial_p_base = self.initial()
        
        ATDPinitial = np.where((initial_p_base==1) & (self.years > 1), (Proinitial/15)*(self.HOURS/full_contract)*biennium, ATDPinitial2)
        ATDPearly = np.where((initial_p_base==2) & (self.years > 3), (ProEarly/15)*(self.HOURS/full_contract)*biennium, ATDPearly2)
        ATDPadvanced = np.where((initial_p_base==3) & (self.years > 3), (Proadvanced/15)*(self.HOURS/full_contract)*biennium, ATDPadvanced2)
        ATDPadvancedfixed = np.where((initial_p_base==3) & (self.years > 3), (Proadvancedfixed)*(self.HOURS/full_contract), ATDPadvancedfixed2)
        ATDPexpert1 = np.where((initial_p_base==4) & (self.years > 7), (Proexpert1/15)*(self.HOURS/44)*biennium, ATDPexpert12)
        ATDPexpert1fixed = np.where((initial_p_base==4) & (self.years > 7), (Proexpert1fixed)*(self.HOURS/full_contract), ATDPexpert1fixed2)
        ATDPexpert2 = np.where((initial_p_base==5) & (self.years > 11), (Proexpert2/15)*(self.HOURS/full_contract)*biennium, ATDPexpert22)
        ATDPexpert2fixed = np.where((initial_p_base==5) & (self.years > 11), (Proexpert2fixed)*(self.HOURS/full_contract), ATDPexpert2fixed2)
        
        prioirtyap1 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_base==1), ((ATDPinitial+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap11)
        prioirtyap2 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_base==2), (((ATDPearly+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0])), prioirtyap22)
        prioirtyap3 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_base==3), ((ATDPadvanced+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap33)
        prioirtyap4 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_base==4), ((ATDPexpert1+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap44)
        prioirtyap5 = np.where(((self.priotity >= 0.6) & (self.priotity < 0.8)) & (initial_p_base==5), ((ATDPexpert2+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[0]), prioirtyap55)
        prioirtyap51 = np.where((self.priotity >= 0.8) & (initial_p_2==1), ((ATDPinitial+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[1]), prioirtyap661)
        prioirtyap52 = np.where((self.priotity >= 0.8) & (initial_p_2==2), ((ATDPearly+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[1]), prioirtyap771)
        prioirtyap6 = np.where((self.priotity >= 0.8) & (initial_p_2==3), ((ATDPadvanced+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[2]), prioirtyap66)
        prioirtyap7 = np.where((self.priotity >= 0.8) & (initial_p_2==4), ((ATDPexpert1+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[2]), prioirtyap77)
        prioirtyap8 = np.where((self.priotity >= 0.8) & (initial_p_2==5), ((ATDPexpert2+ExpTrameE)*0.2)+((self.HOURS/full_contract)*self.param.priori[2]), prioirtyap88)
        prioirtyap91 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_base==1), (ATDPinitial*0.1), prioirtyap991)
        prioirtyap92 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_base==2), (ATDPearly*0.1), prioirtyap992)
        prioirtyap93 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_base==3), (ATDPadvanced*0.1), prioirtyap993)
        prioirtyap94 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_base==4), (ATDPexpert1*0.1), prioirtyap994)
        prioirtyap95 = np.where(((self.priotity >= 0.45) & (self.priotity < 0.6))  & (self.rural_rbd==1) & (initial_p_base==5),(ATDPexpert2*0.1), prioirtyap995)
        
        prioirtyap = sum([prioirtyap1,prioirtyap2,prioirtyap3,prioirtyap4,prioirtyap5,prioirtyap51,prioirtyap52\
                          ,prioirtyap6,prioirtyap7,prioirtyap8,prioirtyap91,prioirtyap92,prioirtyap93,prioirtyap94,prioirtyap95])


        salary11 = np.where((initial_p_base==1) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPinitial,prioirtyap,localAssig_1]), salary2d)
        #salary2 = np.where((initial_p_2==1) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPinitial,localAssig_1]), salary2d)
        salary31 = np.where((initial_p_base==1) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPinitial,prioirtyap,localAssig_0]), salary2d)
        #salary4 = np.where((initial_p_2==1) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPinitial,localAssig_0]), salary2d)
        
        salary51 = np.where((initial_p_base==2) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPearly,prioirtyap,localAssig_1]), salary2d)
        #salary6 = np.where((initial_p_2==2) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPearly,localAssig_1]), salary2d)
        salary71 = np.where((initial_p_base==2) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPearly,prioirtyap,localAssig_0]), salary2d)
        #salary8 = np.where((initial_p_2==2) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPearly,localAssig_0]), salary2d)
        
        salary91 = np.where((initial_p_base==3) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_1]), salary2d)
        #salary10 = np.where((initial_p_2==3) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPadvanced,ATDPadvancedfixed,localAssig_1]), salary2d)
        salary111 = np.where((initial_p_base==3) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPadvanced,ATDPadvancedfixed,prioirtyap,localAssig_0]), salary2d)
        #salary12 = np.where((initial_p_2==3) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPadvanced,ATDPadvancedfixed,localAssig_0]), salary2d)
        
        salary131 = np.where((initial_p_base==4) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_1]), salary2d)
        #salary14 = np.where((initial_p_2==4) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert1,ATDPexpert1fixed,localAssig_1]), salary2d)
        salary151 = np.where((initial_p_base==4) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert1,ATDPexpert1fixed,prioirtyap,localAssig_0]), salary2d)
        #salary16 = np.where((initial_p_2==4) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert1,ATDPexpert1fixed,localAssig_0]), salary2d)
        
        salary171 = np.where((initial_p_base==5) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_1]), salary2d)
        #salary18 = np.where((initial_p_2==5) & (self.treatment == 0) & (self.typeSchool == 1), sum([RBMNElemt,ExpTrameE,BRP,ExpTrameE,ATDPexpert2,ATDPexpert2fixed,localAssig_1]), salary2d)
        salary191 = np.where((initial_p_base==5) & (self.treatment == 0) & (self.typeSchool == 0), sum([RBMNSecond,ExpTrameS,BRP,ExpTrameS,ATDPexpert2,ATDPexpert2fixed,prioirtyap,localAssig_0]), salary2d)

            

            
        salary_pr = sum([salary11,salary31,salary51,salary71,salary91,salary111,salary131,salary151,salary171,salary191])


        return [salary, salary_pr]
