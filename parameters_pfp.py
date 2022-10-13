
"""
Parameters class. Defines set of parameters for the cPFP ounterfactual experiment
"""

class Parameters:
    """

	List of structural parameters and prices

	"""
    def __init__(self,alphas,betas,gammas,hw,porc,pro,pol,AEP,priori,cutoffs_min,cutoffs_max):
        self.alphas,self.betas,self.gammas,self.hw,self.porc,self.pro,self.pol,self.AEP,self.priori= alphas,betas,gammas,hw,porc,pro,pol,AEP,priori
        self.cutoffs_min, self.cutoffs_max = cutoffs_min,cutoffs_max
        
        
