
"""
Parameters class. Defines set of parameters
"""

class Parameters:
    """

	List of structural parameters and prices

	"""
    def __init__(self,alphas,betas,gammas,alphas_control,betas_control,hw,porc,pro,pol,AEP,priori):
        self.alphas,self.betas,self.gammas,self.alphas_control,self.betas_control,self.hw,self.porc,self.pro,self.pol,self.AEP,self.priori= alphas,betas,gammas,alphas_control,betas_control,hw,porc,pro,pol,AEP,priori
