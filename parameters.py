
"""
Parameters class. Defines set of parameters
"""

class Parameters:
    """

	List of structural parameters and prices

	"""
    def __init__(self,alphas,betas,gammas,hw,porc,pro,pol,AEP,priori):
        self.alphas,self.betas,self.gammas,self.hw,self.porc,self.pro,self.pol,self.AEP,self.priori= alphas,betas,gammas,hw,porc,pro,pol,AEP,priori
