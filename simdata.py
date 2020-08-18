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

		initial_p = self.model.initial()

		teacher_scores = self.model.t_test(effort)

		placement = self.model.placement(teacher_scores)

		income = self.model.income(placement)

		simce = self.model.student_h(effort)

		return self.model.utility(income, effort, simce)

	def choice(self):
		"""
		computes optimal effort values. Maximizes util(effort).
	
		"""

		
