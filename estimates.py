"""
This code creates a table with estimated parameters and their SEs


"""
import numpy as np
import pandas as pd
import pickle
import itertools
import sys, os
from scipy import stats
#from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from joblib import Parallel, delayed
from scipy import interpolate
import matplotlib.pyplot as plt
#sys.path.append("C:\\Users\\Jorge\\Dropbox\\Chicago\\Research\\Human capital and the household\]codes\\model")
sys.path.append("D:\Git\TeacherPrincipal")
sys.path.append("D:\Git\result")
#import gridemax
import time
#import int_linear
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
#import pybobyqa
#import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
import time

betas_opt = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/model/betas_v1.npy")

ses = np.load("/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/model/se_v1.npy")


#Estimated parameters
alphas = [[0.5,0.1,0.2,-0.01,0.1],
		[0.5,0.1,0.2,-0.01,0.1]]

se_alphas = [[0.5,0.1,0.2,-0.01,0.1],
		[0.5,0.1,0.2,-0.01,0.1]]

#betas = [100,0.9,0.9,-0.05,-0.05,20]
#Parámetros más importantes
#betas = [100,10,33,20]

betas = [-0.4,0.3,0.9,1]
se_betas = [-0.4,0.3,0.9,1]

gammas = [-0.1,-0.2,0.8]
se_gammas = [-0.1,-0.2,0.8]


###########.TEX table##################

utility_list_beta = [gammas[0],gammas[1],gammas[2]]
utility_list_se = [se_gammas[0],se_gammas[1],se_gammas[2]]
utility_names = [r'Medium-effort disutility',
r'High-effort disutility',
r'Preference for student achievment']


student_list_beta = [betas[0],betas[1],betas[2],betas[3]]
student_list_se = [se_betas[0],se_betas[1],se_betas[2],se_betas[3]]
student_names = [r'Constant',
r'Medium-effort productivity',
r'High-effort productivity',r'S.D. of shock']


test1_list_beta = [alpha[0][0],alpha[0][1],alpha[0][2],alpha[0][3],alpha[0][4]]
test1_list_se = [se_alpha[0][0],se_alpha[0][1],se_alpha[0][2],se_alpha[0][3],se_alpha[0][4]]
test1_names = [r'Constant',
r'Medium-effort productivity',
r'High-effort productivity',r'Experience',r'Variance of shock']

test2_list_beta = [alpha[1][0],alpha[1][1],alpha[1][2],alpha[1][3],alpha[1][4]]
test2_list_se = [se_alpha[1][0],se_alpha[1][1],se_alpha[1][2],se_alpha[1][3],se_alpha[1][4]]



#This is for the paper
with open('/Users/jorge-home/Dropbox/Research/teachers-reform/teachers/Results/model/estimates.tex','w') as f:
	f.write(r'\begin{tabular}{lcccc}'+'\n')
	f.write(r'\hline' + '\n')
	f.write(r'Parameter &  & Estimate & & S.E. \bigstrut\\' + '\n')
	f.write(r'\hline' + '\n')
	f.write(r'\emph{A. Utility function} &       &       &       &  \\' + '\n')
	for j in range(len(utility_list_beta)):
		f.write(utility_names[j]+r' &  &  '+ '{:04.3f}'.format(utility_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(utility_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{B. Student production function} &       &       &       &  \\' + '\n')
	for j in range(len(student_list_beta)):
		f.write(wage_names[j]+r' &  &  '+ '{:04.3f}'.format(student_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(student_list_seu[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{C. Portfolio production function} &       &       &       &  \\' + '\n')
	for j in range(len(test1_list_beta)):
		f.write(test1_names[j]+r' &  &  '+ '{:04.3f}'.format(test1_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(test1_list_se[j])+r' \\' + '\n')

	f.write(r' &       &       &       &  \\' + '\n')
	f.write(r'\emph{D. STEI production function} &       &       &       &  \\' + '\n')
	for j in range(len(test2_list_beta)):
		f.write(test1_names[j] + r' &  &  '+ '{:04.3f}'.format(test2_list_beta[j]) +
			r' &  & '+ '{:04.3f}'.format(test2_list_se[j])+r' \\' + '\n')


	f.write(r'\hline'+'\n')
	f.write(r'\end{tabular}' + '\n')
	f.close()