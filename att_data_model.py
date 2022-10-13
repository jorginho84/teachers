"""

exec(open("/home/jrodriguezo/teachers/codes/att_data_model.py").read())

This script compares the estimated ATT from the data and structural model.

"""


import numpy as np
import pandas as pd
import pickle
import tracemalloc
import itertools
import sys, os
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from tabulate import tabulate
from texttable import Texttable
import xlsxwriter
from openpyxl import Workbook 
from openpyxl import load_workbook
import time
#import int_linear
sys.path.append("/home/jrodriguezo/teachers/codes")
import utility as util
import parameters as parameters
import simdata as sd
import estimate as est
from scipy.stats import norm

#ver https://pythonspeed.com/articles/python-multiprocessing/
import multiprocessing as mp
from multiprocessing import Pool






np.random.seed(123)

betas_nelder = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v23.npy")
df = pd.read_stata('/home/jrodriguezo/teachers/data/data_main_regmain.dta')
moments_vector = np.load("/home/jrodriguezo/teachers/codes/moments.npy")
ses_opt = np.load('/home/jrodriguezo/teachers/codes/ses_model.npy')

n_sim = 100

w_matrix = np.zeros((ses_opt.shape[0],ses_opt.shape[0]))
for j in range(ses_opt.shape[0]):
    w_matrix[j,j] = ses_opt[j]**(-2)

n = df.shape[0]


#---This function delivers att data - att model---#

def data_model(j):
    
    np.random.seed(j+100)
    data_python = df.sample(n, replace=True)
    y = np.array(data_python['stdsimce'])
    x_1 = np.array(data_python['d_trat'])
    x_2 = np.array(data_python['d_year'])
    x_3 = np.array(data_python['inter'])
    x = np.transpose(np.array([x_1, x_2, x_3]))
    x = sm.add_constant(x)
    cov_drun = np.array(data_python['drun'])

    model_reg = sm.OLS(exog=x, endog=y)
    results = model_reg.fit(cov_type='cluster', cov_kwds={'groups': cov_drun}, use_t=True)

    att_data = results.params[3].round(8)

    data_python = data_python.loc[data_python['agno'].isin([2016,2017])]
    data_python = data_python[data_python['eval_year'] != 2017]
    data_python = data_python[data_python['experience'].notnull()]

    data_python_1 = data_python.drop(['stdsimce'], axis=1)
    data_python_1 = data_python_1.groupby(['drun']).first()

    data_python_2 = data_python[['drun','stdsimce']].groupby(['drun']).mean()

    data_python = pd.merge(data_python_1,data_python_2, on='drun')


    data_python['score_port'] = data_python['ptj_portafolio_rec2016']
    data_python['score_test'] = data_python['ptj_prueba_rec2016']
    data_python['cat_port'] = data_python['cat_portafolio_rec2016']
    data_python['cat_test'] = data_python['cat_prueba_rec2016']
    data_python['trame'] = data_python['tramo_rec2016']

    data_python.loc[data_python['eval_year'] != 1, 'trame'] = data_python.loc[data_python['eval_year'] != 1, 'tramo_a2016']

    data_python.rename(columns = {'ptj_portafolio_a2016':'score_port_past', 'ptj_prueba_a2016':'score_test_past'}, inplace = True)

    data_python = data_python[(data_python['score_port_past'].notnull()) & (data_python['score_test_past'].notnull())]


    data_python = data_python[data_python['stdsimce'].notnull()]
    data_python = data_python[data_python['experience'].notnull()]
    data_python = data_python[data_python['d_trat'].notnull()]
    data_python = data_python[data_python['eval_year'].notnull()]
    data_python = data_python[data_python['typeschool'].notnull()]
    data_python = data_python[data_python['por_priority'].notnull()]
    data_python = data_python[data_python['rural_rbd'].notnull()]
    data_python = data_python[data_python['AsignacionZona'].notnull()]


    """
    Obtains one set of estimates
    """
      

    # TREATMENT #
    #treatmentOne = rev[['d_trat']]
    treatment = np.array(data_python['d_trat'])
    # EXPERIENCE #
    #yearsOne = rev[['experience']]
    years = np.array(data_python['experience'])
    # SCORE PORTFOLIO #
    #p1_0_1 = rev[['score_port']]
    p1_0 = np.array(data_python['score_port'])
    p1 = np.array(data_python['score_port'])
    # SCORE TEST #
    #p2_0_1 = rev[['score_test']]
    p2_0 = np.array(data_python['score_test'])
    p2 = np.array(data_python['score_test'])
    # CATEGORY PORTFOLIO #
    #categPortfolio = rev[['cat_port']]
    catPort = np.array(data_python['cat_port'])
    # CATEGORY TEST #
    #categPrueba = data_python[['cat_test']]
    catPrueba = np.array(data_python['cat_test'])
    # TRAME #
    #TrameInitial = rev[['trame']]
    TrameI = np.array(data_python['trame'])
    # TYPE SCHOOL #
    #typeSchoolOne = data_python[['typeschool']]
    typeSchool = np.array(data_python['typeschool'])
    
    
    # Priority #
    priotity = np.array(data_python['por_priority'])
    
    rural_rbd = np.array(data_python['rural_rbd'])
    
    locality = np.array(data_python['AsignacionZona'])


    

    #### PARAMETERS MODEL ####
    N = np.size(p1_0)
    HOURS = np.array([44]*N)
    alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
          betas_nelder[3], betas_nelder[4]],
         [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
          betas_nelder[8], betas_nelder[9]]]
    
    betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12] ,betas_nelder[13],betas_nelder[14]]
    
    gammas = [betas_nelder[15],betas_nelder[16],betas_nelder[17]]
    dolar= 600
    value = [14403, 15155]
    hw = [value[0]/dolar,value[1]/dolar]
    porc = [0.0338, 0.0333]
    #inflation adjustment: 2012Jan-2019Dec: 1.266
    qualiPesos = [72100*1.266, 24034*1.266, 253076, 84360] 
    pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]
    progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]
    pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,
           progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]
    
    pri = [47872,113561]
    priori = [pri[0]/dolar, pri[1]/dolar]

    Asig = [150000*1.111,100000*1.111,50000*1.111]
    AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar] 

    param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)
        
    output_ins = est.estimate(N, years,param0, p1_0,p2_0,treatment, \
         typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI, priotity,rural_rbd,locality, \
         w_matrix,moments_vector)
    
    start_time = time.time()
    output_me = output_ins.optimizer()
    time_opt=time.time() - start_time
    print ('Done in')
    print("--- %s seconds ---" % (time_opt))

    beta_1 = output_me.x[0]
    beta_2 = output_me.x[1]
    beta_3 = output_me.x[2]
    beta_4 = np.exp(output_me.x[3])
    beta_5 = output_me.x[4]
    beta_6 = output_me.x[5]
    beta_7 = output_me.x[6]
    beta_8 = output_me.x[7]
    beta_9 = np.exp(output_me.x[8])
    beta_10 = output_me.x[9]
    beta_11 = output_me.x[10]
    beta_12 = output_me.x[11]
    beta_13 = output_me.x[12]
    beta_14 = output_me.x[13]
    beta_15 = output_me.x[14]
    beta_16 = output_me.x[15]
    beta_17 = output_me.x[16]
    beta_18 = output_me.x[17]





    betas_opt = np.array([beta_1, beta_2,
		beta_3,
		beta_4,beta_5,beta_6,beta_7,beta_8,
		beta_9,beta_10,beta_11,beta_12,
		beta_13,beta_14,beta_15,
		beta_16,beta_17,beta_18])

    #Updating parameters to compute ATT
    alphas = [[betas_opt[0], betas_opt[1],0,betas_opt[2],
          betas_opt[3], betas_opt[4]],
         [betas_opt[5], 0,betas_opt[6],betas_opt[7],
          betas_opt[8], betas_opt[9]]]
    
    betas = [betas_opt[10], betas_opt[11], betas_opt[12] ,betas_opt[13],betas_opt[14]]
    
    gammas = [betas_opt[15],betas_opt[16],betas_opt[17]]

    param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)




    # SIMULACIÓN SIMDATA
    
    simce_sims = np.zeros((N,n_sim))

    data_python = data_python[data_python['d_trat']==1]

    simce = []

    for x in range(0,2):

        treatment = np.ones(N)*x

        model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality)


        for j in range(n_sim):
    	        modelSD = sd.SimData(N,model)
    	        opt = modelSD.choice()
    	        simce_sims[:,j] = opt['Opt Simce']
	    
        simce.append(np.mean(simce_sims,axis=1))

    att_sim = simce[1] - simce[0]


	#aca voy: tomar betas y estimar ATT. Comparar con ATT data.


    return [att_data - np.mean(att_sim)]



#---Boostrap to get SE---#
boot_n = 400
diff_boot = np.zeros(boot_n)


start_time = time.time()


if __name__ == '__main__':
    with Pool(processes=8) as pool:
        mp.set_start_method('spawn', force = True)
        dics = pool.map(data_model,range(boot_n))
        pool.join()
        

time_opt = time.time() - start_time
print ('Bootstrap done in')
print("--- %s seconds ---" % (time_opt))


for j in range(boot_n):

    diff_boot[j] = dics[j][0]

#this is the s.e.
se_diff_boot = np.std(diff_boot)








#------Point estimate of att data - att model--------#

data_1 = pd.read_stata('/home/jrodriguezo/teachers/data/data_pythonpast.dta')

data = data_1[data_1['d_trat']==1]

N = np.array(data['experience']).shape[0]

for x in range(0,2):
    

    # TREATMENT #
    treatment = np.ones(N)*x
    # EXPERIENCE #
    years = np.array(data['experience'])
    # SCORE PORTFOLIO #
    p1_0 = np.array(data['score_port_past'])
    p1 = np.array(data['score_port'])
    # SCORE TEST #
    p2_0 = np.array(data['score_port_past'])
    p2 = np.array(data['score_test'])
    # CATEGORY PORTFOLIO #
    catPort = np.array(data['cat_port'])
    # CATEGORY TEST #
    catPrueba = np.array(data['cat_test'])
    # TRAME #
    #Recover initial placement from data (2016)
    TrameI = np.array(data['trame'])
    # TYPE SCHOOL #
    typeSchool = np.array(data['typeschool'])
 
    # Priority #
    priotity = np.array(data['por_priority'])
    
    rural_rbd = np.array(data['rural_rbd'])
    
    locality = np.array(data['AsignacionZona'])
    

    #### PARAMETERS MODEL ####
    N = np.size(p1_0)
    HOURS = np.array([44]*N)
    
    alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
          betas_nelder[3], betas_nelder[4]],
         [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
          betas_nelder[8], betas_nelder[9]]]
        
    betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12] ,betas_nelder[13],betas_nelder[14]]

    gammas = [betas_nelder[15],betas_nelder[16],betas_nelder[17]]
    
    dolar= 600
    value = [14403, 15155]
    hw = [value[0]/dolar,value[1]/dolar]
    porc = [0.0338, 0.0333]
    
    #inflation adjustemtn: 2012Jan-2020Jan: 1.111
    Asig = [150000*1.111,100000*1.111,50000*1.111]
    AEP = [Asig[0]/dolar,Asig[1]/dolar,Asig[2]/dolar] 
    
    # *** This is withouth teaching career ***
    # * value professional qualification (pesos)= 72100 *
    # * value professional mention (pesos)= 24034 *
    # *** This is with teaching career ***
    # * value professional qualification (pesos)= 253076 *
    # * value professional mention (pesos)= 84360 *
    
    #inflation adjustment: 2012Jan-2019Dec: 1.266
    qualiPesos = [72100*1.266, 24034*1.266, 253076, 84360] 
    qualiPesos = [72100, 24034, 253076, 84360]
    pro = [qualiPesos[0]/dolar, qualiPesos[1]/dolar, qualiPesos[2]/dolar, qualiPesos[3]/dolar]
    
    #* Progression component by tranche *
    #* value progression initial (pesos)= 14515
    #* value progression early (pesos)= 47831
    #* value progression advanced (pesos)= 96266
    #* value progression advanced (pesos)= 99914 Fixed component
    #* value progression expert 1 (pesos)= 360892
    #* value progression expert 1 (pesos)= 138769 Fixed component
    #* value progression expert 2 (pesos)= 776654
    #* value progression expert 2 (pesos)= 210929 Fixed component
    
    progress = [14515, 47831, 96266, 99914, 360892, 138769, 776654, 210929]
    
    pol = [progress[0]/dolar, progress[1]/dolar, progress[2]/dolar, progress[3]/dolar,  
       progress[4]/dolar, progress[5]/dolar, progress[6]/dolar, progress[7]/dolar]
    
    pri = [47872,113561]
    priori = [pri[0]/dolar, pri[1]/dolar]
    
    param0 = parameters.Parameters(alphas,betas,gammas,hw,porc,pro,pol,AEP,priori)
    
    model = util.Utility(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality)
    
    # SIMULACIÓN SIMDATA
    
    simce_sims = np.zeros((N,n_sim))
    income_sims = np.zeros((N,n_sim))
    baseline_sims = np.zeros((N,n_sim,2))
    
    for j in range(n_sim):
        modelSD = sd.SimData(N,model)
        opt = modelSD.choice()
        simce_sims[:,j] = opt['Opt Simce']
        income_sims[:,j] = opt['Opt Income'][1-x]
        baseline_sims[:,j,0] = opt['Potential scores'][0]
        baseline_sims[:,j,1] = opt['Potential scores'][1]
    
    simce.append(np.mean(simce_sims,axis=1))
    income.append(np.mean(income_sims,axis=1))
    baseline_p.append(np.mean(baseline_sims,axis=1))
    



print ('')
print ('ATT equals ', np.mean(simce[1] - simce[0]))
print ('')


#For validation purposes
att_sim = simce[1] - simce[0]
att_cost = income[1] - income[0]
att_mean_sim = np.mean(att_sim)


#Data complete
data_reg = pd.read_stata('/home/jrodriguezo/teachers/data/data_main_regmain.dta')



#REG Stata
#reg stdsimce_m d_trat d_year inter if (eval_year == 1 | eval_year == 2018 | eval_year == 0), vce(cluster drun)
#data_reg = data_reg[ (data_reg['eval_year'] == 1) | (data_reg['eval_year'] == 2018) | (data_reg['eval_year'] == 0) ]

y = np.array(data_reg['stdsimce'])
x_1 = np.array(data_reg['d_trat'])
x_2 = np.array(data_reg['d_year'])
x_3 = np.array(data_reg['inter'])
x = np.transpose(np.array([x_1, x_2, x_3]))
x = sm.add_constant(x)
cov_drun = np.array(data_reg['drun'])

model_reg = sm.OLS(exog=x, endog=y)
results = model_reg.fit(cov_type='cluster', cov_kwds={'groups': cov_drun}, use_t=True)

#reg_data = sm.OLS("stdsimce_m ~ d_trat + d_year + inter", data_reg).fit(cov_type='cluster', cov_kwds={'groups': data_reg['drun']})

print(results.summary())

#Recover the data

inter_data = results.params[3].round(8)
error_data = results.bse[3].round(8)
number_obs = results.nobs
inter_posit = inter_data + 1.96 * np.sqrt(results.normalized_cov_params[3,3])
inter_negat = inter_data - 1.96 * np.sqrt(results.normalized_cov_params[3,3])



#-----------Getting pvals and figure---------------#

att_diff = inter_data - att_mean_sim
t_stat = att_diff/se_diff_boot
pval_ht = 2*(1-norm.cdf(abs(t_stat)))


#Data Graph

#dataf_graph = {'ATTsim': att_sim, 'ATTdata': att_sim}
dataf_graph = {'ATTsim': att_sim}
dataf_graph = pd.DataFrame(dataf_graph, columns=['ATTsim'])

# Graphs Density: quitar data-based SEs y agregar p-val de test de igualdad

dataf_graph.plot.kde(linewidth=3, legend=False,alpha = 0.6, color = 'blue');
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.axvline(x=att_mean_sim, ymin=0, ymax=0.95, color='sandybrown', linestyle='-', linewidth=2,alpha = 0.8)
plt.axvline(x=inter_data, ymin=0, ymax=0.95, color='black', linestyle='--', linewidth=2,alpha = 0.6)
plt.annotate("Simulated ATT" "\n" + "("   +'{:04.2f}'.format(att_mean_sim) + r"$\sigma$s)", xy=(0.08, 1),
            xytext=(0.32, 1), arrowprops=dict(arrowstyle="->"))
plt.annotate("Data ATT" "\n" + "(" +'{:04.2f}'.format(inter_data) + r"$\sigma$s)", xy=(0.018, 1),
            xytext=(-0.4, 1), arrowprops=dict(arrowstyle="->"))
plt.annotate("Treatment effects distribution", xy=(0.2, 1.7),
            xytext=(0.2, 2), arrowprops=dict(arrowstyle="->"))
#plt.annotate(r"$\}$",fontsize=24, xy=(0.27, 0.77))
plt.text(0.055, 2.8, r'$p$-val $=$'    +'{:04.2f}'.format(pval_ht), ha="center", va="center")
plt.text(0.055, 2.7, r'$\}$', ha="center", va="center", rotation= 90, fontsize = 18)
plt.xlim(-0.6,0.6)
plt.savefig('/home/jrodriguezo/teachers/results/att_diffs.pdf', format='pdf')








