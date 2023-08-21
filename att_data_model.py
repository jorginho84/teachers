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
import data_models
from utility_counterfactual_att import Count_att

#ver https://pythonspeed.com/articles/python-multiprocessing/
import multiprocessing as mp
from multiprocessing import Pool



np.random.seed(123)

betas_nelder  = np.load("/home/jrodriguezo/teachers/codes/betasopt_model_v29.npy")
df = pd.read_stata('/home/jrodriguezo/teachers/data/data_pythonpast_v2023.dta')
moments_vector = np.load("/home/jrodriguezo/teachers/codes/moments_v2023.npy")
ses_opt = np.load("/home/jrodriguezo/teachers/codes/ses_model_v2023.npy")
data_reg = pd.read_stata('/home/jrodriguezo/teachers/data/FINALdata.dta')


df.to_pickle("df.pkl")
data_reg.to_pickle("data_reg.pkl")


n_sim = 100

simce = []
baseline_p = []
income = [] 


#---Boostrap to get SE---#
boot_n = 400
diff_boot = np.zeros(boot_n)


start_time = time.time()


if __name__ == '__main__':
    with Pool(processes=5) as pool:
        mp.set_start_method('spawn', force = True)
        dics = pool.map(data_models.data_model,range(boot_n))
        pool.join()
        

time_opt = time.time() - start_time
print ('Bootstrap done in')
print("--- %s seconds ---" % (time_opt))


for j in range(boot_n):

    diff_boot[j] = dics[j][0]

#this is the s.e.
se_diff_boot = np.std(diff_boot)





#------Point estimate of att data - att model--------#
df = pd.read_pickle("df.pkl")

df = data_1[data_1['d_trat']==1]

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

    AEP_priority = np.array(data['priority_aep'])
    
    rural_rbd = np.array(data['rural_rbd'])
    
    locality = np.array(data['AsignacionZona'])
    

    #### PARAMETERS MODEL ####
    N = np.size(p1_0)
    HOURS = np.array([44]*N)
    
    alphas = [[betas_nelder[0], betas_nelder[1],0,betas_nelder[2],
      betas_nelder[3], betas_nelder[4]],
     [betas_nelder[5], 0,betas_nelder[6],betas_nelder[7],
      betas_nelder[8], betas_nelder[9]]]

    betas = [betas_nelder[10], betas_nelder[11], betas_nelder[12] ,betas_nelder[13],betas_nelder[14],betas_nelder[15]]
    gammas = [betas_nelder[16],betas_nelder[17],betas_nelder[18]]
    
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
    
    model = Count_att(param0,N,p1_0,p2_0,years,treatment,typeSchool,HOURS,p1,p2,catPort,catPrueba,TrameI,
                         priotity,rural_rbd,locality,AEP_priority)
    
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
data_reg = pd.read_pickle("data_reg.pkl")

# first drop Stata 1083190 rows 
data_reg = data_reg[(data_reg["stdsimce_m"].notna()) & (data_reg["stdsimce_l"].notna())]

#destring
data_reg["drun_l"] = pd.to_numeric(data_reg["drun_l"], errors='coerce')
data_reg["drun_m"] = pd.to_numeric(data_reg["drun_m"], errors='coerce')


##### generates variables #####
#eval_year
data_reg.loc[data_reg["eval_year_m"]==data_reg["eval_year_l"],'eval_year'] = data_reg["eval_year_m"]
data_reg.loc[(data_reg["eval_year_m"].notna()) & (data_reg["eval_year_l"].isna()),'eval_year'] = data_reg["eval_year_m"]
data_reg.loc[(data_reg["eval_year_m"].isna()) & (data_reg["eval_year_l"].notna()),'eval_year'] = data_reg["eval_year_l"]

#drun
data_reg.loc[data_reg["drun_m"]==data_reg["drun_l"],'drun'] = data_reg["drun_m"]
data_reg.loc[(data_reg["drun_m"].notna()) & (data_reg["drun_l"].isna()),'drun'] = data_reg["drun_m"]
data_reg.loc[(data_reg["drun_m"].isna()) & (data_reg["drun_l"].notna()),'drun'] = data_reg["drun_l"]

#experience
data_reg.loc[data_reg["experience_m"]==data_reg["experience_l"],'experience'] = data_reg["experience_m"]
data_reg.loc[(data_reg["experience_m"].notna()) & (data_reg["experience_l"].isna()),'experience'] = data_reg["experience_m"]
data_reg.loc[(data_reg["experience_m"].isna()) & (data_reg["experience_l"].notna()),'experience'] = data_reg["experience_l"]

#d_trat
data_reg.loc[data_reg["d_trat_m"]==data_reg["d_trat_l"],'d_trat'] = data_reg["d_trat_m"]
data_reg.loc[(data_reg["d_trat_m"].notna()) & (data_reg["d_trat_l"].isna()),'d_trat'] = data_reg["d_trat_m"]
data_reg.loc[(data_reg["d_trat_m"].isna()) & (data_reg["d_trat_l"].notna()),'d_trat'] = data_reg["d_trat_l"]

#inter
data_reg.loc[data_reg["inter_m"]==data_reg["inter_l"],'inter'] = data_reg["inter_m"]
data_reg.loc[(data_reg["inter_m"].notna()) & (data_reg["inter_l"].isna()),'inter'] = data_reg["inter_m"]
data_reg.loc[(data_reg["inter_m"].isna()) & (data_reg["inter_l"].notna()),'inter'] = data_reg["inter_l"]

#d_year
data_reg.loc[data_reg["d_year_m"]==data_reg["d_year_l"],'d_year'] = data_reg["d_year_m"]
data_reg.loc[(data_reg["d_year_m"].notna()) & (data_reg["d_year_l"].isna()),'d_year'] = data_reg["d_year_m"]
data_reg.loc[(data_reg["d_year_m"].isna()) & (data_reg["d_year_l"].notna()),'d_year'] = data_reg["d_year_l"]
        
##### drop nan #####
data_reg = data_reg[(data_reg["edp"].notna())]
data_reg = data_reg[(data_reg["edm"].notna())]
data_reg = data_reg[(data_reg["ingreso"].notna())]
data_reg = data_reg[(data_reg["experience"].notna())]
data_reg = data_reg[(data_reg["drun"].notna())]
data_reg = data_reg[(data_reg["d_trat"].notna())]
data_reg = data_reg[(data_reg["d_year"].notna())]
data_reg = data_reg[(data_reg["inter"].notna())]
                                                                   
# keep if eval_year == 1 | eval_year == 2018 | eval_year == 0
data_reg = data_reg[(data_reg["eval_year"] == 1) | (data_reg["eval_year"] == 2018) | (data_reg["eval_year"] == 0)]

# mean simce
data_reg['stdsimce'] = data_reg[['stdsimce_m', 'stdsimce_l']].mean(axis=1)


data_reg['constant'] = np.ones(np.size(data_reg['stdsimce']))
data_reg.set_index(['rbd','agno'],inplace = True)

model_reg = PanelOLS(dependent = data_reg['stdsimce'], exog = data_reg[['constant', 'd_trat', 'd_year', 'inter', 'edp', 'edm', 'ingreso', 'experience']], entity_effects = True)
results = model_reg.fit(cov_type='clustered', clusters=data_reg['drun'])

#reg_data = sm.OLS("stdsimce_m ~ d_trat + d_year + inter", data_reg).fit(cov_type='cluster', cov_kwds={'groups': data_reg['drun']})

print(results.summary())

#Recover the data

inter_data = results.params.inter.round(8)
error_data = results.std_errors.inter.round(8)
number_obs = results.nobs
inter_posit = inter_data + 1.96 * np.sqrt(results.cov.inter.inter)
inter_negat = inter_data - 1.96 * np.sqrt(results.cov.inter.inter)



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








