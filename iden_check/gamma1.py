#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:39:57 2021

@author: jorge-home
"""

lenght = 0.05
size_grid = 5
max_p = -0.2
min_p = 0.2
p_list = np.linspace(min_p,max_p,size_grid)
obs_moment = moments_vector[14].copy()

target_moment = np.zeros((size_grid,))

for i in range(size_grid):
    param0.gammas[1] = p_list[i]
    corr_data = output_ins.simulation(50,modelSD)
    target_moment[i] = corr_data['perc adv/exp control']

#Back to original
exec(open('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/iden_check/load_param.py').read())

#the graph
fig, ax=plt.subplots()
plot1=ax.plot(p_list,target_moment,'b-o',label='Simulated',alpha=0.9)
plot2=ax.plot(p_list,np.full((size_grid,),obs_moment),'b-.',label='Observed',alpha=0.9)
plt.setp(plot1,linewidth=3)
plt.setp(plot2,linewidth=3)
ax.legend()
ax.set_ylabel(r'% of "advanced" teachers',fontsize=font_size)
ax.set_xlabel(r'Utility loss from effort type 2',fontsize=font_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.legend(loc=0)
plt.show()
fig.savefig('/Users/jorge-home/Dropbox/Research/teachers-reform/codes/teachers/iden_check/gamma1.pdf', format='pdf')
plt.close()


