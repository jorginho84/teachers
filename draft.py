"""
This code computes effects on simce of different PFP policies

exec(open("/home/jrodriguezo/teachers/codes/draft.py").read())
"""


for x in range(blen):

    #wtps
    wtp_list_b.append(income_b[:,x] - np.exp(utils_list[0] + gammas[0]*effort_p_b[:,x] + gammas[1]*effort_t_b[:,x] - gammas[2]*simce_b[:,x] ))
    
    #Changes in income (to compute added revenues and provision cost)
    delta_income_b.append(income_b[:,x] - income[0])

    #ATTs on simce
    delta_simce_b.append(simce_b[:,x] - simce[0])

wtp_student_b = np.zeros(blen)
wtp_teachers_b = np.zeros(blen)
wtp_overall_b = np.zeros(blen)
provision_b = np.zeros(blen)
revenue_b = np.zeros(blen)
net_cost_b = np.zeros(blen)
mvpf_b = np.zeros(blen)

for x in range(blen):
    wtp_student_b[x] = np.mean(delta_simce_b[x])*rho*lifetime_earnings
    wtp_teachers_b[x] = np.mean(wtp_list_b[x])
    wtp_overall_b[x] = wtp_student_b[x] + wtp_teachers_b[x]
    provision_b[x] = np.mean(delta_income_b[x])*12
    revenue_b[x] = np.mean(delta_simce_b[x])*rho*lifetime_earnings*tax + provision_b[x]*tax
    net_cost_b[x] = provision_b[x] - revenue_b[x]
    mvpf_b[x] = wtp_overall_b[x] / net_cost_b[x]


b_points = np.arange(0,3000,100)
#WTP Effects across b   
fig, ax=plt.subplots()
plot1 = ax.scatter(b_points,wtp_student_b,color='b' ,marker = 'o',alpha=.8, label='WTP students',s=70)
plot2 = ax.scatter(b_points,wtp_teachers_b,color='r' ,marker = '^', alpha=.8, label='WTP teachers',s=70)
ax.set_ylabel(r'WTP', fontsize=15)
ax.set_xlabel(r'Slope in wage schedule', fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
#ax.set_ylim(-0.1,1.05)
#plt.xticks(x, ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-'],fontsize=12)
ax.legend(loc = 'upper left',fontsize = 15)
plt.tight_layout()
plt.show()
fig.savefig('/home/jrodriguezo/teachers/results/pfp_slopes_wtp.pdf', format='pdf')
plt.close()


