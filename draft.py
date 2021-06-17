"""
exec(open("/home/jrodriguez/teachers/codes/draft.py").read())

"""

#saving results

for j in range(boot_n):

    alpha_00[j] = dics[j][0]
    alpha_01[j] = dics[j][1]
    alpha_03[j] = dics[j][2]
    alpha_04[j] = dics[j][3]
    alpha_05[j] = dics[j][4]
    alpha_10[j] = dics[j][5]
    alpha_12[j] = dics[j][6]
    alpha_13[j] = dics[j][7]
    alpha_14[j] = dics[j][8]
    alpha_15[j] = dics[j][9]
    beta_0[j] = dics[j][10]
    beta_1[j] = dics[j][11]
    beta_2[j] = dics[j][12]
    beta_3[j] = dics[j][13]
    gamma_0[j] = dics[j][14]
    gamma_1[j] = dics[j][15]
    gamma_2[j] = dics[j][16]



est_alpha_00 = np.std(alpha_00)
est_alpha_01 = np.std(alpha_01)
est_alpha_03 = np.std(alpha_03)
est_alpha_04 = np.std(alpha_04)
est_alpha_05 = np.std(alpha_05)
est_alpha_10 = np.std(alpha_10)
est_alpha_12 = np.std(alpha_12)
est_alpha_13 = np.std(alpha_13)
est_alpha_14 = np.std(alpha_14)
est_alpha_15 = np.std(alpha_15)
est_beta_0 = np.std(beta_0)
est_beta_1 = np.std(beta_1)
est_beta_2 = np.std(beta_2)
est_beta_3 = np.std(beta_3)
est_gamma_0 = np.std(gamma_0)
est_gamma_1 = np.std(gamma_1)
est_gamma_2 = np.std(gamma_2)

dics_se = {'SE alpha_00': est_alpha_00,
                'SE alpha_01': est_alpha_01,
                'SE alpha_03': est_alpha_03,
                'SE alpha_04': est_alpha_04,
                'SE alpha_05': est_alpha_05,
                'SE alpha_10': est_alpha_10,
                'SE alpha_12': est_alpha_12,
                'SE alpha_13': est_alpha_13,
                'SE alpha_14': est_alpha_14,
                'SE alpha_15': est_alpha_15,
                'SE beta_0': est_beta_0,
                'SE beta_1': est_beta_1,
                'SE beta_2': est_beta_2,
                'SE beta_3': est_beta_3,
                'SE gamma_0': est_gamma_0,
                'SE gamma_1': est_gamma_1,
                'SE gamma_2': est_gamma_2}
    
betas_opt = np.array([dics_se['SE alpha_00'], dics_se['SE alpha_01'], 
                              dics_se['SE alpha_03'],dics_se['SE alpha_04'],
                              dics_se['SE alpha_05'],dics_se['SE alpha_12'], 
                                  dics_se['SE alpha_13'],dics_se['SE alpha_14'],dics_se['SE alpha_15'],
                                  dics_se['SE beta_0'],
                                  dics_se['SE beta_1'],dics_se['SE beta_2'], 
                                      dics_se['SE beta_3'],dics_se['SE gamma_0'],dics_se['SE gamma_1'], dics_se['SE gamma_2']])



np.save('/home/jrodriguez/teachers/results/se_model_v4.npy',betas_opt)
