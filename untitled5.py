# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:27:37 2025

@author: gangu
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file_name = ["VF_CF_values_Sampled_RSLA_new_no_lambdaRS_A2C","VF_CF_values_Fast_RSLA_new_RS_A2C"]
env_nm = "River_swim_"
lab = ["Sampled_Fast_RSLA","Fast_RSLA"]

# Create figures explicitly
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for i,file in enumerate(file_name):
    data = pd.read_excel(file+".xlsx")
    vf, cf  = data['vf'], data['cf']
    ax1.plot(vf, label=lab[i])
    ax2.plot(cf, label=lab[i])

# Customize figure 1
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Value function')
ax1.set_title(env_nm+"sampled_comparison")
ax1.legend()
fig1.savefig('Value_fn_comp_'+env_nm+'sampled_comparison.pdf')

# Customize figure 2
ax2.plot(np.ones(len(cf))*7,label='baseline')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost function')
ax2.set_title(env_nm+"sampled_comparison")
ax2.legend()
fig2.savefig('Cost_fn_comp_'+env_nm+'sampled_comparison.pdf')

plt.show()