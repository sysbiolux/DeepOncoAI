# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:20:38 2023

@author: apurva.badkas
"""

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import textwrap


datafile = pd.read_csv('Partial_data_MeasuresRNARPPADISC.csv')
trans = datafile.T
trans.columns = trans.iloc[0,:]
trans = trans.drop(['Drug'])

# fig = sns.violinplot(datafile, inner="points")
# plt.xticks(rotation=90)
# fig.set_ylabel("AUC")
# for violin, alpha in zip(fig.collections[::2], [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]):
#     violin.set_alpha(alpha)

# fig2 = sns.boxplot(datafile)
# plt.xticks(rotation=90)
# #fig2 = sns.violinplot(trans)

# fig3 = sns.violinplot(trans,inner="points")
# plt.xticks(rotation=90)
# fig3.set_ylabel("AUC")
# for violin, alpha in zip(fig3.collections[::2], [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]):
#     violin.set_alpha(alpha)

df_heat = datafile.set_index('Drug')
    
fig, ax = plt.subplots(figsize=(26, 15))
cmap = sns.diverging_palette(220,15, n = 10)
sns.set(font_scale=1.4)
fig = sns.heatmap(df_heat, cmap=cmap, vmin=0,vmax=1, annot=True, fmt='.2f', linewidths=.1, ax=ax, annot_kws={
                'fontsize': 16,
                'fontweight': 'bold',
                'fontfamily': 'serif'
            })
#plt.set_xticklabels(fontsize = 16)
fig.set_xticklabels(fig.get_xmajorticklabels(), fontsize = 18, fontweight = 'bold', rotation=90)
fig.set_yticklabels(fig.get_ymajorticklabels(), fontsize = 18, fontweight = 'bold')
ax.set_xticklabels([textwrap.fill(e, 11) for e in df_heat.columns])
#plt.xticks(rotation=90)
#plt.xlabel('')
plt.savefig('Partial_data_MeasuresRNARPPADISC_greyred.svg', dpi=300)
plt.close()
    