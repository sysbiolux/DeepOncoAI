# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:01:56 2023

@author: apurva.badkas
"""

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

stats_file = pd.read_csv('Eigen_Concated_stats.csv', index_col = 0)
stats_trans = stats_file.T
stat_df = stats_trans.loc[:,'Cell_line'].str.split('_', expand=True)
stat_df = stat_df.drop(columns = [0, 2,3])
merged = stats_trans.join(stat_df)
merged.columns = ['Run_time', 'Nodes', 'Edges', 'Cell_line', 'Can_type']

melted_test = merged.drop(columns = ['Run_time','Cell_line'])
melted = pd.melt(melted_test, id_vars = ['Can_type'], value_vars =['Nodes','Edges'])
melted['value'] = pd.to_numeric(melted['value'])

fig1 = sns.violinplot(data=merged, x = 'Can_type', y = pd.to_numeric(merged['Nodes']), inner='points')
plt.xticks(rotation=90)
plt.xlabel('')
plt.savefig('Cancer_types_nodes.svg')
plt.close()
    
fig2 = sns.scatterplot(merged, x=pd.to_numeric(merged['Nodes']), y=pd.to_numeric(merged['Edges']), hue='Can_type')


fig3 = sns.boxplot(data = melted, x = 'variable', y = 'value', hue='Can_type')



sns.set(font="serif")
fig4 = sns.catplot(x = 'Can_type', y='value', data=melted, row='variable', kind="swarm", sharey=False,palette=sns.diverging_palette(220,15, n = 5))
plt.savefig('Nodes_and_edges_T_5_swarm.svg')
plt.close()

#ax0, ax1 = grid.axes[0] 


g = sns.FacetGrid(melted, col="variable")
g.map_dataframe(sns.histplot, x="total_bill")