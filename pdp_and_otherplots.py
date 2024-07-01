import pandas as pd
import numpy as np
import scipy as sp

from config import Config
from sklearn.metrics import roc_curve, auc
from DBM_toolbox.data_manipulation.data_utils import pickle_objects, unpickle_objects
from DBM_toolbox.data_manipulation import data_utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

config = Config("testall/config_paper.yaml")
algos = config.raw_dict["modeling"]["general"]["algorithms"]
metric = config.raw_dict["modeling"]["general"]["metric"]
rng = np.random.default_rng(42)

data, ActAreas, IC50s, dose_responses = config.read_data()
datat = unpickle_objects('FINAL_preprocessed_data_2023-02-16-10-30-39-935233.pkl')
ddf = data.dataframe
t = datat.to_pandas(omic='TYPE')
df = ddf.join(t)

def plot_swarm(genelist, druglist, title, labels):
    df_x = df.loc[:, genelist + druglist].dropna()
    df_x[title] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
    plt.figure(figsize=(8,7))
    ax = sns.swarmplot(data=df_x, x=title, y=druglist[0], linewidth=1)
    sns.boxplot(data=df_x, x=title, y=druglist[0], ax=ax)
    ax.set(xlabel=title)
    ax.set_xticklabels(labels)
    x1 = [df_x.loc[x, druglist[0]] for x in df_x.index if df_x.loc[x, title]]
    x2 = [df_x.loc[x, druglist[0]] for x in df_x.index if not df_x.loc[x, title]]
    t, p = sp.stats.ttest_ind(x1, x2)
    ax.set(ylim=(-0.2, 8.8))
    ax.text(0.35, 7.5, 't-test: p={:.2g}'.format(p))

def plot_linreg(genelist, druglist, title):
    df_x = df.loc[:, genelist + druglist].dropna()
    plt.figure(figsize=(7,6))
    ax = sns.regplot(data=df_x, x=genelist[0], y=druglist[0])
    r, p = sp.stats.pearsonr(df_x[genelist[0]], df_x[druglist[0]])
    ax.set(ylim=(-0.2, 8.8))
    ax.text(0.35, 7.5, 'r={:.2f}, p={:.2g}'.format(r, p))
    ax.set(xlabel=title)



############################### AZD6244 ##################

genelist = ['BRAF_MUT', 'BRAF.V600E_MUT', 'BRAF.MC_MUT', 'KRAS.G12_13_MUT', 'KRAS_MUT', 'NRAS_MUT']
druglist = ['AZD6244_ActArea']
title = 'Ras-Raf'
labels = ['WT', 'Mutated']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist = ['ETV4_ENSG00000175832.8']
druglist = ['AZD6244_ActArea']
title = 'ETV4'
labels = ['Res', 'Sens']
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['GSTM1_DEL']
title = 'GSTM1'
labels = ['normal', 'deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['ERBB4_DEL']
title = 'ERBB4'
labels = ['normal', 'deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['ETV4_ENSG00000175832.8']
title = 'ETV4'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['CMTM7_ENSG00000153551.9']
title = 'CMTM7'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['S100A4_ENSG00000196154.7']
title = 'S100A4'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['Akt_pS473']
title = 'phospho-Akt'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['c-Jun_pS73']
title = 'phospho-c-Jun'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['MEK1_pS217_S221']
title = 'phospho-MEK1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-130a_nmiR00132.1']
title = 'miR-130a'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-1257_nmiR00070.1']
title = 'miR-1257'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-1302_nmiR00125.1']
title = 'miR-1302'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

for genelist in [[' IL-1'], [' VEGF'], [' TGFB'], [' MAPK_only']]:
    title = genelist[0]
    plot_linreg(genelist, druglist, title)
    plt.savefig(f'Result_{druglist[0]}_{title}')


################################ PD-0324901 #####################

genelist = ['BRAF_MUT', 'BRAF.V600E_MUT', 'BRAF.MC_MUT', 'KRAS.G12_13_MUT', 'KRAS_MUT', 'NRAS_MUT']
druglist = ['PD-0325901_ActArea']
title = 'Ras-Raf'
labels = ['WT', 'Mutated']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')



genelist = ['OR2T11_AMP', 'OR2T10_AMP', 'OR4N4_DEL', 'OR4M2_DEL', 'OR51A4_DEL']
title = 'OR_various'
labels = ['No CNV', 'Amp or Del']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['GSTM1_DEL']
title = 'GSTM1'
labels = ['Normal', 'Deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist  = ['KANK1_AMP']
title = 'KANK1'
labels = ['Normal', 'Amplified']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['ETV4_ENSG00000175832.8']
title = 'ETV4'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['SPRY2_ENSG00000136158.6']
title = 'SPRY2'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['TOR4A_ENSG00000198113.2']
title = 'TOR4A'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['Akt_pS473']
title = 'phospho-Akt'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['Chk2_pT68_Caution']
title = 'phospho-Chk2'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['GABA']
title = 'GABA'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-34a_nmiR00324.1']
title = 'miR-34a'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-130a_nmiR00132.1']
title = 'miR-130a'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

############################ Lapatinib ###########################

genelist  = ['OR51A2_DEL', 'OR51A4_DEL', 'OR2T35_AMP']
druglist = ['Lapatinib_ActArea']
title = 'OR various'
labels = ['No CNV', 'Amp or Del']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist  = ['ZFP14_DEL']
title = 'ZFP14'
labels = ['Normal', 'Deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist  = ['TNK2_MUT']
title = 'TNK2'
labels = ['WT', 'Mutated']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['GPX3_ENSG00000211445.7']
title = 'GPX3'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['SYTL1_ENSG00000142765.13']
title = 'STYL1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['HER2_pY1248_Caution']
title = 'HER2'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['creatine']
title = 'creatine'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['EGFR']
title = 'EGFR'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['PAI-1']
title = 'PAI-1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-558_nmiR00511.1']
title = 'miR-558'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['2-deoxycytidine']
title = '2-deoxycytidine'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-141_nmiR00151.1']
title = 'miR-141'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')



############################ Erlotinib ###############################

druglist = ['Erlotinib_ActArea']
genelist  = ['OR51A2_DEL', 'OR51A2_DEL', 'OR2T35_AMP']
title = 'OR various'
labels = ['No CNV', 'Amp or Del']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist  = ['NAALADL2_DEL']
title = 'NAALADL2'
labels = ['Normal', 'Deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist  = ['GALC_DEL']
title = 'GALC'
labels = ['Normal', 'Deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist  = ['FLNA_MUT']
title = 'FLNA'
labels = ['WT', 'Mutated']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist = ['CORO2A_ENSG00000106789.8']
title = 'CORO2A'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['TSTD1_ENSG00000215845.6']
title = 'TSTD1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['AKR1E2_ENSG00000165568.13']
title = 'AKR1E2'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['SYTL1_ENSG00000142765.13']
title = 'SYTL1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['P-Cadherin_Caution']
title = 'P-Cadherin'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['C-Raf_pS338']
title = 'phospho-C_Raf'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['guanosine']
title = 'guanosine'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-429_nmiR00367.2']
title = 'miR-429'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-548a-3p_nmiR00480.1']
title = 'miR-548a-3p'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-216a_nmiR00244.1']
title = 'miR-216a'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

############################# Irinotecan ####################################

genelist = ['HLA-DRB6_AMP', 'HLA-DRB1_AMP', 'HLA-DRB5_AMP']
druglist = ['Irinotecan_ActArea']
title = 'HLA-DRB1/5/6'
labels = ['No CNV', 'Amplified']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['KRAS.G12_13_MUT', 'KRAS_MUT']
title = 'KRAS'
labels = ['WT', 'Mutated']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['FBXL7_DEL']
title = 'FBXL7'
labels = ['Normal', 'Deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist = ['SLFN11_ENSG00000172716.12']
title = 'SLFN11'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['XIST_ENSG00000229807.5']
title = 'XIST'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['HNRNPA1_ENSG00000135486.13']
title = 'HNRNPA1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['Chk1_Caution']
title = 'Chk1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['alpha-hydroxybutyrate']
title = 'alpha-hydroxybutyrate'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = [' JAK-STAT']
title = 'JAK-STAT_pathway'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-574-5p_nmiR00526.1']
title = 'miR-574-5p'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-608_nmiR00562.1']
title = 'miR-608'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-22_nmiR00251.1']
title = 'miR-22'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')


############################### Paclitaxel ####################################

genelist = ['EP300_MUT']
druglist = ['Paclitaxel_ActArea']
title = 'EP300'
labels = ['WT', 'Mutated']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['ADAM5_DEL', 'ADAM3A_DEL']
title = 'ADAM5/ADAM3A'
labels = ['Normal', 'Deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['ABCB1_ENSG00000085563.10']
title = 'ABCB1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['LEPREL2_ENSG00000110811.15']
title = 'LEPREL2'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['XBP1_Caution']
title = 'XBP1'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['Bcl-xL']
title = 'Bcl-xL'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['AMPK_alpha_Caution']
title = 'AMPK_alpha'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist = ['hsa-miR-30a_nmiR00287.1']
title = 'miR-30a'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist = ['hsa-miR-607_nmiR00561.1']
title = 'miR-607'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist = ['hsa-miR-22_nmiR00251.1']
title = 'miR-22'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')


genelist = ['hsa-let-7e_nmiR00005.1']
title = 'let-7e'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')


############################# Panobinostat ######################################

genelist = ['LINC00226_AMP']
druglist = ['Panobinostat_ActArea']
title = 'LINC00226'
labels = ['Normal', 'Amplified']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['PRODH_DEL']
title = 'PRODH'
labels = ['Normal', 'Deleted']
plot_swarm(genelist, druglist, title, labels)
plt.savefig(f'Result_{druglist[0]}_{title}')



genelist = ['beta-Catenin_pT41_S45']
title = 'phospho-beta-Catenin'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['Akt_pS473']
title = 'phospho-Akt'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['1-methylnicotinamide']
title = '1-methylnicotinamide'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['alpha-glycerophosphate']
title = 'alpha-glycerophosphate'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['pantothenate']
title = 'pantothenate'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-24_nmiR00261.1']
title = 'miR-24'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-let-7i_nmiR00008.1']
title = 'let-7i'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-320d_nmiR00297.2']
title = 'miR-320d'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-miR-22_nmiR00251.1']
title = 'miR-22'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')

genelist = ['hsa-let-7e_nmiR00005.1']
title = 'let-7e'
plot_linreg(genelist, druglist, title)
plt.savefig(f'Result_{druglist[0]}_{title}')
