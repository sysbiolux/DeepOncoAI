import pandas as pd
import numpy as np
import scipy as sp

from config import Config
from sklearn.metrics import roc_curve, auc
from functions import pickle_objects, unpickle_objects
from DBM_toolbox.data_manipulation import data_utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

config = Config("testall/config.yaml")
algos = config.raw_dict["modeling"]["general"]["algorithms"]
metric = config.raw_dict["modeling"]["general"]["metric"]
rng = np.random.default_rng(42)

data, ActAreas, IC50s, dose_responses = config.read_data()
datat = unpickle_objects('f_testall_01_data_2022-07-21-21-31-08-689557.pkl')
ddf = data.dataframe
t = datat.to_pandas(omic='TYPE')
df = ddf.join(t)



############################### AZD6244 ##################

df_x = df.loc[:, ['BRAF_MUT', 'BRAF.V600E_MUT', 'BRAF.MC_MUT', 'KRAS.G12_13_MUT', 'KRAS_MUT', 'NRAS_MUT',
                  'AZD6244_ActArea']].dropna()
df_x['Ras_RAF_MUT'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='Ras_RAF_MUT', y='AZD6244_ActArea')
ax = plt.gca()
x1 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if df_x.loc[x, 'Ras_RAF_MUT']]
x2 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if not df_x.loc[x, 'Ras_RAF_MUT']]
t, p = sp.stats.ttest_ind(x1, x2)
ax.text(0.35, 6, 't-test: p={:.2g}'.format(p))


df_x = df.loc[:, ['OR4M2_DEL', 'OR4N4_DEL', 'OR2A7_AMP', 'OR2T10_DEL', 'AZD6244_ActArea']].dropna()
df_x['OR_MUT'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='OR_MUT', y='AZD6244_ActArea')
ax = plt.gca()
x1 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if df_x.loc[x, 'OR_MUT']]
x2 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if not df_x.loc[x, 'OR_MUT']]
t, p = sp.stats.ttest_ind(x1, x2)
ax.text(0.35, 6, 't-test: p={:.2g}'.format(p))

df_x = df.loc[:, ['GSTM1_DEL', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='GSTM1_DEL', y='AZD6244_ActArea')
ax = plt.gca()
x1 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if df_x.loc[x, 'GSTM1_DEL']]
x2 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if not df_x.loc[x, 'GSTM1_DEL']]
t, p = sp.stats.ttest_ind(x1, x2)
ax.text(0.35, 6, 't-test: p={:.2g}'.format(p))


df_x = df.loc[:, ['ERBB4_DEL', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='ERBB4_DEL', y='AZD6244_ActArea')
ax = plt.gca()
x1 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if df_x.loc[x, 'ERBB4_DEL']]
x2 = [df_x.loc[x, 'AZD6244_ActArea'] for x in df_x.index if not df_x.loc[x, 'ERBB4_DEL']]
t, p = sp.stats.ttest_ind(x1, x2)
ax.text(0.35, 6, 't-test: p={:.2g}'.format(p))


df_x = df.loc[:, ['ETV4_ENSG00000175832.8', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='ETV4_ENSG00000175832.8', y='AZD6244_ActArea')
ax = plt.gca()
r, p = sp.stats.pearsonr(df_x['ETV4_ENSG00000175832.8'], df_x['AZD6244_ActArea'])
ax.text(0.35, 6.5, 'r={:.2f}, p={:.2g}'.format(r, p))


df_x = df.loc[:, ['CMTM7_ENSG00000153551.9', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='CMTM7_ENSG00000153551.9', y='AZD6244_ActArea')
ax = plt.gca()
r, p = sp.stats.pearsonr(df_x['CMTM7_ENSG00000153551.9'], df_x['AZD6244_ActArea'])
ax.text(0.35, 6.5, 'r={:.2f}, p={:.2g}'.format(r, p))

df_x = df.loc[:, ['S100A4_ENSG00000196154.7', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='S100A4_ENSG00000196154.7', y='AZD6244_ActArea')

df_x = df.loc[:, ['Akt_pS473', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='Akt_pS473', y='AZD6244_ActArea')
ax = plt.gca()
r, p = sp.stats.pearsonr(df_x['Akt_pS473'], df_x['AZD6244_ActArea'])
ax.text(0.35, 6.5, 'r={:.2f}, p={:.2g}'.format(r, p))


df_x = df.loc[:, ['c-Jun_pS73', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='c-Jun_pS73', y='AZD6244_ActArea')

df_x = df.loc[:, ['hsa-miR-130a_nmiR00132.1', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-130a_nmiR00132.1', y='AZD6244_ActArea')
ax = plt.gca()
r, p = sp.stats.pearsonr(df_x['hsa-miR-130a_nmiR00132.1'], df_x['AZD6244_ActArea'])
ax.text(0.35, 6.5, 'r={:.2f}, p={:.2g}'.format(r, p))


df_x = df.loc[:, ['hsa-miR-1302_nmiR00125.1', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-1302_nmiR00125.1', y='AZD6244_ActArea')
ax = plt.gca()
r, p = sp.stats.pearsonr(df_x['hsa-miR-1302_nmiR00125.1'], df_x['AZD6244_ActArea'])
ax.text(0.35, 6.5, 'r={:.2f}, p={:.2g}'.format(r, p))


df_x = df.loc[:, ['C34:4 PC', 'AZD6244_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='C34:4 PC', y='AZD6244_ActArea')

################################ PD-0324901 #####################

df_x = df.loc[:, ['BRAF_MUT', 'BRAF.V600E_MUT', 'BRAF.MC_MUT', 'KRAS.G12_13_MUT', 'KRAS_MUT', 'NRAS_MUT',
                  'PD-0325901_ActArea']].dropna()
df_x['Ras_RAF_MUT'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='Ras_RAF_MUT', y='PD-0325901_ActArea')

df_x = df.loc[:, ['OR2T11_AMP', 'OR2T10_AMP', 'OR4N4_DEL', 'OR4M2_DEL', 'OR51A4_DEL', 'PD-0325901_ActArea']].dropna()
df_x['OR_MUT'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='OR_MUT', y='PD-0325901_ActArea')

df_x = df.loc[:, ['GSTM1_DEL', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='GSTM1_DEL', y='PD-0325901_ActArea')

df_x = df.loc[:, ['KANK1_AMP', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='KANK1_AMP', y='PD-0325901_ActArea')


df_x = df.loc[:, ['ETV4_ENSG00000175832.8', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='ETV4_ENSG00000175832.8', y='PD-0325901_ActArea')

df_x = df.loc[:, ['SPRY2_ENSG00000136158.6', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='SPRY2_ENSG00000136158.6', y='PD-0325901_ActArea')

df_x = df.loc[:, ['TOR4A_ENSG00000198113.2', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='TOR4A_ENSG00000198113.2', y='PD-0325901_ActArea')

df_x = df.loc[:, ['Akt_pS473', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='Akt_pS473', y='PD-0325901_ActArea')

df_x = df.loc[:, ['Chk2_pT68_Caution', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='Chk2_pT68_Caution', y='PD-0325901_ActArea')

df_x = df.loc[:, ['GABA', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='GABA', y='PD-0325901_ActArea')

df_x = df.loc[:, ['hsa-miR-34a_nmiR00324.1', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-34a_nmiR00324.1', y='PD-0325901_ActArea')

df_x = df.loc[:, ['hsa-miR-130a_nmiR00132.1', 'PD-0325901_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-130a_nmiR00132.1', y='PD-0325901_ActArea')


############################ Lapatinib ###########################

df_x = df.loc[:, ['OR51A2_DEL', 'OR51A4_DEL', 'OR2T35_AMP', 'Lapatinib_ActArea']].dropna()
df_x['OR_MUT'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='OR_MUT', y='Lapatinib_ActArea')

df_x = df.loc[:, ['ZFP14_DEL', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='ZFP14_DEL', y='Lapatinib_ActArea')

df_x = df.loc[:, ['TNK2_MUT', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='TNK2_MUT', y='Lapatinib_ActArea')


df_x = df.loc[:, ['GPX3_ENSG00000211445.7', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='GPX3_ENSG00000211445.7', y='Lapatinib_ActArea')

df_x = df.loc[:, ['SYTL1_ENSG00000142765.13', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='SYTL1_ENSG00000142765.13', y='Lapatinib_ActArea')

df_x = df.loc[:, ['HER2_pY1248_Caution', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='HER2_pY1248_Caution', y='Lapatinib_ActArea')

df_x = df.loc[:, ['EGFR', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='EGFR', y='Lapatinib_ActArea')

df_x = df.loc[:, ['PAI-1', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='PAI-1', y='Lapatinib_ActArea')

df_x = df.loc[:, ['creatine', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='creatine', y='Lapatinib_ActArea')

df_x = df.loc[:, ['2-deoxycytidine', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='2-deoxycytidine', y='Lapatinib_ActArea')

df_x = df.loc[:, ['hsa-miR-558_nmiR00511.1', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-558_nmiR00511.1', y='Lapatinib_ActArea')

df_x = df.loc[:, ['hsa-miR-216a_nmiR00244.1', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-216a_nmiR00244.1', y='Lapatinib_ActArea')

df_x = df.loc[:, ['hsa-miR-141_nmiR00151.1', 'Lapatinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-141_nmiR00151.1', y='Lapatinib_ActArea')


############################ Erlotinib ###############################


df_x = df.loc[:, ['OR51A2_DEL', 'OR51A2_DEL', 'OR2T35_AMP', 'Erlotinib_ActArea']].dropna()
df_x['OR_MUT'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='OR_MUT', y='Erlotinib_ActArea')

df_x = df.loc[:, ['NAALADL2_DEL', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='NAALADL2_DEL', y='Erlotinib_ActArea')

df_x = df.loc[:, ['GALC_DEL', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='GALC_DEL', y='Erlotinib_ActArea')

df_x = df.loc[:, ['FLNA_MUT', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='FLNA_MUT', y='Erlotinib_ActArea')



df_x = df.loc[:, ['CORO2A_ENSG00000106789.8', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='CORO2A_ENSG00000106789.8', y='Erlotinib_ActArea')

df_x = df.loc[:, ['TSTD1_ENSG00000215845.6', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='TSTD1_ENSG00000215845.6', y='Erlotinib_ActArea')

df_x = df.loc[:, ['AKR1E2_ENSG00000165568.13', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='AKR1E2_ENSG00000165568.13', y='Erlotinib_ActArea')

df_x = df.loc[:, ['SYTL1_ENSG00000142765.13', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='SYTL1_ENSG00000142765.13', y='Erlotinib_ActArea')

df_x = df.loc[:, ['P-Cadherin_Caution', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='P-Cadherin_Caution', y='Erlotinib_ActArea')

df_x = df.loc[:, ['C-Raf_pS338', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='C-Raf_pS338', y='Erlotinib_ActArea')

df_x = df.loc[:, ['guanosine', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='guanosine', y='Erlotinib_ActArea')

df_x = df.loc[:, ['hsa-miR-429_nmiR00367.2', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-429_nmiR00367.2', y='Erlotinib_ActArea')

df_x = df.loc[:, ['hsa-miR-548a-3p_nmiR00480.1', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-548a-3p_nmiR00480.1', y='Erlotinib_ActArea')

df_x = df.loc[:, ['hsa-miR-216a_nmiR00244.1', 'Erlotinib_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-216a_nmiR00244.1', y='Erlotinib_ActArea')


############################# Irinotecan ####################################


df_x = df.loc[:, ['HLA-DRB6_AMP', 'HLA-DRB1_AMP', 'HLA-DRB5_AMP', 'Irinotecan_ActArea']].dropna()
df_x['HLA_AMP'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='HLA_AMP', y='Irinotecan_ActArea')

df_x = df.loc[:, ['KRAS.G12_13_MUT', 'KRAS_MUT', 'Irinotecan_ActArea']].dropna()
df_x['KRAS_MUT'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='KRAS_MUT', y='Irinotecan_ActArea')

df_x = df.loc[:, ['FBXL7_DEL', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='FBXL7_DEL', y='Irinotecan_ActArea')



df_x = df.loc[:, ['SLFN11_ENSG00000172716.12', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='SLFN11_ENSG00000172716.12', y='Irinotecan_ActArea')

df_x = df.loc[:, ['XIST_ENSG00000229807.5', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='XIST_ENSG00000229807.5', y='Irinotecan_ActArea')

df_x = df.loc[:, ['HNRNPA1_ENSG00000135486.13', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='HNRNPA1_ENSG00000135486.13', y='Irinotecan_ActArea')

df_x = df.loc[:, ['Chk1_Caution', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='Chk1_Caution', y='Irinotecan_ActArea')

df_x = df.loc[:, ['alpha-hydroxybutyrate', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='alpha-hydroxybutyrate', y='Irinotecan_ActArea')

df_x = df.loc[:, ['C34:2 DAG', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='C34:2 DAG', y='Irinotecan_ActArea')

df_x = df.loc[:, ['hsa-miR-574-5p_nmiR00526.1', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-574-5p_nmiR00526.1', y='Irinotecan_ActArea')

df_x = df.loc[:, ['hsa-miR-608_nmiR00562.1', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-608_nmiR00562.1', y='Irinotecan_ActArea')

df_x = df.loc[:, ['hsa-miR-22_nmiR00251.1', 'Irinotecan_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-22_nmiR00251.1', y='Irinotecan_ActArea')



############################### Paclitaxel ####################################


df_x = df.loc[:, ['EP300_MUT', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='EP300_MUT', y='Paclitaxel_ActArea')

df_x = df.loc[:, ['ADAM5_DEL', 'ADAM3A_DEL', 'Paclitaxel_ActArea']].dropna()
df_x['ADAM_DEL'] = (df_x.iloc[:, :-1]).sum(axis=1) > 0
plt.figure()
sns.swarmplot(data=df_x, x='ADAM_DEL', y='Paclitaxel_ActArea')

df_x = df.loc[:, ['ABCB1_ENSG00000085563.10', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='ABCB1_ENSG00000085563.10', y='Paclitaxel_ActArea')


df_x = df.loc[:, ['LEPREL2_ENSG00000110811.15', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='LEPREL2_ENSG00000110811.15', y='Paclitaxel_ActArea')


df_x = df.loc[:, ['XBP1_Caution', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='XBP1_Caution', y='Paclitaxel_ActArea')


df_x = df.loc[:, ['Bcl-xL', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='Bcl-xL', y='Paclitaxel_ActArea')


df_x = df.loc[:, ['AMPK_alpha_Caution', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='AMPK_alpha_Caution', y='Paclitaxel_ActArea')


df_x = df.loc[:, ['beta-alanine', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='beta-alanine', y='Paclitaxel_ActArea')


df_x = df.loc[:, ['hsa-miR-30a_nmiR00287.1', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-30a_nmiR00287.1', y='Paclitaxel_ActArea')

df_x = df.loc[:, ['hsa-miR-607_nmiR00561.1', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-607_nmiR00561.1', y='Paclitaxel_ActArea')

df_x = df.loc[:, ['hsa-miR-22_nmiR00251.1', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-22_nmiR00251.1', y='Paclitaxel_ActArea')

df_x = df.loc[:, ['hsa-let-7e_nmiR00005.1', 'Paclitaxel_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-let-7e_nmiR00005.1', y='Paclitaxel_ActArea')


############################# Panobinostat ######################################



df_x = df.loc[:, ['LINC00226_AMP', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='LINC00226_AMP', y='Panobinostat_ActArea')

df_x = df.loc[:, ['PRODH_DEL', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.swarmplot(data=df_x, x='PRODH_DEL', y='Panobinostat_ActArea')


df_x = df.loc[:, ['beta-Catenin_pT41_S45', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='beta-Catenin_pT41_S45', y='Panobinostat_ActArea')

df_x = df.loc[:, ['Akt_pS473', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='Akt_pS473', y='Panobinostat_ActArea')

df_x = df.loc[:, ['1-methylnicotinamide', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='1-methylnicotinamide', y='Panobinostat_ActArea')

df_x = df.loc[:, ['alpha-glycerophosphate', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='alpha-glycerophosphate', y='Panobinostat_ActArea')

df_x = df.loc[:, ['C22:6 CE', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='C22:6 CE', y='Panobinostat_ActArea')

df_x = df.loc[:, ['pantothenate', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='pantothenate', y='Panobinostat_ActArea')

df_x = df.loc[:, ['hsa-miR-24_nmiR00261.1', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-24_nmiR00261.1', y='Panobinostat_ActArea')

df_x = df.loc[:, ['hsa-let-7i_nmiR00008.1', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-let-7i_nmiR00008.1', y='Panobinostat_ActArea')

df_x = df.loc[:, ['hsa-miR-320d_nmiR00297.2', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-320d_nmiR00297.2', y='Panobinostat_ActArea')

df_x = df.loc[:, ['hsa-miR-22_nmiR00251.1', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.regplot(data=df_x, x='hsa-miR-22_nmiR00251.1', y='Panobinostat_ActArea')

df_x = df.loc[:, ['hsa-let-7e_nmiR00005.1', 'SKIN', 'Panobinostat_ActArea']].dropna()
plt.figure()
sns.lmplot(data=df_x, x='hsa-let-7e_nmiR00005.1', y='Panobinostat_ActArea')