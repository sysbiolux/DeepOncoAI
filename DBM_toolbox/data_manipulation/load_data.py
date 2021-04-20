import os

import pandas as pd

from DBM_toolbox.data_manipulation.dataset_class import Dataset
from DBM_toolbox.data_manipulation import preprocessing

def read_data(folder:str, omic:str, database:str, nrows:int=None, keywords:str=None):
	omic_root = omic.split('_')[0]
	if database == 'CCLE':
		filename = {
			'RNA' : 'CCLE_RNAseq_genes_rpkm_20180929.csv',
			'MIRNA': 'CCLE_miRNA_20181103.csv',
			'RPPA' : 'CCLE_RPPA_20181003.csv',
			'META' : 'CCLE_metabolomics_20190502.csv',
			'CNV' : 'placeholder', #TODO: import file
			'DNA' : 'placeholder', #TODO: import file
			'DRUGS': 'CCLE_NP24.2009_Drug_data_2015.02.24.csv',
			}[omic_root]
		
	elif database == 'GDSC':
		filename = {
			'RNA' : 'Cell_line_RMA_proc_basalExp.txt',
			'MIRNA' : 'placeholder', #TODO: import file
			'CNV' : 'placeholder', #TODO: import file
			'DNA' : 'placeholder', #TODO: import file
			'DRUGS' : 'GDSC2_fitted_dose_response_25Feb20.csv',
			}[omic_root]
	elif database == 'OWN':
		filename = {
			'PATHWAYS' : 'SPEED_Scores_namechange.csv',
			'TOPOLOGY' : 'CCLE_SKIN_eigenvector.csv', #TODO: import file (need more samples)
			}[omic_root]
	file_string, file_extension = os.path.splitext(filename)
	
	if file_extension == '.csv':
		dataframe = pd.read_csv(os.path.join(folder, filename), nrows=nrows)
	elif file_extension == '.txt':
		pass ## TODO: implement here
	elif file_extension in ['.xlsx', '.xls']:
		dataframe = pd.read_excel(os.path.join(folder, filename), engine='openpyxl')
		pass ## TODO: implement here
	
	dataset = Dataset(dataframe=dataframe, omic=omic, database=database)
	
	if omic_root == 'DRUGS':
		dataset = preprocessing.reformat_drugs(dataset)
		if keywords is not None:
			for this_keyword in keywords:
				dataset = preprocessing.select_drug_metric(dataset, this_keyword)
	else:
		dataset = preprocessing.preprocess_data(dataset)
		
	return dataset
