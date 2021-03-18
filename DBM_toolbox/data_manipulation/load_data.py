import os

import pandas as pd

from DBM_toolbox.data_manipulation.dataset_class import Dataset
from DBM_toolbox.data_manipulation import preprocessing

def read_data(folder, omic, database, nrows=None, keywords=None):
	omic_root = omic.split('_')[0]
	if database == 'CCLE':
		filename = {
			'RNA' : 'CCLE_RNAseq_genes_rpkm_20180929.csv',
			'MIRNA': 'CCLE_miRNA_20181103.csv',
			'RPPA' : 'CCLE_RPPA_20181003.csv',
			'CNV' : 'placeholder',
			'DNA' : 'placeholder',
			'DRUGS': 'CCLE_NP24.2009_Drug_data_2015.02.24.csv',
			}[omic_root]
		
	elif database == 'GDSC':
		filename = {
			'RNA' : 'Cell_line_RMA_proc_basalExp.txt',
			'MIRNA' : 'placeholder',
			'CNV' : 'placeholder',
			'DNA' : 'placeholder',
			'DRUGS' : 'GDSC2_fitted_dose_response_25Feb20.csv',
			}[omic_root]
	
	file_string, file_extension = os.path.splitext(filename)
	
	if file_extension == '.csv':
		dataframe = pd.read_csv(os.path.join(folder, filename), nrows=nrows)
	elif file_extension == '.txt':
		pass ## TODO: implement here
	elif file_extension == '.xlsx':
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
