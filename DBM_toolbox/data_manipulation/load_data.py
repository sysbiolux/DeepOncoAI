import os

import pandas as pd
import logging

from DBM_toolbox.data_manipulation.dataset_class import Dataset
from DBM_toolbox.data_manipulation import preprocessing


def read_data(folder: str, omic: str, database: str, nrows: int = None):
    omic_root = omic.split("_")[0]
    print(f'omic = {omic_root}')
    if database == "CCLE":
        filename = {
            "RNA": "CCLE_RNAseq_genes_rpkm_20180929.csv",
            "RNA-FILTERED": "AWS_filtered_RNA.csv",
            "MIRNA": "CCLE_miRNA_20181103.csv",
            "RPPA": "CCLE_RPPA_20181003.csv",
            "META": "CCLE_metabolomics_20190502.csv",
            "CNV": "placeholder",  # TODO: import file
            "DNA": "CCLE_MUT_CNA_AMP_DEL_binary_Revealer.csv",
            "DRUGS": "CCLE_NP24.2009_Drug_data_2015.02.24.csv",
            "CHROMATIN": "CCLE_GlobalChromatinProfiling_20181130.csv",
        }[omic_root]

    elif database == "GDSC":
        filename = {
            "RNA": "Cell_line_RMA_proc_basalExp.txt",
            "MIRNA": "placeholder",  # TODO: import file
            "CNV": "placeholder",  # TODO: import file
            "DNA": "placeholder",  # TODO: import file
            "DRUGS": "GDSC2_fitted_dose_response_25Feb20.csv",
        }[omic_root]
    elif database == "OWN":
        filename = {
            "PATHWAYS": "SPEED_Scores_namechange.csv",
            "EIGENVECTOR": "Combined_Eigen_T_5.csv",
            "BETWEENNESS": "Combined_Betweenness_T_5.csv",
            "CLOSENESS": "Combined_Closeness_T_5.csv",
            "PAGERANK": "Combined_PageRank_T_5.csv",
            "AVNEIGHBOUR": "Combined_Avg_neighbor_T_5.csv",
            "HARMONIC": "Combined_Harmonic_T_5.csv",
            "INFORMATION": "Combined_Information_T_5.csv",
            "CONSTRAINT": "Combined_Constraint_T_5.csv",
            "ECCENTRICITY":"Combined_Eccentricity_T_5.csv",
            "SUBGRAPH": "Combined_Subgraphcentrality_T_5.csv",
            "APPROXCURRBET": "Combined_Approxcurrbet_T_5.csv",
            "CLIQUENO": "Combined_Cliqueno_T_5.csv",
            "SQUARECLUSTERING": "Combined_Squareclustering_T_5.csv",
            "DEGREECENT":"Combined_DegreeCent_T_5.csv",
            "DISCRETIZED": "Discretized_combined_T_5_all_cancers.csv"
        }[omic_root]
       
    else:
        logging.info(f"load_data.py/read_data: Database not recognized: {database}")
    file_string, file_extension = os.path.splitext(filename)

    print(f"file: {filename}")

    if file_extension == ".csv":
        dataframe = pd.read_csv(os.path.join(folder, filename), nrows=nrows)
    elif file_extension == ".txt":
        pass  ## TODO: implement here
    elif file_extension in [".xlsx", ".xls"]:
        dataframe = pd.read_excel(os.path.join(folder, filename), engine="openpyxl")
        pass  ## TODO: test excel imports

    dataset = Dataset(dataframe=dataframe, omic=omic, database=database)

    print('...pre-processing...')

    if omic_root == "DRUGS":
        dataset = preprocessing.reformat_drugs(dataset)
    else:
        logging.info(f"load_data.py/read_data: Dataset loaded, pre-processing...")
        dataset = preprocessing.preprocess_data(dataset)
        logging.info(f"load_data.py/read_data: Dataset pre-processed")
    return dataset
