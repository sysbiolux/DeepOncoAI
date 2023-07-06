import pandas as pd
import numpy as np
import logging
import _pickle
from pathlib import Path


def merge_and_clean(dataframe, series):
    """
    Removes all rows that contain NaN values in either the dataframe or series
    and returns the cleaned dataframe and series.
    """
    # drop na values
    new_df = dataframe.dropna(axis=0)
    new_series = series.dropna()

    common_index = new_df.index.intersection(new_series.index)
    new_df = new_df.loc[common_index]
    new_series = new_series.loc[common_index]

    # print log
    npos = sum(new_series == 1)
    nneg = sum(new_series == 0)
    n_dropped = dataframe.shape[0] - new_df.shape[0]
    logging.info(f"X: {new_df.shape[0]} samples and {new_df.shape[1]} features")
    logging.info(f"y: {new_series.size} samples, with {npos} positives and {nneg} negatives ({n_dropped} dropped)")

    return new_df, new_series


def recurse_to_float(weird_object):
    print(weird_object)
    if isinstance(weird_object, float) or isinstance(weird_object, int) or isinstance(weird_object, np.float16) or isinstance(weird_object, np.float32):
        return weird_object
    else:
        try:
            return recurse_to_float(weird_object[1])
        except:  # TODO: specify exception
            return recurse_to_float(weird_object[0])


def pickle_objects(objects, location):
    """pickle objects at location"""
    with open(location, "wb") as f:
        _pickle.dump(objects, f)
    f.close()
    return None


def unpickle_objects(location):
    """pickle objects at location"""
    with open(location, "rb") as f:
        try:
            loaded_objects = _pickle.load(f)
        except AttributeError:
            loaded_objects = pd.read_pickle(f)
    return loaded_objects


def define_combinations(combinations=None):
    """
    

    Parameters
    ----------
    combinations : TYPE
        DESCRIPTION.

    Returns
    -------
    comb_name : TYPE
        DESCRIPTION.
    results_folder : TYPE
        DESCRIPTION.

    """
    if combinations is None:
        name_str = []
        comb_name = []
    name_str = [x[0:3] for x in combinations]
    comb_name = ''.join([y for y in name_str])
    results_folder = f'{comb_name}'
    return comb_name, results_folder

def generate_folder_comb(res_filename:str, results_folder:str, additional_string=None):
    filename = res_filename
    fname_split = filename.split('_')
    file_identifier = fname_split[0]
    len_of_file = len(file_identifier)
    y = [file_identifier[i:i+3] for i in range(0,len_of_file,3)]
    base_omics_dict = dict()
    measures_dict = dict()
    disc_dict = dict()
    base_omics_dict['RNA'] = 'RNA'
    base_omics_dict['RPP'] = 'RPPA'
    base_omics_dict['DNA'] = 'DNA'
    measures_dict['EIG'] = 'EIGENVECTOR'
    measures_dict['BET'] = 'BETWEENNESS'                   
    measures_dict['CLO'] = 'CLOSENESS'
    measures_dict['PAG'] = 'PAGERANK'
    measures_dict['AVN'] = 'AVNEIGHBOUR'
    measures_dict['HAR'] = 'HARMONIC'
    measures_dict['INF'] = 'INFORMATION'
    measures_dict['CON'] = 'CONSTRAINT'
    measures_dict['ECC'] = 'ECCENTRICITY'
    measures_dict['SUB'] = 'SUBGRAPH'
    measures_dict['APP'] = 'APPROXCURRBET'
    measures_dict['CLI'] = 'CLIQUENO'
    measures_dict['SQU'] = 'SQUARECLUSTERING'
    measures_dict['DEG'] = 'DEGREECENT' 
    disc_dict['DIS'] = 'DISCRETIZED'
    
    
    base_omics_list = []
    measures_list = []
    disc_list = []
    for xi in y:
        if xi in base_omics_dict:
            base_omics_list.append(base_omics_dict[xi])
        elif xi in measures_dict:
            measures_list.append(measures_dict[xi])
        elif xi in disc_dict:
            disc_list.append(disc_dict[xi])
        else: 
            pass
    
    combinations = base_omics_list + measures_list + disc_list
    
    if additional_string is not None:
        results_folder = fname_split[0] + additional_string
    else:
        results_folder = fname_split[0]
    Path(f"{results_folder}/").mkdir(parents=True, exist_ok=True)
    return combinations, results_folder







def partial_dataset_filtering(final_dataframe=None, config=None):
    """
    

    Parameters
    ----------
    final_dataframe : dataset class (dataframe)
        dataframe with filtered RNA (after removing cross-correlated features). This function then filters out 
        the non-overlapping features from RNA and network measures features. Does it for all the measures and discretized
        data

    Returns
    -------
    None.

    """
    
    refiltered_data = final_dataframe.copy()
    for ref_omic in ['EIGENVECTOR',
                     'BETWEENNESS',
                     'CLOSENESS',
                     'PAGERANK',
                     'AVNEIGHBOUR',
                     'HARMONIC',
                     'INFORMATION',
                     'CONSTRAINT',
                     'ECCENTRICITY',
                     'SUBGRAPH',
                     'APPROXCURRBET',
                     'CLIQUENO',
                     'SQUARECLUSTERING',
                     'DEGREECENT',
                     'DISCRETIZED'
                     ]:
        print(ref_omic)
        refiltered_data = refiltered_data.filter_att(target_omic = 'RNA', reference_omic= ref_omic, separator='_')
        
    config.save(to_save=refiltered_data, name="Topo_PARTIAL")

