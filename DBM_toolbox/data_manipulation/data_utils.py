import pandas as pd
import numpy as np
import logging
import _pickle


def merge_and_clean(dataframe, series):
    index1 = series.index[series.apply(np.isnan)]
    index2 = dataframe.index[dataframe.apply(np.isnan).any(axis=1)]
    indices_to_drop = index1.union(index2)
    n_dropped = len(indices_to_drop)
    npos = sum(series == 1)
    nneg = sum(series == 0)

    new_df = dataframe.drop(indices_to_drop)
    new_series = series.drop(indices_to_drop)

    logging.info(
        f"X: {dataframe.shape[0]} samples and {dataframe.shape[1]} features"
    )
    logging.info(f"y: {series.size} samples, with {npos} positives and {nneg} negatives ({n_dropped} dropped)")

    return new_df, new_series


def recurse_to_float(weird_object):
    print(weird_object)
    if isinstance(weird_object, float) or isinstance(weird_object, int) or isinstance(weird_object, np.float16) or isinstance(weird_object, np.float32):
        return weird_object
    else:
        try:
            return recurse_to_float(weird_object[1])
        except:
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
