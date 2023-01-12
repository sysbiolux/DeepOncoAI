import pandas as pd
import numpy as np
import logging
import _pickle


def merge_and_clean(dataframe, series, axis=0):
    """
    Removes all rows that contain NaN values in either the dataframe or series
    and returns the cleaned dataframe and series.
    """
    # drop na values
    new_df = dataframe.dropna(axis=axis)
    new_series = series.dropna()

    # print log
    npos = sum(series == 1)
    nneg = sum(series == 0)
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
