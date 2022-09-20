##############################
### HOUSEKEEPING FUNCTIONS ###
##############################

import _pickle

import pandas as pd


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
