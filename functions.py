##############################
### HOUSEKEEPING FUNCTIONS ###
##############################

import _pickle


def pickle_objects(objects, location):
    """pickle objects at location"""
    with open(location, "wb") as f:
        _pickle.dump(objects, f)
    f.close()
    return None


def unpickle_objects(location):
    """pickle objects at location"""
    with open(location, "rb") as f:
        loaded_objects = _pickle.load(f)
    return loaded_objects