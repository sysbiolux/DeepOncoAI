####################
### HOUSEKEEPING ###
####################
import _pickle
import argparse
import logging
import os

#TODO: consider numba for acceleration (useful?)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='chunk 1 running script')
    # # Optional arguments
    optional_arguments = parser.add_argument_group('optional arguments')
    optional_arguments.add_argument('--overwrite', action='store_true', default=False,
                                    help='overwrite existing pickles, use option to rerun all parts again')
    # input arguments
    input_arguments = parser.add_argument_group('Input arguments or paths (all required)')
    input_arguments.add_argument('-c', '--config', type=str, default=None, required=True,
                                 help='configuration yaml file')
    # Output arguments
    output_arguments = parser.add_argument_group('output arguments (all required)')
    output_arguments.add_argument('-o', '--output_dir', type=str, default=None, required=True,
                                  help='path where output results are written to')
    output_arguments.add_argument('-f', '--final_data', type=str, default=None, required=True,
                                  help='path where final_data is written to')
    args = parser.parse_args()
    return args

def pickle_objects(objects,location):
    """pickle objects at location"""
    with open(location, 'wb') as f:
        _pickle.dump(objects, f)
    f.close()
    return None



def unpickle_objects(location):
    """pickle objects at location"""
    with open(location, 'rb') as f:
        loaded_objects = _pickle.load(f)
    return loaded_objects

def main():
    args = parse_args()
    #make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #initiate logging
    logging.basicConfig(filename=os.path.join(args.output_dir,'run.log'), level=logging.INFO, filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
    #import Config Class instance
    from config import Config
    #instantiate config Class instance with configuration file
    config = Config(args.config)

    ###################################
    ### READING AND PROCESSING DATA ###
    ###################################
    logging.info("Reading data")
    #set location read_data_pickle where pickled read data will be stored.
    read_data_pickle = os.path.join(args.output_dir, 'read_data.pickle')
    if not os.path.exists(read_data_pickle) or args.overwrite:
        data, ActAreas, IC50s, dose_responses = config.read_data()
        objects = [data, ActAreas, IC50s, dose_responses]
        pickle_objects(objects, read_data_pickle)
    else:
        [data, ActAreas, IC50s, dose_responses] = unpickle_objects(read_data_pickle)

    #TODO: this could be a separate chunk, endpoint
    logging.info("Creating visualizations")
    figures_output_dir = os.path.join(args.output_dir,'figures')
    if not os.path.exists(figures_output_dir) or args.overwrite:
        os.makedirs(figures_output_dir)
        #TODO: check if we can skip this if done, what criteria to use here?
        config.visualize_dataset(data, ActAreas, IC50s, dose_responses, mode='pre',outputdir=figures_output_dir)

    logging.info("Filtering data")
    # TODO: this could be a separate chunk, does not depend on visualization
    # set location filtered_data_pickle where pickled filtered data will be stored.
    filtered_data_pickle = os.path.join(args.output_dir, 'filtered_data.pickle')
    if not os.path.exists(filtered_data_pickle) or args.overwrite:
        filtered_data, filters = config.filter_data(data)
        objects = [filtered_data, filters]
        pickle_objects(objects, filtered_data_pickle)
    else:
        [filtered_data, filters] = unpickle_objects(filtered_data_pickle)
    print(filtered_data.dataframe.shape)
    for omic in list(set(filtered_data.omic)):
        print(f"{omic}: {filtered_data.omic[filtered_data.omic == omic].shape[0]}")

    #TODO: this should be a separate chunk,
    logging.info("Selecting subsets for feature engineering")
    selected_subset_pickle = os.path.join(args.output_dir, 'selected_subset.pickle')
    if not os.path.exists(selected_subset_pickle) or args.overwrite:
        selected_subset = config.select_subsets(filtered_data)
        pickle_objects(selected_subset, selected_subset_pickle)
    else:
        selected_subset = unpickle_objects(selected_subset_pickle)

    logging.info("Engineering features")
    if selected_subset is not None:
        engineered_features = config.engineer_features(selected_subset)
        logging.info("Merging engineered features")
        engineered_data = filtered_data.merge_with(engineered_features)
    else:
        engineered_data = filtered_data

    logging.info("Quantizing targets")
    quantized_data = config.quantize(engineered_data, target_omic="DRUGS", IC50s=IC50s)

    final_data = quantized_data.normalize().optimize_formats()
    pickle_objects(final_data, args.final_data)
    #TODO: store data in pickle in output directory



if __name__ == '__main__':
    main()