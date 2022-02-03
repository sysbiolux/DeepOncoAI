import argparse
import os
import logging
import _pickle


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='chunk 3 running script')
    # # Optional arguments
    optional_arguments = parser.add_argument_group('optional arguments')
    optional_arguments.add_argument('--overwrite', action='store_true', default=False,
                                    help='overwrite existing pickles, use option to rerun all parts again')
    # input arguments
    input_arguments = parser.add_argument_group('Input arguments or paths (all required)')
    input_arguments.add_argument('-c', '--config', type=str, default=None, required=True,
                                 help='configuration yaml file')
    input_arguments.add_argument('-i', '--input', type=str, default=None, required=True,
                                 help='input processed data file')
    input_arguments.add_argument('-m', '--models', type=str, default=None, required=True,
                                 help='input trained models file')
    # Output arguments
    output_arguments = parser.add_argument_group('output arguments (all required)')
    output_arguments.add_argument('-o', '--output_dir', type=str, default=None, required=True,
                                  help='path where output results are written to, '
                                       'subdirectories are automatically created!')
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
    # make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    #initiate logging
    logging.basicConfig(filename=os.path.join(args.output_dir,'run.log'), level=logging.INFO, filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
    #import Config Class instance
    from config import Config
    #instantiate config Class instance with configuration file
    config = Config(args.config)
    
    #######################
    ### TRAINING STACKS ###
    #######################
    
    logging.info("Creating best stacks")
    
    trained_stacks_pickle = os.path.join(args.output_dir, 'trained_stacks.pickle')
    
    models_pickle = args.models
    data_pickle = args.input
    
    [algos_dict, results_prim] = unpickle_objects(models_pickle)
    final_data = unpickle_objects(data_pickle)
    
    
    
    
    if not os.path.exists(trained_stacks_pickle) or args.overwrite:
        best_stacks, results_sec = config.get_best_stacks(models=algos_dict, dataset=final_data)
        objects = [best_stacks, results_sec]
        pickle_objects(objects, trained_stacks_pickle)
    else:
        [best_stacks, results_sec] = unpickle_objects(trained_stacks_pickle)


if __name__ == '__main__':
    main()