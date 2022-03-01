import argparse
import os
import logging
import _pickle


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='chunk 2 running script')
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
    ### TRAINING MODELS ###
    #######################
    
    trained_models_pickle = os.path.join(args.output_dir, 'trained_models.pickle')
    logging.info("Getting optimized models")
    
    data_pickle = args.input
       
    data = unpickle_objects(data_pickle)
    
    if not os.path.exists(trained_models_pickle) or args.overwrite:
        models = config.get_models(dataset=data)
        algos_dict, results_prim = config.get_best_algos(models)
        objects = [algos_dict, results_prim]
        pickle_objects(objects, trained_models_pickle)
    else:
        [algos_dict, results_prim] = unpickle_objects(trained_models_pickle)
        
        
    # config.show_results(config, results_prim)


if __name__ == '__main__':
    main()