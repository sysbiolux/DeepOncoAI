####################
### TRAIN MODELS ###
####################

import argparse
import os
import logging

from functions import pickle_objects, unpickle_objects


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="model training running script")
    # # Optional arguments
    optional_arguments = parser.add_argument_group("optional arguments")
    optional_arguments.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite existing pickles, use option to rerun all parts again",
    )
    # input arguments
    input_arguments = parser.add_argument_group(
        "Input arguments or paths (all required)"
    )
    input_arguments.add_argument(
        "-c",
        "--config",
        type=str,
        default="input\config.yaml",
        required=True,
        help="configuration yaml file",
    )
    input_arguments.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="input preprocessed data file",
    )
    # Output arguments
    output_arguments = parser.add_argument_group("output arguments (all required)")
    output_arguments.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="path where output results are written to, "
        "subdirectories are automatically created!",
    )
    output_arguments.add_argument(
        "-f",
        "--final_data",
        type=str,
        default=None,
        required=True,
        help="path where trained models are written to",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # initiate logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "run.log"),
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    # import Config Class instance
    from config import Config

    # instantiate config Class instance with configuration file
    config = Config(args.config)

    #######################
    ### TRAINING MODELS ###
    #######################
    logging.info("Training models...")

    trained_models_pickle = args.final_data
    logging.info("Getting optimized models")

    data_pickle = args.input
    data = unpickle_objects(data_pickle)

    if not os.path.exists(trained_models_pickle) or args.overwrite:
        models = config.get_models(dataset=data)  # TODO: get rid of sklearn warnings
        algos_dict, results_prim = config.get_best_algos(models)
        objects = [models, algos_dict]
        pickle_objects(objects, trained_models_pickle)
        filename = trained_models_pickle[:-4] + ".csv"
        results_prim.to_csv(filename)
    else:
        [models, algos_dict] = unpickle_objects(trained_models_pickle)

    logging.info("Models trained")

    # config.show_results(config, results_prim)


if __name__ == "__main__":
    main()
