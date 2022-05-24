
import argparse
import os
import logging

from functions import pickle_objects, unpickle_objects


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="explaining models running script")
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
    input_arguments.add_argument(
        "-m",
        "--models",
        type=str,
        default=None,
        required=True,
        help="input trained models file",
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
        help="path where filtered data is written to",
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
        filemode="a",
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # import Config Class instance
    from config import Config

    # instantiate config Class instance with configuration file
    config = Config(args.config)

    #########################
    ### EXPLAINING MODELS ###
    #########################

    logging.info("Retrieving features...")

    models_explanations_pickle = args.final_data

    models_pickle = args.models
    data_pickle = args.input

    models, algos_dict = unpickle_objects(models_pickle)
    data = unpickle_objects(data_pickle)

    if not os.path.exists(models_explanations_pickle) or args.overwrite:
        explanations = config.retrieve_features(
            trained_models=models, dataset=data
        )
        objects = explanations
        pickle_objects(objects, models_explanations_pickle)
        for target in explanations.keys():
            for omic in explanations[target].keys():
                filename = "explanations_" + target + "_" + omic + ".csv"
                explanations[target][omic].to_csv(os.path.join(args.output_dir, filename))
    else:
        explanations = unpickle_objects(models_explanations_pickle)
    logging.info("Model explanations performed")


if __name__ == "__main__":
    main()
