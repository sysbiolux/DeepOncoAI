####################
### DATA LOADING ###
####################
import argparse
import os
import logging

from functions import pickle_objects, unpickle_objects

# TODO: consider numba for acceleration (useful?)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="loading chunk running script")
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
    # Output arguments
    output_arguments = parser.add_argument_group("output arguments (all required)")
    output_arguments.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="path where output results are written to",
    )
    output_arguments.add_argument(
        "-f",
        "--final_data",
        type=str,
        default=None,
        required=True,
        help="path where raw_data is written to",
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

    logging.info("*** DeepOncoAI *** commencing analysis...")
    logging.info("Loading datasets...")
    # set location read_data_pickle where pickled read data will be stored.
    raw_data_pickle = args.final_data
    if not os.path.exists(raw_data_pickle) or args.overwrite:
        (
            data,
            ActAreas,
            IC50s,
            dose_responses,
        ) = config.read_data()  # TODO: get rid of Pandas warnings
        objects = [data, ActAreas, IC50s, dose_responses]
        pickle_objects(objects, raw_data_pickle)
    else:
        [data, ActAreas, IC50s, dose_responses] = unpickle_objects(raw_data_pickle)
    logging.info("Data loading completed")


if __name__ == "__main__":
    main()
