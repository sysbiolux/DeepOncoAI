######################
### VISUALIZE DATA ###
######################
import argparse
import os
import logging

from functions import pickle_objects, unpickle_objects


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="visualization chunk running script")
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
        "-i", "--input", type=str, default=None, required=True, help="input data file"
    )
    input_arguments.add_argument(
        "-m", "--mode", type=str, default=None, required=True, help="analysis mode"
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
    # output_arguments.add_argument('-f', '--final_data', type=str, default=None, required=True,
    #                               help='path where final_data is written to')
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

    logging.info("Creating visualizations...")

    data_pickle = args.input
    data = unpickle_objects(data_pickle)
    mode = args.mode

    figures_output_dir = os.path.join(args.output_dir, "figures")
    if not os.path.exists(figures_output_dir) or args.overwrite:
        os.makedirs(figures_output_dir)
        config.visualize_dataset(
            data,
            ActAreas,
            IC50s,
            dose_responses,
            mode=mode,
            outputdir=figures_output_dir,
        )
    logging.info("Visualizations created")


if __name__ == "__main__":
    main()
