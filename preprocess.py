##########################
### DATA PREPROCESSING ###
##########################
import argparse
import os
import logging

from functions import pickle_objects, unpickle_objects


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="preprocessing chunk running script")
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
        help="input filtered data file",
    )
    input_arguments.add_argument(
        "-r",
        "--input_raw",
        type=str,
        default=None,
        required=True,
        help="input raw data file",
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
        help="path where preprocessed data is written to",
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

    data_pickle = args.input
    [filtered_data, filters] = unpickle_objects(data_pickle)

    logging.info(
        "(snake) Preprocessing 1/2: Selecting subsets and feature engineering..."
    )
    selected_subset_pickle = os.path.join(args.output_dir, "selected_subset.pickle")
    if not os.path.exists(selected_subset_pickle) or args.overwrite:
        selected_subset = config.select_subsets(filtered_data)
        if selected_subset is not None:
            engineered_features = config.engineer_features(selected_subset)
            logging.info("Merging engineered features")
            engineered_data = filtered_data.merge_with(engineered_features)
            pickle_objects(selected_subset, selected_subset_pickle)
        else:
            engineered_data = filtered_data

    print(engineered_data)

    logging.info("(snake) Preprocessing 2/2: Quantizing targets...")
    raw_data_pickle = args.input_raw
    [data, ActAreas, IC50s, dose_responses] = unpickle_objects(raw_data_pickle)
    quantized_data = config.quantize(engineered_data, target_omic="DRUGS", IC50s=IC50s)

    final_data = quantized_data.normalize().optimize_formats()
    pickle_objects(final_data, args.final_data)
    logging.info("(snake) Data preprocessing completed")


if __name__ == "__main__":
    main()
