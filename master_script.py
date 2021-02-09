
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')


# from DBM_toolbox.modeling.get_optimized_models import get_optimal_models

from config import Config


config = Config()

logging.info("Reading data")
data = config.read_data()

logging.info("Creating filters")
filters = config.create_filters(data)

logging.info("Applying filters")
filtered_data = data.apply_filters(filters=filters)

logging.info("Selecting subsets")
selected_subsets = config.select_subsets(filtered_data)

logging.info("Engineering features")
engineered_features = config.engineer_features(selected_subsets)

logging.info("Merging engineered features")
engineered_data = filtered_data.merge_with(engineered_features)

logging.info("Imputing missing data")
engineered_data = engineered_data.impute()

logging.info("Quantizing targets")
engineered_data = engineered_data.quantize(target_omic="DRUGS")

###debug
x = engineered_data.dataframe
target_series = engineered_data.dataframe['AEW541_ActArea']

logging.info("Getting optimized models")
optimal_algos = config.get_optimized_models(engineered_data)

print(optimal_algos)

for omic_db in optimal_algos.keys():
	print(omic_db)
	print(optimal_algos[omic_db])

#%%


logging.info("Creating best stack")
best_stack = config.get_best_stack(engineered_data, optimal_algos)

logging.info("Generating results")
config.generate_results(best_stack, optimal_algos)

