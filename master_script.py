####################
### HOUSEKEEPING ###
####################

import logging
from functions import unpickle_objects
import pandas as pd
#import seaborn as sns
#from matplotlib import pyplot as plt

from config import Config

# from DBM_toolbox.data_manipulation import dataset_class
# from DBM_toolbox.interpretation import gsea

logging.basicConfig(
    filename="run_alldrugs_01.log",
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

config = Config("testall/config.yaml")

###################################
### READING AND PROCESSING DATA ###
###################################

logging.info("Reading data")
data, ActAreas, IC50s, dose_responses = config.read_data()

# logging.info("Creating visualizations")
# config.visualize_dataset(data, ActAreas, IC50s, dose_responses, mode="pre")

logging.info("Filtering data")
filtered_data, filters = config.filter_data(data)

#####

logging.info("Selecting subsets for feature engineering")
selected_subset = config.select_subsets(filtered_data)

logging.info("Engineering features")
engineered_features = config.engineer_features(filtered_data)

logging.info("Merging engineered features")
engineered_data = filtered_data.merge_with(engineered_features)

logging.info("Quantizing targets")
quantized_data = config.quantize(engineered_data, target_omic="DRUGS", IC50s=IC50s)

final_data = quantized_data.normalize().optimize_formats()
config.save(to_save=final_data, name="f_testall_01_data")

missing_data = final_data.dataframe.loc[:, final_data.dataframe.isnull().any(axis=0)]

######

logging.info("Getting optimized models")

# trained_models = config.get_models(dataset=final_data, method="standard")
# config.save(to_save=trained_models, name="f_test67_2_models")
# preds = config.loo(final_data)
# config.save(to_save=preds, name="f_test77_preds")

# # build models based on primary predictions:
# trained_sec_models = config.get_models(dataset=preds, method="standard")
#
# logging.info("Creating best stacks")
# results_sec = config.get_stacks(dataset=final_data, models_dict=trained_models)
# config.save(to_save=results_sec, name="f_test77_stack_results")

###
########################
logging.info("single-loo")
loo_preds = config.loo(final_data)
config.save(to_save=loo_preds, name="f_testsmall_preds")
loo_preds = unpickle_objects("f_testsmall_preds_2022-07-18-09-54-08-286211.pkl")
#############

logging.info("final validation")
results_valid = config.get_valid_loo(original_dataset=final_data)
config.save(to_save=results_valid, name="f_testsmall_valid")







######################################################################
### rerun from saved data

# final_data = unpickle_objects("f_test2_data_2022-06-29-15-36-18-152559.pkl")
# trained_models = unpickle_objects("f_test2_models_2022-06-29-17-20-07-296704.pkl")
# results_sec = unpickle_objects("f_test2_stack_results_2022-07-04-10-37-53-902000.pkl")

######################################################################

expl_dict = config.retrieve_features(trained_models=trained_models, dataset=final_data)
config.save(to_save=expl_dict, name="f_testsmall_expl_dict")

print("DONE")

##############################################################################

models, algos_dict = config.get_best_algos(trained_models)
# config.show_results(config, algos_dict)

results = pd.DataFrame(index=list(trained_models.keys()))
x = ["TYPE only", "RPPA", "RNA", "DNA", "PATHWAYS", "META", "MIRNA"]

for target in trained_models.keys():
    print(target)
    df = algos_dict[algos_dict["target"] == target]
    df_best_type = df[df["omic"] == "TYPE"].sort_values(by="perf", ascending=False)
    results.loc[target, "best_type_algo"] = df_best_type.iloc[0, 2]
    results.loc[target, "best_type_perf"] = df_best_type.iloc[0, -3]
    stacks_add = results_sec[target].loc[:, "TYPE_" + df_best_type.iloc[0, 2]].sort_values(ascending=False) # - df_best_type.iloc[0, -3]

    for omic in x[1:]:
        spec_df = stacks_add[stacks_add.index.str.startswith(omic)]
        results.loc[target, omic + "_best_stack_algo"] = spec_df.index[0].split('_')[-1]
        results.loc[target, omic + "_best_stack_add"] = spec_df[0]

