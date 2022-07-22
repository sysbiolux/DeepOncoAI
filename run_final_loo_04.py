if __name__ == '__main__':
    from config import Config
    from functions import unpickle_objects
    from DBM_toolbox.modeling import validation
    config = Config("testall/config.yaml")

    final_data = unpickle_objects('f_testall_data_04.pkl')
    algos = config.raw_dict["modeling"]["general"]["algorithms"]
    metric = config.raw_dict["modeling"]["general"]["metric"]

    target = final_data.dataframe.columns[-1]

    loo_preds = validation.loo(final_data, algos=algos, metric=metric, targets_list=[target])
    config.save(to_save=loo_preds, name="f_testall_01_preds_04")
