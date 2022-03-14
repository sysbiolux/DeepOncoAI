def compare_datasets(dataset_1, dataset_2):
    dataframe_1 = dataset_1.to_pandas()
    dataframe_2 = dataset_2.to_pandas()

    overlap_columns = list(
        set(dataframe_1.columns).intersection(set(dataframe_2.columns))
    )
    only_present_in_1 = [
        column for column in list(dataframe_1.columns) if column not in overlap_columns
    ]
    only_present_in_2 = [
        column for column in list(dataframe_2.columns) if column not in overlap_columns
    ]

    result_dict = dict()

    for column in overlap_columns:
        result_dict[column] = {"mean": 0.1, "variance": 0.5, "min": 0.2, "max": 0.7}
        return
        # Compare two columns
