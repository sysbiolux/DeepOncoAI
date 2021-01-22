import yaml

from DBM_toolbox.data_manipulation import load_data


class Config:
    def __init__(self):
        with open("config.yaml") as f:
            self.raw_dict = yaml.load(f)

    def read_data(self):
        omic = self.raw_dict["data"]["omics"][0]
        full_dataset = load_data.read_data('data', omic=omic["name"], database=omic["database"])
        for omic in self.raw_dict["data"]["omics"][1:]:
            additional_dataset = load_data.read_data('data', omic=omic["name"], database=omic["database"])
            full_dataset = full_dataset.merge_with(additional_dataset)
        return full_dataset

    def create_filters(self):
        return []
