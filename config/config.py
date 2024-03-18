dataset = {
}

dataset["path"] = "./data"
dataset["path_raw"] = dataset["path"] + "/raw"
dataset["path_raw_es"] = dataset["path_raw"] + "/es"
dataset["path_raw_ca"] = dataset["path_raw"] + "/ca"

model = {
}

model["dataset_es"] = dataset["path_raw_es"]
model["dataset_ca"] = dataset["path_raw_ca"]
model["output"] = "./data/output"
model["name"] = "model.joblib"