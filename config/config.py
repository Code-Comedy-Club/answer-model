dataset = {
}

dataset["path"] = "./data"
dataset["path_raw"] = dataset["path"] + "/raw"
dataset["path_raw_es"] = dataset["path_raw"] + "/es"
dataset["path_raw_ca"] = dataset["path_raw"] + "/ca"
dataset["path_raw_en"] = dataset["path_raw"] + "/en"

model = {
}

model["dataset_es"] = dataset["path_raw_es"]
model["dataset_ca"] = dataset["path_raw_ca"]
model["dataset_en"] = dataset["path_raw_en"]
model["output"] = "./data/output"
model["name"] = "model.joblib"