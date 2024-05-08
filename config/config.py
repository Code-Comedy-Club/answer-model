dataset = {
}

dataset["path"] = "./data"
dataset["path_raw"] = dataset["path"] + "/raw"
dataset["path_raw_es"] = "/es"
dataset["path_raw_ca"] = "/ca"
dataset["path_raw_en"] = "/en"

model = {
}

model["name"] = "model"
model["output"] = "./data/output"
model["datasets"] = [
    dataset["path_raw_es"],
    dataset["path_raw_ca"],
    dataset["path_raw_en"]
]