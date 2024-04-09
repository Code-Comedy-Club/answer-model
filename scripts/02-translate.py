import sys

sys.path.insert(0, 'config')
sys.path.insert(0, 'utils')

import os
import yaml
import config
import mtranslate
from utils import create_path

def translate(original_path: str = None, translated_path: str = None) -> None:
    """
    TODO
    Check if file exists
    """

    for d in os.listdir(original_path):
        if os.path.isdir(os.path.join(original_path, d)):
            translate(os.path.join(original_path, d), os.path.join(translated_path, d))
        
        if d.endswith("yml") and os.path.exists(os.path.join(translated_path, d)) == False:
            create_path(translated_path)

            with open(os.path.join(original_path, d), "r") as file:
                data_original = yaml.safe_load(file)
                data_translated = {}
                
                data_translated["category"] = data_original[0]["category"]
                data_translated["answers"] = [mtranslate.translate(answer, from_language = "es", to_language = "ca").replace("\n", " ") for answer in data_original[0]["answers"]]
                data_translated["questions"] = [mtranslate.translate(question, from_language = "es", to_language = "ca").replace("\n", " ") for question in data_original[0]["questions"]]
                
                if len(data_translated["answers"]) == len(data_original[0]["answers"]) and len(data_translated["questions"]) == len(data_original[0]["questions"]):
                    with open(os.path.join(translated_path, d), "w") as outfile:
                        yaml.dump([data_translated], outfile, sort_keys = False, allow_unicode = True, width = float("inf"))

def main():
    translate(config.dataset["path_raw_es"], config.dataset["path_raw_ca"])

if __name__ == "__main__":
    main()