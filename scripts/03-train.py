import sys

sys.path.insert(0, 'config')
sys.path.insert(0, 'utils')

import os
import config
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import import_data_as_pandas_dataframe, split_data, print_matrix, save_matrix_as_png, create_path

class Algorithm(object):
    def __init__(self, *, data, model_name, model_path):
        """
        TODO add types
        Initialize the Algorithm class with training and testing data.

        Parameters:
        - data: Pandas DataFrame containing the data
        - model_name: Name of the model
        - model_path: Path to save the model
        """
        
        self.train_data, self.test_data = split_data(data)
        self.model_name = model_name
        self.model_path = model_path

        # Initialize Support Vector Machine (SVM) model with a linear kernel
        self.model = SVC(kernel = 'linear', probability = True)
        
        # Initialize TF-IDF tokenizer for text data
        self.tokenizer = TfidfVectorizer()

    def train(self):
        """
        Train the SVM model using the TF-IDF transformed features of the training data.
        """
        
        features_train_data = self.tokenizer.fit_transform(self.train_data["text"])
        labels_train_data = self.train_data["label"]

        # Fit the SVM model
        self.model.fit(features_train_data, labels_train_data)
    
    def test(self) -> None:
        """
        Test the trained model on the test dataset and calculate evaluation metrics.
        """
        
        labels_test_data = self.test_data["label"].values
        features_test_data = self.test_data["text"].values

        # Tokenization
        tokens_test_data = self.tokenizer.transform(features_test_data)

        # Predictions
        predictions = [self.model.predict(text) for text in tokens_test_data]

        # Generate matrix
        self.test_results = {label: {item: 0 for item in labels_test_data} for label in set(labels_test_data)}

        for test, expected_category, predicted_category in zip(features_test_data, labels_test_data, predictions):
            self.test_results[expected_category][predicted_category[0]] += 1
            
    def save(self) -> None:
        """
        Save the trained model to a specified file path.
        """

        print(f"[INFO] Saving the model to {self.model_name}")

        create_path(self.model_path)
                
        joblib.dump(self.model, os.path.join(self.model_path, self.model_name))
        joblib.dump(self.tokenizer, os.path.join(self.model_path, "tokenizer.joblib"))

        save_matrix_as_png(self.test_results, self.model_path + "/results.png")
        
        print("[INFO] Model saved successfully")
    
    def play(self):
        """
        TODO
        """
        
        while True:
            text = input("Introduce un texto: ")

            if text == "exit":
                break

            tokens = self.tokenizer.transform([text])
            prediction = self.model.predict(tokens)

            print(f"Prediction: {prediction}")

if __name__ == '__main__':

    for dataset in config.model["datasets"]:
        # Initialize Algorithm instance with training and testing data
        algorithm = Algorithm(
            data            = import_data_as_pandas_dataframe(config.dataset["path_raw"] + dataset),
            model_name      = config.model["name"],
            model_path      = config.model["output"] + dataset
        )

        algorithm.train()
        algorithm.test()
        algorithm.save()
        #algorithm.play()
