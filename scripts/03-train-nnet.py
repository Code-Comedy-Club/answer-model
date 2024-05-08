import sys

sys.path.insert(0, 'config')
sys.path.insert(0, 'utils')

import os
import json
import config
import joblib
import numpy as np
from tensorflow import keras
from utils import import_data_as_pandas_dataframe, split_data, save_matrix_as_png, create_path

class LabelEncoder:
    """
    Class for encoding categorical labels into numerical values.
    """

    def __init__(self, labels: np.ndarray) -> None:
        """
        Initialize LabelEncoder.

        Parameters:
        - labels (np.ndarray): Array of labels to be encoded.
        """

        self.labels = labels        

    def fit_transform(self) -> np.ndarray:
        """
        Fit the encoder to the labels and transform them into numerical values.

        Returns:
        - np.ndarray: Transformed labels as an array of integers.
        """
        
        label_map = {}  # Store label mappings
        unique_labels = set(self.labels)  # Get unique labels

        # Create a mapping
        for i, label in enumerate(unique_labels):
            label_map[label] = i

        # Transform labels
        transformed_labels = np.zeros(len(self.labels), dtype = np.int32)
        
        for i, label in enumerate(self.labels):
            transformed_labels[i] = label_map[label]

        return transformed_labels

class Algorithm(object):
    def __init__(self, *, data, vocab_size: int, embedding_dim: int, max_length: int, trunc_type: str, padding_type: str, oov_tok: str, optimizer: str, loss: str, metric_names: list[str], epochs: int, batch_size: int, model_name: str, model_path: str):
        # Data
        self.train_data, self.test_data = split_data(data)

        # Hyperparameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.trunc_type = trunc_type
        self.padding_type = padding_type
        self.oov_tok = oov_tok
        self.optimizer = optimizer
        self.loss = loss
        self.metric_names = metric_names
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Save
        self.model_path = model_path
        self.model_name = model_name

        # Tokenizer
        self.tokenizer = None

    def train(self):
        labels_train_data = LabelEncoder(self.train_data['label'].values).fit_transform() # Encode labels to numbers
        features_train_data = self.train_data['text'].values

        # Save labels as dictionary
        labels_number = list(set(labels_train_data))
        labels_string = list(set(self.train_data['label'].values))

        self.labels_dict = {labels_number[i]: labels_string[i] for i in range(len(labels_number))}

        # Tokenization
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words = self.vocab_size, split = " ", oov_token = self.oov_tok)
        self.tokenizer.fit_on_texts(features_train_data)

        tokens_train_data = self.tokenizer.texts_to_sequences(features_train_data)

        # Truncate
        truncated_train_data = keras.preprocessing.sequence.pad_sequences(tokens_train_data, maxlen = (self.max_length))

        # Layers
        # TODO: Maybe parameterize this??
        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length = self.max_length))
        self.model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences = True)))
        self.model.add(keras.layers.Conv1D(256, 5, activation="relu"))
        self.model.add(keras.layers.MaxPooling1D(5))
        self.model.add(keras.layers.Conv1D(256, 5, activation="relu"))
        self.model.add(keras.layers.GlobalMaxPooling1D())
        self.model.add(keras.layers.Dense(512, activation="relu"))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(256, activation="relu"))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(128, activation="relu"))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(len(set(labels_train_data)), activation = "softmax"))

        # Compilation
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metric_names)

        # Actual training
        self.model.fit(truncated_train_data, labels_train_data, epochs = self.epochs, batch_size = self.batch_size)
    
    def test(self) -> None:
        labels_test_data = self.test_data['label'].values
        features_test_data = self.test_data['text'].values

        # Tokenization
        tokens_test_data = self.tokenizer.texts_to_sequences(features_test_data)

        # Truncate
        truncated_test_data = keras.preprocessing.sequence.pad_sequences(tokens_test_data, maxlen=self.max_length)

        # Predictions
        predictions = np.argmax(self.model.predict(truncated_test_data), axis=1)
        predicted_categories = [self.labels_dict[i] for i in predictions]

        self.test_results = {str(label): {str(item): 0 for item in self.labels_dict.keys()} for label in self.labels_dict.keys()}

        for test, expected_category, predicted_category in zip(features_test_data, labels_test_data, predictions):
            for key, value in self.labels_dict.items():
                if value == expected_category:
                    self.test_results[str(key)][str(predicted_category)] += 1
            
    def save(self) -> None:
        """
        Save the model, tokenizer, and label dictionary to files.
        """
        
        create_path(self.model_path)  # Ensure that the model path exists

        # Save the model
        self.model.save(os.path.join(self.model_path, self.model_name + ".keras"))

        # Save the tokenizer as a JSON file
        with open(os.path.join(self.model_path, "tokenizer.json"), "w") as json_file:
            json_file.write(self.tokenizer.to_json())

        # Save the keys as a JSON file
        str_labels_dict = {int(key): value for key, value in self.labels_dict.items()}
        
        with open(os.path.join(self.model_path, "labels.json"), "w") as json_file:
            json.dump(str_labels_dict, json_file)
        
        save_matrix_as_png(self.test_results, self.model_path + "/results.png")
    
    def play(self):
        while True:
            text = input("Introduce un texto: ")

            if text == "exit":
                break

            # TODO: Try to fix this
            # Tokenization
            tokens_test_data = self.tokenizer.texts_to_sequences([text])

            # Truncate
            truncated_test_data = keras.preprocessing.sequence.pad_sequences(tokens_test_data, maxlen = self.max_length)

            predictions = np.argmax(self.model.predict(truncated_test_data)[0])

            print(predictions)

            print(f"Prediction: {self.labels_dict[predictions]}")

if __name__ == '__main__':

    for dataset in config.model["datasets"]:
        # Initialize Algorithm instance with training and testing data
        algorithm = Algorithm(
            data            = import_data_as_pandas_dataframe(config.dataset["path_raw"] + dataset),
            vocab_size      = 100_000,
            embedding_dim   = 64,
            max_length      = 100,
            trunc_type      = "post",
            padding_type    = "post",
            oov_tok         = "<OOV>",
            optimizer       = "adam",
            loss            = "sparse_categorical_crossentropy",
            metric_names    = ["accuracy"],
            epochs          = 50,
            batch_size      = 64,
            model_name      = config.model["name"],
            model_path      = config.model["output"] + dataset
        )

        algorithm.train()
        algorithm.test()
        algorithm.save()
        algorithm.play()
