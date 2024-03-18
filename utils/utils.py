import os
import yaml
import df2img
import shutil
import random
import numpy as np
import pandas as pd

def create_path(p: str = None) -> bool:
    """
    Create a directory path if it does not exist.

    Parameters:
    - p (str): The path to be created.

    Returns:
    - bool: True if the path is created or already exists, False otherwise.
    """

    # Check if the path is not specified
    if p == None:
        print("[INFO] No path specified")
        return False
    
    # Check if the path already exists
    if os.path.exists(p):
        print(f"[INFO] Path {p} already exists")
        return True
    
    try:
        # Create the directory path
        os.makedirs(p)
        print(f"[INFO] Path {p} created successfully")
        return True

    except Exception as e:
        # Handle potential errors during path creation
        print(f"[ERROR] Unable to create path {p}: {e}")
        return False

def delete_path(p: str = None, f: bool = False) -> bool:
    """
    Delete a directory path if it does exist.

    Parameters:
    - p (str): The path to be deleted.

    Returns:
    - bool: True if the path is deleted or does not already exist, False otherwise.
    """

    # Check if the path is not specified
    if p == None:
        print("[INFO] No path specified")
        return False
    
    # Check if the path does not exist
    if not os.path.exists(p):
        print(f"[INFO] Path {p} does not exist")
        return True
    
    try:
        # Check if the path is empty and if the user wants to force it
        if os.listdir(p) != 0 and f == False:
            raise Exception()

        shutil.rmtree(p)
        print(f"[INFO] Path {p} deleted successfully")

        return True
    except Exception as e:
        # Handle potential errors during path creation
        print(f"[ERROR] Unable to delete path {p}: {e}")
        return False

def import_data_as_pandas_dataframe(path: str = None) -> pd.DataFrame:
    """
    Recursively imports data from YAML files in the specified path and returns a pandas DataFrame.
    """

    # Initialize an empty DataFrame with columns for label and text
    df = pd.DataFrame(columns = ["label", "text"])

    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            # Recursively call the function for subdirectories and concatenate the resulting DataFrame
            df = pd.concat([df, import_data_as_pandas_dataframe(os.path.join(path, d))])
        
        if d.endswith("yml"):
            with open(os.path.join(path, d), "r") as file:
                data = yaml.safe_load(file)

                label = data[0]["category"]
                questions = data[0]["questions"]

                for question in questions:
                    df = df._append({"label": label, "text": clean_word(question)}, ignore_index = True)
    
    return df

def clean_word(word: str):
    """
    TODO: Write this
    """

    word = word.lower()

    for i, x in zip(["à", "è", "ì", "ò", "ù", "á", "é", "í", "ó", "ú", "ç", "ñ"], ["a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "c", "n"]):
        word = word.replace(i, x)

    for i in [".", ",", ";", ":", "-", "_", "´", "{", "}", "*", "/", "`", "[", "]", "^", "+", "º", "ª", "!", "|", '"', "@", "#", "·", "$", "~", "%", "½", "¬", "&", "/", "=", "?", "¿", "¡"]:
        word = word.replace(i, "")
    
    return word

def print_matrix(results: dict[str, dict[str, int]], column_width: int = 10) -> None:
    """
    Print a matrix of results with given column width.
    """

    # Get all unique labels
    labels = list(results.keys())

    # Print table header
    print(" " * column_width, end=' ')

    for label in labels:
        print(f"{label:^{column_width}}", end=' ')
    
    print()

    # Print table rows
    for label in labels:
        print(f"{label:<{column_width}}", end=' ')
        
        for inner_label in labels:
            print(f"{results[label].get(inner_label, 0):^{column_width}}", end=' ')
        
        print()

def split_data(data, test_size = 0.2):
    """
    Split data into train and test datasets.
    
    Parameters:
    - data: DataFrame, input data with "label" and "text" columns
    - test_size: float, proportion of the data to include in the test dataset (default is 0.2)
    
    Returns:
    - train_data: DataFrame, training dataset
    - test_data: DataFrame, test dataset
    """

    train_data = pd.DataFrame(columns = ["label", "text"])
    test_data = pd.DataFrame(columns = ["label", "text"])
        
    for label in data["label"].unique():
        texts = data[data["label"] == label]["text"].tolist()
        np.random.shuffle(texts)
        
        n_test = int(len(texts) * test_size)
        test_samples = texts[:n_test]
        train_samples = texts[n_test:]
        
        for text in test_samples:
            test_data = test_data._append({"label": label, "text": text}, ignore_index=True)
            
        for text in train_samples:
            train_data = train_data._append({"label": label, "text": text}, ignore_index=True)

    return train_data, test_data

def save_matrix_as_png(results: dict[str, dict[str, int]], filename: str = "results.png"):
    """
    TODO
    """
    
    # Get all unique labels
    labels = list(results.keys())

    # Create an empty DataFrame with labels as both index and columns
    df = pd.DataFrame(index=labels, columns=labels)

    # Populate the DataFrame with values from the results dictionary
    for label in labels:
        for inner_label in labels:
            df.at[label, inner_label] = results[label].get(inner_label, 0)

    fig = df2img.plot_dataframe(df, fig_size = (9000, len(labels) * 23))
    df2img.save_dataframe(fig = fig, filename = filename)