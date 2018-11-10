import pandas as pd
import numpy as np

# ouverture du fichier true_label
path = 'true_labels_training.txt'

# Converts training label file to dataframe for easy retrieval
# Returns: dataframe 1col int (0 or 1)
def label_to_data_series():
    f = open(path)
    labels = f.read()
    return pd.Series(tuple(labels))
