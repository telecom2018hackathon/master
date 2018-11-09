import pandas as pd
import numpy as np

# ouverture du fichier true_label
path = '*/true_labels_training.txt'
path2 = 'true_labels_training.txt'
f = open(path2)
labels = f.read()
ds_label = pd.Series(tuple(labels))
