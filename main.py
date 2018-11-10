# -*- coding: utf-8 -*-

import glob
import re
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------
# -                   TOM                        -
# ------------------------------------------------


# ouverture du fichier true_label
path_true_labels_training = '/home/cloud/hackathon/true_labels_training.txt'

# Converts training label file to dataframe for easy retrieval
# Returns: dataframe 1col int (0 or 1)
def label_to_data_series():
    f = open(path_true_labels_training)
    labels = f.read()
    return pd.Series(tuple(labels))



path_training_dataset = '/home/cloud/hackathon/training_dataset'

# ------------------------------------------------
# -                   JACKY                      -
# ------------------------------------------------

## Récupération de la liste de fichiers en fonction de l'extension


def training_files_to_list(path):
    """Create a dict with paths and key in key. 
    
    Variables :
        Path : location of the files
    
    Return :
        L = [(XXXXXX,process_path,behavior_path)]   
        """

    filenames_proc = glob.glob(f'{path}/*_process_generation.txt')

    re_index = r"(\d+)"
    
    return {re.findall(re_index,elmt)[0]:
                         [elmt,
                         re.sub('process_generation','behavior_sequence', elmt)] 
                         for elmt in filenames_proc}
    
dico = training_files_to_list(path_training_dataset)

def path_from_index(index,file):
    
    """return the path given index number and file type (process or behavior)
    
    index : str
    file : string : behavior or process
    dico : dictionnaire avec indexes """
    
    
    if file == "behavior":
        return dico[str(index)][1]
    
    if file == "process":
        return dico[str(index)][0]
    
    else:
        return None


# ------------------------------------------------
# -                   ARNAUD                     -
# ------------------------------------------------

def behavior_file_to_pandas(index):
	filepath = path_from_index(index, 'behavior')
	data = pd.read_csv(filepath, header = None)
	data.columns = columns = ['PROCESS', 'RPI', 'API']
	return data

def behavior_file_to_metrics(index):
	data = behavior_file_to_pandas(index)
	return data.describe().loc[['count', 'unique', 'freq'],:]
