import glob
import re
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


path = "~/hackathon/training_dataset/"

# ------------------------------------------------
# -                   JACKY                      -
# ------------------------------------------------

## Récupération de la liste de fichiers en fonction de l'extension


def training_files_to_list(p):
    """Create a dict with paths and key in key. 
    
    Variables :
        Path : location of the files
    
    Return :
        L = [(XXXXXX,process_path,behavior_path)]   
        """
    os.chdir(p)
    
    filenames_proc = glob.glob('*_process_generation.txt')

    re_index = r"(\d+)"
    
    return {re.findall(re_index,elmt)[0]:
                         [path + "/" + elmt,
                         path + "/" + re.sub('process_generation','behavior_sequence', elmt)] 
                         for elmt in filenames_proc}
    

def path_from_index(index,file):
    
    """return the path given index number and file type (process or behavior)
    
    index : str
    file : string : behavior or process"""
    
    dico = training_files_to_list(path)
    
    if file == "behavior":
        return dico[str(index)][1]
    
    if file == "process":
        return dico[str(index)][0]
    
    else:
        return None


# ------------------------------------------------
# -                   ARNAUD                     -
# ------------------------------------------------


file_name = '/training_108006_behavior_sequence.txt'

def behavior_file_to_pandas(index):
	filepath = path_from_index(index, 'behavior')
	data = pd.read_csv(filepath, header = None)
	data.columns = columns = ['PROCESS', 'RPI', 'API']
	return data

def behavior_file_to_metrics(index):
	data = behavior_file_to_pandas(index)
	return data.describe().loc[['count', 'unique', 'freq'],:]
