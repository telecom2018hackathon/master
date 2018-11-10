## Récupération de la liste de fichiers en fonction de l'extension

path = "/home/jacky/test"
import glob
import re
import os
import pandas as pd
import time 

def training_files_to_list(path):
    """Create a dict with paths and key in key. 
    
    Variables :
        Path : location of the files
    
    Return :
        L = [(XXXXXX,process_path,behavior_path)]   
        """
    re_index = r"(\d+)"
        
    filenames_proc = glob.glob(f'{path}/*_process_generation.txt')
    f = [[re.findall(re_index,elmt)[0],
         re.sub('process_generation','behavior_sequence', elmt),
         elmt ]
         for elmt in filenames_proc]
    
    return f
    

dico = training_files_to_list(path)
    
def to_pandas(list_path):
    t= time.time()
    """return a dataset of the files in the dico"""
    l = [[elmt[0],file_read(elmt[1]).split("\n"),file_read(elmt[2]).split("\n")] for elmt in list_path]
    
    print(t-time.time())
    df = pd.DataFrame(l,columns = ["index","behavior","process"])
    df = df.set_index("index")
    return df
    
 
def file_read(file_path):
    """return the string of the full content of a text"""
    with open(file_path) as file:
        data = file.read()
    return data
        
    
