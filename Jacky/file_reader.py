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
    

list_path = training_files_to_list(path)
    
def to_pandas():
    list_path = training_files_to_list(path)
    n = len(list_path)//20
    df_l = []
    for i in range(20):
        u = list_path[i*n:(i+1)*n]
        g = [[elmt[0],file_read(elmt[1]).split("\n"),file_read(elmt[2]).split("\n")] for elmt in u]
        print(round(i/21*100))
        df = pd.DataFrame(g,columns = ["index","behavior","process"])
        df = df.set_index("index")
        df_l.append(df)
    u = list_path[(20)*n:]
    g = [[elmt[0],file_read(elmt[1]).split("\n"),file_read(elmt[2]).split("\n")] for elmt in u]
    print(100)
    df = pd.DataFrame(g,columns = ["index","behavior","process"])
    df = df.set_index("index")
    df_l.append(df)
    return pd.concat(df_l)
    
 
def file_read(file_path):
    """return the string of the full content of a text"""
    with open(file_path) as file:
        data = file.read()
    return data
        
    
