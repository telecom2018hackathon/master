## Récupération de la liste de fichiers en fonction de l'extension

path = "/home/jacky/test"
import glob
import re
import os

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
    

dico = training_files_to_list(path)
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
    
    
