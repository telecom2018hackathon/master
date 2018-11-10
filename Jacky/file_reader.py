## Récupération de la liste de fichiers en fonction de l'extension

path = "~/hackathon/training_dataset"
import glob
import re
import os


def training_files_to_list(path):
    """Create a list of tuples representing the files ordered by index. 
    
    Variables :
        Path : location of the files
    
    Return :
        L = [(XXXXXX,process_path,behavior_path)]   
        """
    os.chdir(path)
    
    filenames_proc = glob.glob('*_process_generation.txt')
    print(filenames_proc)
    re_index = r"(\d+)"
    return [(re.findall(re_index,elmt)[0],
                         path + "/" + elmt,
                         path + "/" + re.sub('process_generation','behavior_sequence', elmt)) 
            for elmt in filenames_proc]
    


