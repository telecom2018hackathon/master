import pandas as pd
import numpy as np


index = '020820'

def get_best_api_from_behavior(index):
    """
This fonction get the index as argument and return a dataframe with the id of the API in the first column and the number of occurence in the second column sorted by the number descending.

    Variable : Index

    Return : DataFrame with the API number as the first column sorted by the number of occurence of the API in descending order

    """

    path = f'/Users/lyadrien/Documents/Telecom/hackathon_Huawei/training_set/training_{index}_behavior_sequence.txt'

    behavior = pd.read_csv(path, names=('PROCESS','RPI','API'))
    behavior['COUNT'] = 1
    behavior.describe()
    behavior_grpd = behavior.groupby('API').sum()
    behavior_sorted = behavior_grpd.sort_values('COUNT', ascending=False)
    behavior_sorted =  behavior_sorted.reset_index(level = 'API')
    behavior_sorted['API'] = behavior_sorted['API'].replace({r'api_':''}, regex = True)
    return behavior_sorted 
 
