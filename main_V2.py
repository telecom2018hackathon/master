# -*- coding: utf-8 -*-

import glob
import re
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time 
from multiprocessing import Process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

pd.options.display.float_format = '${:,.4f}'.format

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
    return pd.DataFrame(np.array([tuple(labels)]).T, columns = ['LABELS'])


path_training_dataset = '/home/cloud/hackathon/training_dataset/training_'
#path_training_dataset = '/home/cloud/hackathon/validation_dataset/validation_'

# ------------------------------------------------
# -                   JACKY                      -
# ------------------------------------------------

## Récupération de la liste de fichiers en fonction de l'extension


def index_to_path(index, typ):
    if typ == 'behavior' :
        return path_training_dataset + index + '_' + typ + '_sequence.txt'
    elif typ == 'process' :
        return path_training_dataset + index + '_' + typ + '_generation.txt'


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

path = '/home/cloud/hackathon/training_dataset'

def file_read(file_path):
    """return the string of the full content of a text"""
    with open(file_path) as file:
        data = file.read()
    return data

def api(i, list_path, n):
    u = list_path[i*n :(i+1)*n]
    j = 0
    g = []
    for elmt in u :
        if j%1000 == 0 :
            print(str(i) + " : " + str(j))
        g.append([elmt[0],file_read(elmt[1]).split("\n"),file_read(elmt[2]).split("\n")])
        j+=1
    df = pd.DataFrame(g,columns = ["index","behavior","process"])
    df = df.set_index("index")
    df.to_csv('./Api_' + str(i) + '.csv')

def to_pandas():
    list_path = training_files_to_list(path)
    n = len(list_path)//20
    for i in range(20):
        p = Process(target = api, args = (i, list_path, n))
        p.start()

 

# ------------------------------------------------
# -                   ARNAUD                     -
# ------------------------------------------------

def behavior_file_to_pandas(index):
    filepath = index_to_path(index, 'behavior')
    data = pd.read_csv(filepath, header = None)
    data.columns = columns = ['PROCESS', 'RPI', 'API']
    return data

def behavior_file_to_metrics(index):
    data = behavior_file_to_pandas(index)
    return data.describe().loc[['count', 'unique', 'freq'],:]

def behavior_metrics(index):
    data = behavior_file_to_metrics(index)
    return [data.loc['count','PROCESS'], data.loc['unique', 'RPI'], data.loc['unique', 'API']]


# ------------------------------------------------
# -                   MAT                        -
# ------------------------------------------------
# List of metrics regarding the processes
def process_metrics(index):
    filepath = index_to_path(index, 'process')
    df = pd.read_csv(filepath, header=None, sep=' -> ', engine='python')
    return [len(df), len(np.unique(df.values))]

# -------------- API METRICS --------------------------


l = list(range(3561))
api_index = pd.DataFrame(l)
# List and count of API regarding the behavior
def get_best_api_from_behavior(index):
    """
This fonction get the index as argument and return a dataframe with the id of the API in the first column and the number of occurence in the second column sorted by the number descending.
    Variable : Index
    Return : DataFrame with the API number as the first column sorted by the number of occurence of the API in descending order
    """
    filepath = index_to_path(index, 'behavior')
    behavior = pd.read_csv(filepath, names=('PROCESS','RPI','API'), engine='python')
    behavior['API'] =pd.to_numeric(behavior['API'].replace({'api_':''}, regex = True))
    behavior[index] = 1
    behavior.describe()
    behavior_grpd = behavior.groupby('API').sum()
   # behavior_sorted = behavior_grpd.sort_values(index, ascending=False)
   # behavior_sorted =  behavior_sorted.reset_index(level = 'API')
   # behavior_grpd['API'] = behavior_grpd['API'].replace({r'api_':''}, regex = True)
    behavior_grpd = behavior_grpd.reindex(api_index[0])
    return behavior_grpd.iloc[:,0].values.tolist()


# ------------- TRAINING SET ---------------------------------

# X creation
def thread_run(start, stop, threadNumber):
    l = []
    errorCount = 0
    for i in range(start, stop) :
        if i%1000 == 0:
            print(str(threadNumber) + " : " + str(i))
        index = str(i).zfill(6)
        try :
            l.append([index] + behavior_metrics(index) + process_metrics(index) + get_best_api_from_behavior(index))
        except :
            errorCount += 1
    pd.DataFrame(l).to_csv('./X_'+ str(threadNumber) + '.csv')


def launch_processes() :
    for i in range(40):
        p = Process(target = thread_run, args = (i*5000, (i+1)*5000, i))
        p.start()


def concat_csv():
    df = pd.DataFrame()
    for i in range(40) :
        print(i)
        temp = pd.read_csv('./X_' + str(i) + '.csv', header = 0, index_col = 0)
        temp.reindex(temp.index.values + i*5000)
        df = pd.concat([df,temp], ignore_index = True)
    api_cols = list(df)[6:]
    df.columns = ['Index', 'Behavior_lenght', 'Behavior_dist_RPI', 'Behavior_dist_API', 'Process_length', 'Process_dist'] + api_cols
    df.set_index('Index', inplace=True)
    df.to_csv('./XY_temp.csv')

def labelisation():
    labels = label_to_data_series()
    data = pd.read_csv('./XY_temp.csv', index_col = 0, header = 0).fillna(0)
    data = pd.merge(data, labels, left_index=True, right_index=True)
    return data


# ------------- VALIDATION SET ---------------------------------

def OLS():
    data = pd.read_csv('./XY.csv', header = 0, index_col = 0)
    X = data[['Behavior_lenght', 'Behavior_dist_RPI', 'Behavior_dist_API', 'Process_length', 'Process_dist']]
    Y = data['LABELS']
    reg = LogisticRegression(fit_intercept=True)
    reg.fit(X, Y)
    R2 = reg.score(X,Y)
    coef = reg.coef_

    Val = pd.read_csv('./Val.csv', header = 0, index_col = 0)

    (Val @ coef.T).plot()
    plt.show()

def decisionTreeClass():
    data = pd.read_csv('./XY.csv', header = 0, index_col = 0)
    X = data[['Behavior_lenght', 'Behavior_dist_RPI', 'Behavior_dist_API', 'Process_length', 'Process_dist']]
    Y = data['LABELS']
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, Y)

    Val = pd.read_csv('./Val.csv', header = 0, index_col = 0)
    Val = Val.reindex(range(40000), fill_value = 0)
    
    pd.DataFrame(clf.predict_proba(Val))[1].to_csv('answer.txt', index = False, float_format='%.4f')


def RandomForestClass():
    data = pd.read_csv('./XY.csv', header = 0, index_col = 0)
    X = data[['Behavior_lenght', 'Behavior_dist_RPI', 'Behavior_dist_API', 'Process_length', 'Process_dist']]
    Y = data['LABELS']
    rf = RandomForestClassifier()
    rf.fit(X, Y)

    Val = pd.read_csv('./Val.csv', header = 0, index_col = 0)
    Val = Val.reindex(range(40000), fill_value = 0)
    
    pd.DataFrame(rf.predict_proba(Val))[1].to_csv('answer.txt', index = False, float_format='%.4f')


def GradientBoostingClass():
    data = pd.read_csv('./XY.csv', header = 0, index_col = 0)
    X = data[['Behavior_lenght', 'Behavior_dist_RPI', 'Behavior_dist_API', 'Process_length', 'Process_dist']]
    Y = data['LABELS']
    rf = GradientBoostingClassifier()
    rf.fit(X, Y)

    Val = pd.read_csv('./Val.csv', header = 0, index_col = 0)
    Val = Val.reindex(range(40000), fill_value = 0)
    
    pd.DataFrame(rf.predict_proba(Val))[1].to_csv('answer.txt', index = False, float_format='%.4f')


def RandomForestClass2():
    data = labelisation();
    print('labelisation DONE')
    Y = data['LABELS']
    X = data.drop(columns = 'LABELS')
    print(X.shape)
    rf = RandomForestClassifier()
    rf.fit(X, Y)
    print('DONE')
    print(rf.score)
    Val = pd.read_csv('./Val.csv', header = 0, index_col = 0).fillna(0)
    Val = Val.reindex(range(40000), fill_value = 0)
    
    pd.DataFrame(rf.predict_proba(Val))[1].to_csv('answer.txt', index = False, float_format='%.4f')

RandomForestClass2()