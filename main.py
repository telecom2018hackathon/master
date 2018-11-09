import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(sys.maxsize)

listofstr = []
"""
with open('../true_labels_training.txt') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		listofstr.append(row)

print(listofstr)
"""

df = pd.DataFrame(np.ones(50))
df.plot()
plt.show()