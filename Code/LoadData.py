
'''
Load Data Module

'''


import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import linear_kernel
from sklearn import linear_model
import pylab as pl
from tempfile import TemporaryFile



'''
Load ADA Score Values
'''

ADA = np.loadtxt('ADASSCORES_trim.csv', delimiter =',', skiprows=1)
# np.load(ADA_scores)

'''
DATA COLUMNS
1. RID
2. VISCODE
3. Q1
4. Q2
5 Q3
6 Q4
7 Q5
8 Q6
9 Q7
10 Q8
11 Q9
12 Q10
13 Q11
14 Q12
15 Q14
16 TOTAL11
17 TOTALMOD
'''

ADA_scores = TemporaryFile()

np.save(ADA_scores, ADA)

print ADA_scores

