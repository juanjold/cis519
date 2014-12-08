
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
import csv
import pandas
import StringIO
from numpy import genfromtxt
from collections import Counter
from itertools import groupby
from pandas import DataFrame
from combine_data import AD_Data_Functions
import time
from sklearn import tree
from sklearn.metrics import accuracy_score

'''
____________________________________________________________
Load ADAS Score Values
____________________________________________________________
'''

filename = 'ADASSCORES_trim.csv'

# Load Data Matrix
ADAS_data = np.matrix(genfromtxt(filename, delimiter=','))[1:,:]
ADAS_dict = np.matrix(genfromtxt(filename, delimiter=',',dtype = 'str'))[0,:]	

N_ADAS, D_ADAS = ADAS_data.shape

'''
____________________________________________________________
Create Change in ADAS
____________________________________________________________
'''
'''
Column Structure
1. RID
2. VISCODE (highest)
3. Baseline TOTAL11
4. Final_Visit TOTAL11
5. Change in TOTAL11
'''

# Create Y
RID_ADAS = np.unique(np.asarray(ADAS_data[:,0]))
ADAS_patients = len(RID_ADAS)

ADAS_change = np.zeros((ADAS_patients,5))
ADAS_change[:,0] = np.asarray(RID_ADAS)

for i in range(0, ADAS_patients):
	RID = ADAS_change[i,0]
	VISCODE = ADAS_change[i,1]
	for j in range(0,N_ADAS):
		RID_curr = ADAS_data[j,0]
		Date = ADAS_data[j,1]
		if RID_curr == RID:
			if Date == 0:
				ADAS_change[i,2] = ADAS_data[j,15]
			if Date >= VISCODE:
				ADAS_change[i,1] = Date
				ADAS_change[i,3] = ADAS_data[j,15]
ADAS_change[:,4] = ADAS_change[:,3] - ADAS_change[:,2]



'''
____________________________________________________________
Load Clinical Data Values
____________________________________________________________
'''
filename = 'ADNI_Training_Q1_APOE_July22.2014_SPENCER_Edit.csv'

# Load Data Matrix
Clinical_data = np.matrix(genfromtxt(filename, delimiter=','))[1:,:]		

# Get Dictionary as List
Clinical_dict = np.matrix(genfromtxt(filename, delimiter=',',dtype = 'str'))[0,:]


'''
____________________________________________________________
Load Diagnosis Values
____________________________________________________________
'''
filename = 'DXSUM_PDXCONV_ADNI1.csv'

# Load Data Matrix
Diagnosis_data = np.matrix(genfromtxt(filename, delimiter=','))[1:,:]
Diagnosis_dict = np.matrix(genfromtxt(filename, delimiter=',',dtype='str'))[0,:]

#print 'Diagnosis_data ', Diagnosis_data.shape

'''
____________________________________________________________
Load MFI Values
____________________________________________________________
'''
filename = 'UCSFFSL_trim.csv'

# Load Data Matrix
MRI_data = np.matrix(genfromtxt(filename, delimiter=','))[1:,:]			
MRI_dict = np.matrix(genfromtxt(filename, delimiter=',',dtype='str'))[0,:]

#print 'MRI_data ', MRI_data.shape


'''
____________________________________________________________
Load Biomarker Values
____________________________________________________________
'''
filename = 'UPENNBIOMKadni1_trim.csv'

# Load Data Matrix
Biomarker_data = np.matrix(genfromtxt(filename, delimiter=','))[1:,:]
Biomarker_dict = np.matrix(genfromtxt(filename, delimiter=',',dtype='str'))[0,:]

#print 'Biomarker_data ', Biomarker_data.shape

'''
____________________________________________________________
Eliminate All Non-Baseline Data from Set Data Specs
____________________________________________________________
'''
ADAS_data = np.asarray(ADAS_data)
Clinical_data = np.asarray(Clinical_data)
Diagnosis_data = np.asarray(Diagnosis_data)
MRI_data = np.asarray(MRI_data)
Biomarker_data = np.asarray(Biomarker_data)

ADAS_data = ADAS_data[ADAS_data[:,1]==0,:]
Clinical_data = Clinical_data[Clinical_data[:,1]==0,:]
Diagnosis_data = Diagnosis_data[Diagnosis_data[:,1]==0,:]
MRI_data = MRI_data[MRI_data[:,1]==0,:]
Biomarker_data = Biomarker_data[Biomarker_data[:,1]==0,:]

'''
____________________________________________________________
PRINT Data Specs
____________________________________________________________
'''

# Compute Data Shape
N_ADAS, D_ADAS = ADAS_data.shape
N_Clinical, D_Clinical = Clinical_data.shape
N_Diagnosis, D_Diagnosis = Diagnosis_data.shape
N_MRI, D_MRI = MRI_data.shape
N_Bio, D_Bio = Biomarker_data.shape

# Size
print 'ADAS Data size ', ADAS_data.shape
print 'Clinical Data size ', Clinical_data.shape
print 'Diagnosis Data size ', Diagnosis_data.shape
print 'MRI Data size ', MRI_data.shape
print 'Biomarker Data size ', Biomarker_data.shape

#Print Dictionaries

print 'ADAS' , ADAS_dict
print 'Clinical', Clinical_dict
print 'Diagnosis', Diagnosis_dict
#print 'MRI', MRI_dict
print 'Bio Marker', Biomarker_dict

# Unique Patients
RID_ADAS = np.unique(np.asarray(ADAS_data[:,0]))
ADAS_patients = len(RID_ADAS)

RID_clinical = np.unique(np.asarray(Clinical_data[:,0]))
Clinical_patients = len(RID_clinical)

RID_diag = np.unique(np.asarray(Diagnosis_data[:,0]))
Diag_patients = len(RID_diag)

RID_MRI = np.unique(np.asarray(MRI_data[:,0]))
MRI_patients = len(RID_MRI)

RID_Bio = np.unique(np.asarray(Biomarker_data[:,0]))
Bio_patients = len(RID_Bio)

print 'ADAS Unique Patients ', ADAS_patients
print 'Clinical Unique Patients ', Clinical_patients
print 'Diagnosis Unique Patients ', Diag_patients
print 'MRI Unique Patients ', MRI_patients
print 'Biomarker Unique Patients ', Bio_patients




'''
____________________________________________________________
Create Merged Datasets
____________________________________________________________
'''

#Eliminate NANs
ADAS_data = np.asarray(ADAS_data)
ADAS_data = ADAS_data[~np.isnan(ADAS_data).any(axis = 1),:]

Biomarker_data = np.asarray(Biomarker_data)
Biomarker_data = Biomarker_data[~np.isnan(Biomarker_data).any(axis = 1),:]

MRI_data = np.asarray(MRI_data)
MRI_data = MRI_data[~np.isnan(MRI_data).any(axis = 1),:]

Diagnosis_data = np.asarray(Diagnosis_data)
Diagnosis_data = Diagnosis_data[~np.isnan(Diagnosis_data).any(axis = 1),:]

Clinical_data = np.asarray(Clinical_data)
Clinical_data = Clinical_data[~np.isnan(Clinical_data).any(axis = 1),:]



start = time.time()
func = AD_Data_Functions(max_Iters = 20000)
X1 = ADAS_data
X2 = Biomarker_data
Y = ADAS_change

(X_out, Y_out) = func.combine_data(X1, X2, Y)
end = time.time()

print 'X out ', X_out
print 'Y out ', Y_out
print 'Runtime', end - start

X1 = X_out
X2 = MRI_data
Y = Y_out

(X_out, Y_out) = func.combine_data(X1, X2, Y)
end = time.time()

X1 = X_out
X2 = Diagnosis_data
Y = Y_out

(X_out, Y_out) = func.combine_data(X1, X2, Y)
end = time.time()

X1 = X_out
X2 = Clinical_data[:,:9]
Y = Y_out

(X_out, Y_out) = func.combine_data(X1, X2, Y)
end = time.time()


'''
____________________________________________________________
Run Machine Learning Algorithms
____________________________________________________________
'''
X = X_out[:,2:]
Y = Y_out[:,4]

# Set Threshhold
Y = np.asarray(Y>=4).astype(int)

print 'DATA SPLIT', np.mean(Y)

n,d = X.shape
nTrain = 0.5*n  #training on 80% of the data

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = Y[idx]


# split the data
Xtrain = X[:nTrain]
ytrain = y[:nTrain]
Xtest = X[nTrain:]
ytest = y[nTrain:]

Acc_mat = np.zeros((20,3))
for i in range(1,20)
	
	clf = tree.DecisionTreeClassifier(max_depth=i)
	clf = clf.fit(Xtrain, ytrain)


	pred_train = clf.predict(Xtrain)
	pred_test = clf.predict(Xtest)

	accuracy_DT_train = accuracy_score(ytrain, pred_train)
	accuracy_DT_test = accuracy_score(ytest, pred_test)

	Acc_mat[i,0] = i
	Acc_mat[i,0] = i
	Acc_mat[i,0] = i

	print 'Accuracy Train ', accuracy_DT_train
	print 'Accuracy Test ', accuracy_DT_test

#tree.export_graphviz(clf,out_file='tree.dot')

# Count unique values



