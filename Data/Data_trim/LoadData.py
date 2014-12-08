
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
Load Clinical Data Values
____________________________________________________________
'''
filename = 'ADNI_Training_Q1_APOE_July22.2014_SPENCER_Edit.csv'

# Load Data Matrix
Clinical_data = np.matrix(genfromtxt(filename, delimiter=','))[1:,:]		

# Get Dictionary as List
Clinical_dict = np.matrix(genfromtxt(filename, delimiter=',',dtype = 'str'))[0,:]

N_Clinical, D_Clinical = Clinical_data.shape

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

N_Diagnosis, D_Diagnosis = Diagnosis_data.shape


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

N_MRI, D_MRI = MRI_data.shape


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
N_Bio, D_Bio = Biomarker_data.shape


'''
____________________________________________________________
PRINT Data Specs
____________________________________________________________
'''

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
Create 
____________________________________________________________
'''

# Create Y
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

	#Check if already in Future Matrix

print ADAS_change[0:20,:]



# Count unique values



