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


# load the data set
filename = 'AD_Challenge_Training_Data_Clinical_Updated_7/ADNI_Training_Q1_APOE_July22.2014_SPENCER_Edit.csv'


X = np.genfromtxt(filename, dtype=None, delimiter=',', names=True, usecols = range(0,12)) #all the data except MMSE_24
y = np.genfromtxt(filename, dtype=None, delimiter=',', names=True, usecols=(12)) #output is the MMSE after 24 months

n = len(X)
nTrain = 0.5*n  #training on 50% of the data

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# split the data
Xtrain = X[:nTrain]
ytrain = y[:nTrain]
Xtest = X[nTrain:]
ytest = y[nTrain:]

#build Model SVM
modelSVM = svm.SVC(C = 3, kernel = linear_kernel)
start2 = time.time()
modelSVM.fit(Xtrain, ytrain)
end2 = time.time()

#Make Predictions
TrainPredSVM = modelSVM.predict(Xtrain)
TestPredSVM = modelSVM.predict(Xtest)

#metrics
TrainAccSVM = accuracy_score(ytrain, TrainPredSVM)
TestAccSVM = accuracy_score(ytest, TestPredSVM)

TrainPrecSVM = precision_score(ytrain, TrainPredSVM)
TestPrecSVM = precision_score(ytest, TestPredSVM)

TrainRecallSVM = recall_score(ytrain, TrainPredSVM)
TestRecallSVM = recall_score(ytest, TestPredSVM)

TrainTimeSVM = end2 - start2

print '...SVM Metrics...'
print 'training acc: ', TrainAccSVM
print 'testing acc: ', TestAccSVM
print 'train precision: ',TrainPrecSVM
print 'test precision : ',TestPrecSVM
print 'train recall: ', TrainRecallSVM
print 'test recall: ', TestRecallSVM
print 'train time: ', TrainTimeSVM, ' seconds'

