import numpy as np


def loadQ1data():
    # load the data set
    filename = 'Data/AD_Challenge_Training_Data_Clinical_Updated_7/ADNI_Training_Q1_APOE_July22.2014_SPENCER_Edit.csv'


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

    return Xtrain, ytrain, Xtest, ytest