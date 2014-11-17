import numpy as np

# load the data set
filename = 'data/ADNI_Training_Q1_APOE_July22.2014_SPENCER_Edit.csv'

data = np.genfromtxt(filename, dtype=float, delimiter=',', names=True)
X = np.loadtxt(filename, delimiter=',')
y = np.loadtxt(filename, delimiter=',')

n,d = X.shape
nTrain = 0.5*n  #training on 50% of the data

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]