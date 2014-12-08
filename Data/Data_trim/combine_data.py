'''
    AUTHOR Spencer Penn
'''

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import fetch_20newsgroups
from PIL import Image


class AD_Data_Functions:

	def __init__(self, max_Iters = 10000):
		'''
		Constructor
		'''
		self.max_Iters = max_Iters

	def combine_data(self,X1, X2, Y):

		X1 = np.matrix(X1)
		X2 = np.matrix(X2)
		Y = np.matrix(Y)


		N_X1, D_X1 = X1.shape
		N_X2, D_X2 = X2.shape
		N_Y, D_Y = Y.shape

		X_out = np.zeros((N_X1,(D_X1+D_X2 - 2)))
		Y_out = np.zeros((N_Y,D_Y))


		count = 0
		a = 0
		b = 0

		#if count < self.max_count:
		for i in range(0,N_Y):
			RID_Y = Y[i,0]

			for j in range(a,N_X1):
				RID_X1 = X1[j,0]

				for k in range(b,N_X2):
					RID_X2 = X2[k,0]

					if RID_X1==RID_Y and RID_X2 == RID_Y:

						Y_out[count,:] = Y[i,:]
						X_out[count,:D_X1] = X1[j,:]
					 	X_out[count,D_X1:] = X2[k,2:]
					 	a = j
					 	b = k
					 	count += 1
					 	print 'count ', count
					 	break
			 	if RID_X1==RID_Y and RID_X2 == RID_Y:
					break

		 		if count > self.max_Iters:
	 				break
	 		if count > self.max_Iters:
 				break




	 	X_out = X_out[0:count,:]
	 	Y_out = Y_out[0:count,:]


		return (X_out, Y_out)