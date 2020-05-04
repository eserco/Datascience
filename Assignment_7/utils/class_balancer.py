from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from utils.preprocessor import preprocess

def doUpsamling(X,y):
	sm = SMOTE(random_state=42)
	X_train_res, y_train_res = sm.fit_sample(X, y)
	
	print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
	print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

	print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
	print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
	
	return X_train_res, y_train_res