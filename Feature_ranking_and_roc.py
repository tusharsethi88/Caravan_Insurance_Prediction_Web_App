# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 

from werkzeug import secure_filename
import os
import datetime
import time

# EDA Packages
import pandas as pd 
import numpy as np
from pandas.io.parsers import read_csv

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

# ML Packages For Preprocessing, Vectorization of Text For Feature Extraction
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc

# For treating imbalanced data
from imblearn.over_sampling import SMOTE  

# For Visualizing the data
import matplotlib.pyplot as plt
import io
import base64

def Feature_ranking_and_roc(i,filename):
	# EDA function
	df = pd.read_csv(os.path.join('static/uploadsDB/Visualization_Data',filename))
	X = (df[df.columns[[1,3,6,9,10,11,13,18,19,20,21,22,23,24,25,26,28,29,30,31,32,35,36,37,39,40,42,43,65,67,68,72,73,75,76,80,83,85]]].values)

	# Normalization - Using MinMax Scaler
	min_max_scaler = preprocessing.MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	y = np.vstack(df['CARAVAN'].values)
	
	
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




	clf_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features=None, 
                                max_leaf_nodes=None, min_impurity_split=1e-07)
	clf_DT.fit(x_train, y_train)
	y_pred_DT = clf_DT.predict(x_test)
	dtree_pred_prob = clf_DT.predict_proba(x_test)[:, 1]
	fpr1, tpr1, thresholds1 = roc_curve(y_test, dtree_pred_prob)
	roc_auc1 = auc(fpr1, tpr1)

	clf_NB = BernoulliNB()
	clf_NB.fit(x_train, y_train)
	y_pred_NB = clf_NB.predict(x_test)
	NB_pred_prob = clf_NB.predict_proba(x_test)[:, 1]
	fpr2, tpr2, thresholds2 = roc_curve(y_test, NB_pred_prob)
	roc_auc2 = auc(fpr2, tpr2)

	clf_Log = LogisticRegression(solver='liblinear', max_iter=1000, 
                             random_state=42,verbose=2,class_weight='balanced')


	clf_Log.fit(x_train, y_train)
	y_pred_Log = clf_Log.predict(x_test)
	log_pred_prob = clf_Log.predict_proba(x_test)[:, 1]
	fpr3, tpr3, thresholds3 = roc_curve(y_test, log_pred_prob)
	roc_auc3 = auc(fpr3, tpr3)


	clf_RF = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=15,
	                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
	                                max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
	                                bootstrap=True, oob_score=False, n_jobs=1, 
	                                random_state=42, verbose=1, warm_start=False, class_weight=None)
	clf_RF.fit(x_train, y_train)
	y_pred_RF = clf_RF.predict(x_test)
	rf_pred_prob = clf_RF.predict_proba(x_test)[:, 1]
	fpr4, tpr4, thresholds4 = roc_curve(y_test, rf_pred_prob)
	roc_auc4 = auc(fpr4, tpr4)


	clf_SVM = SVC(C=10, class_weight='balanced', gamma='auto', kernel='rbf',
	              max_iter=-1, probability=True, random_state=42, verbose=True)
	clf_SVM.fit(x_train, y_train)
	y_pred_SVM = clf_SVM.predict(x_test)
	svm_pred_prob = clf_SVM.predict_proba(x_test)[:, 1]
	fpr5, tpr5, thresholds5 = roc_curve(y_test, svm_pred_prob)
	roc_auc5 = auc(fpr5, tpr5)



	clf_MLP = MLPClassifier(activation='relu', alpha=1e-05,
	       batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
	       epsilon=1e-08, hidden_layer_sizes=(64), learning_rate='constant',
	       learning_rate_init=0.001, max_iter=2000, momentum=0.9,
	       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
	       tol=0.001, validation_fraction=0.1, verbose=True,
	       warm_start=False)

	clf_MLP.fit(x_train, y_train)
	y_pred_MLP = clf_MLP.predict(x_test)
	mlp_pred_prob = clf_MLP.predict_proba(x_test)[:, 1]
	fpr6, tpr6, thresholds6 = roc_curve(y_test, mlp_pred_prob)
	roc_auc6 = auc(fpr6, tpr6)

	img = io.BytesIO()
	fig, ax = plt.subplots()
	plt.plot(fpr6, tpr6,color='blue',label='ROC curve NN (AUC = %0.2f)' % roc_auc6)
	plt.plot(fpr5, tpr5,color='green',label='ROC curve SVM (AUC = %0.2f)' % roc_auc5)
	plt.plot(fpr4, tpr4,color='gold',label='ROC curve RF (AUC = %0.2f)' % roc_auc4)
	plt.plot(fpr3, tpr3,color='red',label='ROC curve LR (AUC = %0.2f)' % roc_auc3)
	plt.plot(fpr2, tpr2,color='black',label='ROC curve NB (AUC = %0.2f)' % roc_auc2)
	plt.plot(fpr1, tpr1,color='navy',label='ROC curve DT (AUC = %0.2f)' % roc_auc1)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.title('ROC curve ')
	plt.xlabel('(FPR)')
	plt.ylabel('(TPR)')
	plt.grid(True)
	plt.legend(loc="lower right")
	plt.savefig(img, format='png')
	img.seek(0)
	graph_url = base64.b64encode(img.getbuffer()).decode()
	plt.close()
	return 'data:image/png;base64,{}'.format(Feature_ranking_and_roc_url)