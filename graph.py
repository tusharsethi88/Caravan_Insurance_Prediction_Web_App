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

# For treating imbalanced data
from imblearn.over_sampling import SMOTE  

# For Visualizing the data
import matplotlib.pyplot as plt
import io
import base64


def build_graph_scatter(x_coordinates, y_coordinates,i,filename):

	# EDA function
	df = pd.read_csv(os.path.join('static/uploadsDB/Visualization_Data',filename))
	df_size = df.size
	df_shape = df.shape
	df_columns = list(df.columns)
	df_targetname = df[df.columns[-1]].name
	df_featurenames = df_columns[0:-1] # select all columns till last column
	df_selectedfeaturenames = df.columns.values[[1,3,6,9,10,11,13,18,19,20,21,22,23,24,25,26,28,29,30,31,32,35,36,37,39,40,42,43,65,67,68,72,73,75,76,80,83,85]]
	df_featurenames = (df[df.columns[[1,3,6,9,10,11,13,18,19,20,21,22,23,24,25,26,28,29,30,31,32,35,36,37,39,40,42,43,65,67,68,72,73,75,76,80,83,85]]].values)
	df_Xfeatures = df.iloc[:,0:-1] 
	df_Ylabels = df[df.columns[-1]] # Select the last column as target
	# same as above df_Ylabels = df.iloc[:,-1]

	# Normalization - Using MinMax Scaler
	min_max_scaler = preprocessing.MinMaxScaler()
	df_scaled_featurenames = min_max_scaler.fit_transform(df_featurenames)
	y = np.vstack(df['CARAVAN'].values)
	X = df_scaled_featurenames
	Y = df_Ylabels
	
	
	img = io.BytesIO()
	fig, ax = plt.subplots()
	plt.title(df_selectedfeaturenames[i], size=9,color='darkslateblue',fontweight='bold')
	plt.scatter(x_coordinates, y_coordinates, s=40, marker= 'o',c=((y[:,0:1])+20).reshape(-1), alpha=0.5)
	plt.savefig(img, format='png')
	img.seek(0)
	graph_url = base64.b64encode(img.getbuffer()).decode()
	plt.close()
	return 'data:image/png;base64,{}'.format(graph_url)


def build_graph_histogram(x_coordinates,j,filename):

	# EDA function
	df = pd.read_csv(os.path.join('static/uploadsDB/Visualization_Data',filename))
	df_size = df.size
	df_shape = df.shape
	df_columns = list(df.columns)
	df_targetname = df[df.columns[-1]].name
	df_featurenames = df_columns[0:-1] # select all columns till last column
	df_selectedfeaturenames = df.columns.values[[1,3,6,9,10,11,13,18,19,20,21,22,23,24,25,26,28,29,30,31,32,35,36,37,39,40,42,43,65,67,68,72,73,75,76,80,83,85]]
	df_featurenames = (df[df.columns[[1,3,6,9,10,11,13,18,19,20,21,22,23,24,25,26,28,29,30,31,32,35,36,37,39,40,42,43,65,67,68,72,73,75,76,80,83,85]]].values)
	df_Xfeatures = df.iloc[:,0:-1] 
	df_Ylabels = df[df.columns[-1]] # Select the last column as target
	# same as above df_Ylabels = df.iloc[:,-1]

	# Normalization - Using MinMax Scaler
	min_max_scaler = preprocessing.MinMaxScaler()
	df_scaled_featurenames = min_max_scaler.fit_transform(df_featurenames)
	y = np.vstack(df['CARAVAN'].values)
	X = df_scaled_featurenames
	Y = df_Ylabels
	
	img = io.BytesIO()
	fig, ax = plt.subplots()
	plt.title(df_selectedfeaturenames[j], size=9,color='darkslateblue',fontweight='bold')
	plt.hist(x_coordinates,alpha=0.7)
	plt.savefig(img, format='png')
	img.seek(0)
	graph_url = base64.b64encode(img.getbuffer()).decode()
	plt.close()
	return 'data:image/png;base64,{}'.format(graph_url)



