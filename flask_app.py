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
import pickle

# For treating imbalanced data
from imblearn.over_sampling import SMOTE  

# For Visualizing the data
import matplotlib.pyplot as plt
import io
import base64
from graph import build_graph_scatter
from graph import build_graph_histogram

import matplotlib
matplotlib.use('Agg')




app = Flask(__name__)
Bootstrap(app)
db = SQLAlchemy(app)

# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'

# Saving Data To Database Storage
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/tableau')
def tableau():
	return render_template('tableau.html')

# Route for Prediction Data Processing and Details Page
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB/Prediction_Data',filename))
		fullfile = os.path.join('static/uploadsDB/Prediction_Data',filename)

		# For Time
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

		# EDA function
		df = pd.read_csv(os.path.join('static/uploadsDB/Prediction_Data',filename))
		df_size = df.size
		df_shape = df.shape
		df_columns = list(df.columns)
		df_targetname = df[df.columns[-1]].name
		df_featurenames = df_columns[0:-1] # select all columns till last column
		df_featurenames = (df[df.columns[[1,3,6,9,10,11,13,18,19,20,21,22,23,24,25,26,28,29,30,31,32,35,36,37,39,40,42,43,65,67,68,72,73,75,76,80,83,85]]].values)
		df_Xfeatures = df.iloc[:,0:-1] 
		df_Ylabels = df[df.columns[-1]] # Select the last column as target
		# same as above df_Ylabels = df.iloc[:,-1]

		# Normalization - Using MinMax Scaler
		min_max_scaler = preprocessing.MinMaxScaler()
		df_scaled_featurenames = min_max_scaler.fit_transform(df_featurenames)

		y = np.vstack(df['CARAVAN'].values)

		# Model Building
		X = df_scaled_featurenames
		Y = df_Ylabels
		seed = 7
		# prepare models
		models = []
		models.append(('Logistic Regression', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42,verbose=2,class_weight='balanced')))
		models.append(('Random Forest', RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=15,
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
                                bootstrap=True, oob_score=False, n_jobs=1, 
                                random_state=42, verbose=1, warm_start=False, class_weight=None)))
		models.append(('Neural Network Classifier', MLPClassifier(activation='relu', alpha=1e-05,
       							batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
       							epsilon=1e-08, hidden_layer_sizes=(64), learning_rate='constant',
       							learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       							nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       							tol=0.001, validation_fraction=0.1, verbose=True,
       							warm_start=False)))
		models.append(('Decision Tree', DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07)))
		models.append(('Naive Bayes', BernoulliNB()))
		models.append(('SVM', SVC(C=10, class_weight='balanced', gamma='auto', kernel='rbf', max_iter=-1, probability=True, random_state=42, verbose=True)))
		# evaluate each model in turn
		
		results = []
		names = []
		allmodels = []
		scoring = 'accuracy'
		for name, model in models:
			kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=False, random_state=seed)
			cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			allmodels.append(msg)
			model_results = results
			model_names = names 

	
		finalmodel = np.mean(model_results, axis=1)


		if finalmodel[0] == max(finalmodel):
   			lr = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42,verbose=2,class_weight='balanced')
   			lr.fit(X,Y)
   			finalfile = open('models/finalized_model.sav', 'wb')
   			pickle.dump(lr, finalfile)
   			finalfile.close()
    
		elif finalmodel[1] == max(finalmodel):
   			rf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=15,
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
                                bootstrap=True, oob_score=False, n_jobs=1, 
                                random_state=42, verbose=1, warm_start=False, class_weight=None)
   			rf.fit(X,Y)
   			finalfile = open('models/finalized_model.sav', 'wb')
   			pickle.dump(rf, finalfile)
   			finalfile.close()
    
		elif finalmodel[2] == max(finalmodel):
   			nn = MLPClassifier(activation='relu', alpha=1e-05,batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(64), learning_rate='constant',learning_rate_init=0.001, max_iter=2000, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,tol=0.001, validation_fraction=0.1, verbose=True,warm_start=False)
   			nn.fit(X,Y)
   			finalfile = open('models/finalized_model.sav', 'wb')
   			pickle.dump(nn, finalfile)
   			finalfile.close()
    
		elif finalmodel[3] == max(finalmodel):
			dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07)
			dt.fit(X,Y)
			finalfile = open('models/finalized_model.sav', 'wb')
			pickle.dump(dt, finalfile)
			finalfile.close()

		elif finalmodel[4] == max(finalmodel):
   			nb = BernoulliNB()
   			nb.fit(X,Y)
   			finalfile = open('models/finalized_model.sav', 'wb')
   			pickle.dump(nb, finalfile)
   			finalfile.close()
     
		else:
			svc = SVC(C=10, class_weight='balanced', gamma='auto', kernel='rbf', max_iter=-1, probability=True, random_state=42, verbose=True)
			svc.fit(X,Y)
			finalfile = open('models/finalized_model.sav', 'wb')
			pickle.dump(svc, finalfile)
			finalfile.close()	

		np_array = np.array(finalmodel)
		item_index = np.where(np_array==max(finalmodel))
		itemindex1=item_index[0]
		itemindex2=itemindex1[0]
		highest_accuracy = model_names[itemindex2]
			
			
		# Saving Results of Uploaded Files  to Sqlite DB
		newfile = FileContents(name=file.filename,data=file.read(),modeldata=msg)
		db.session.add(newfile)
		db.session.commit()		
		
	return render_template('dataupload.html',filename=filename,date=date,
		df_size=df_size,
		df_shape=df_shape,
		df_columns =df_columns,
		df_targetname =df_targetname,
		model_results = allmodels,
		model_names = names,
		fullfile = fullfile,
		dfplot = df,
		highest_accuracy = highest_accuracy
		)


# Route for Prediction Data Processing and Details Page
@app.route('/dataupload2',methods=['GET','POST'])
def dataupload2():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB/Prediction_Data',filename))
		fullfile = os.path.join('static/uploadsDB/Prediction_Data',filename)

		# For Time
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

		model = pickle.load(open('models/finalized_model.sav','rb'))
		df = pd.read_csv(os.path.join('static/uploadsDB/Prediction_Data',filename))
		df_size = df.size
		df_shape = df.shape
		df_columns = list(df.columns)
		df_featurenames = (df[df.columns[[1,3,6,9,10,11,13,18,19,20,21,22,23,24,25,26,28,29,30,31,32,35,36,37,39,40,42,43,65,67,68,72,73,75,76,80,83,85]]].values)
		df_Xfeatures = df.iloc[:,0:-1] 

		# # Normalization - Using MinMax Scaler
		# min_max_scaler = preprocessing.MinMaxScaler()
		# df_scaled_featurenames = min_max_scaler.fit_transform(df_featurenames)

		# Model Building
		test_X = df_featurenames
		seed = 7

		predict_y= model.predict(test_X)
		pd.DataFrame(predict_y).to_csv("CaravanPredictionFile.csv")

	return render_template('dataupload2.html',filename=filename,date=date,
		df_size=df_size,
		df_shape=df_shape,
		df_columns =df_columns,
		fullfile = fullfile,
		dfplot=df)


@app.route('/graphs',methods=['GET','POST'])
def graphs():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB/Visualization_Data',filename))
		fullfile = os.path.join('static/uploadsDB/Visualization_Data',filename)

		# For Time
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

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
		
		# Scatter Plot
		c = [None]*38	
		var = 38
		for i in range(var):
			c[i] = build_graph_scatter(range(len(X)),X[:,i],i,filename)

		# Histogram
		d = [None]*38	
		var = 38
		for j in range(var):
			d[j] = build_graph_histogram(X[:,j],j,filename)

	return render_template('graphs.html', graph1 = c[0], graph2 = c[1], graph3 = c[2], graph4 = c[3], graph5 = c[4],
	graph6 = c[5],graph7 = c[6],graph8 = c[7],graph9 = c[8],graph10 = c[9],graph11 = c[10],graph12 = c[11],graph13 = c[12],
	graph14 = c[13], graph15 = c[14], graph16 = c[15], graph17 = c[16], graph18 = c[17], graph19 = c[18], graph20 = c[19], graph21 = c[20],
	graph22 = c[21],graph23 = c[22],graph24 = c[23],graph25 = c[24],graph26 = c[25],graph27 = c[26],graph28 = c[27],graph29 = c[28],
	graph30 = c[29], graph31 = c[30], graph32 = c[31], graph33 = c[32],graph34 = c[33],graph35 = c[34],
	graph36 = c[35], graph37 = c[36], graph38 = c[37],
	graph39 = d[0], graph40 = d[1], graph41 = d[2], graph42 = d[3], graph43 = d[4],
	graph44 = d[5],graph45 = d[6],graph46 = d[7],graph47 = d[8],graph48 = d[9],graph49 = d[10],graph50 = d[11],graph51 = d[12],
	graph52 = d[13], graph53 = d[14], graph54 = d[15], graph55 = d[16], graph56 = d[17], graph57 = d[18], graph58 = d[19], graph59 = d[20],
	graph60 = d[21],graph61 = d[22],graph62 = d[23],graph63 = d[24],graph64 = d[25],graph65 = d[26],graph66 = d[27],graph67 = d[28],
	graph68 = d[29], graph69 = d[30], graph70 = d[31], graph71 = d[32],graph72 = d[33],graph73 = d[34],
	graph74 = d[35], graph75 = d[36], graph76 = d[37])




if __name__ == '__main__':
	app.run(debug=True)





