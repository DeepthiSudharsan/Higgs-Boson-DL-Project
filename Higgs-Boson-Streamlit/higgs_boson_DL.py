# IMPORTING LIBRARIES
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from collections import Counter
import seaborn as sb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense, LSTM,Dropout,SimpleRNN
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

st.title('HIGGS BOSON - DEEP LEARNING PROJECT')
st.subheader('Upload the Higgs Boson dataset: (.csv)')
# creating a side bar 
st.sidebar.info("Created By : Deepthi Sudharsan")
# Adding an image to the side bar 
st.sidebar.image("https://www.i-programmer.info/images/stories/News/2014/Jun/A/kagglehiggsboson.jpg", width=None)
st.sidebar.subheader("Contact Information : ")
col1, mid, col2 = st.sidebar.beta_columns([1,1,20])
with col1:
	st.sidebar.subheader("LinkedIn : ")
with col2:
	st.sidebar.markdown("[![Linkedin](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLsu_X_ZxDhuVzjTHvk4eZOmUDklreUExhlw&usqp=CAU)](https://www.linkedin.com/in/deepthi-sudharsan/)")

col3, mid, col4 = st.sidebar.beta_columns([1,1,20])
with col3:
	st.sidebar.subheader("Github : ")
with col4:
	st.sidebar.markdown("[![Github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJGtP-Pq0P67Ptyv3tB7Zn2ZYPIT-lPGI7AA&usqp=CAU)](https://github.com/DeepthiSudharsan)")

# if the user chooses to upload the data
file = st.file_uploader('Dataset')
# browsing and uploading the dataset (strictly in csv format)
dataset = pd.DataFrame()
flag = False
class DL_models:

	def __init__(self,X_train,y_train,X_test,y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

	def MLP(self):
		st.subheader("MULTI LAYER PERCEPTRON")
		clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes = 15, random_state=1)
		clf.fit(self.X_train, self.y_train)
		y_pred = clf.predict(self.X_test)
		st.write("ACCURACY : ",accuracy_score(self.y_test, y_pred))
		st.write("CONFUSION MATRIX : ")
		st.write(confusion_matrix(self.y_test, y_pred))
		st.write("CLASSIFICATION REPORT : ")
		st.write(classification_report(self.y_test, y_pred, target_names = ["b","s"], output_dict=True))

	def basic_ANN(self):
		st.subheader("BASIC ARTIFICIAL NEURAL NETWORK")
		classifier = Sequential()
		classifier.add(Dense(units = 128, activation = 'relu'))
		classifier.add(Dropout(0.2))
		classifier.add(Dense(units = 64))
		classifier.add(Dropout(0.2))
		classifier.add(Dense(units = 1))
		classifier.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ["acc"])
		classifier.fit(self.X_train,self.y_train, batch_size = 32, epochs = 10)
		y_pred = classifier.predict(self.X_test)
		y_pred = [0 if y_pred[i] <=0.5 else 1 for i in range(len(y_pred))]
		st.write("ACCURACY : ",accuracy_score(self.y_test, y_pred))
		st.write("CONFUSION MATRIX : ")
		st.write(confusion_matrix(self.y_test, y_pred))
		st.write("CLASSIFICATION REPORT : ")
		st.write(classification_report(self.y_test, y_pred, target_names = ["b","s"], output_dict=True))

	def autoencoder(self):
		st.subheader("AUTOENCODER")
		# No of encoding dimensions
		encoding_dim = 32
		input_layer = Input(self.X_train.shape[1])
		# Encoder
		encoder = Dense(encoding_dim, activation ="tanh")(input_layer)
		encoder = Dense(int(encoding_dim / 2), activation = "relu")(encoder)
		# Decoder
		decoder = Dense(int(encoding_dim / 2), activation ='tanh')(encoder)
		decoder = Dense(1, activation ='relu')(decoder)
		autoencoder = Model(inputs=input_layer, outputs = decoder)
		autoencoder.compile(optimizer ='adam',loss='mean_squared_error',metrics = ["acc"])
		autoencoder.fit(self.X_train, self.y_train,epochs = 10,batch_size = 32)
		predictions = autoencoder.predict(self.X_test)
		predictions = [0 if predictions[i] <=0.5 else 1 for i in range(len(predictions))]
		st.write("ACCURACY : ",accuracy_score(self.y_test, predictions))
		st.write("CONFUSION MATRIX : ")
		st.write(confusion_matrix(self.y_test, predictions))
		st.write("CLASSIFICATION REPORT : ")
		st.write(classification_report(self.y_test, predictions, target_names = ["b","s"], output_dict=True))

	def RNN(self):
		st.subheader("RECURRENT NEURAL NETWORK")
		Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
		# Reshape the data
		Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
		model = Sequential()
		model.add(SimpleRNN(units=4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam',metrics = ["acc"])
		model.fit(Xtrain, self.y_train, epochs=10, batch_size=32)
		# predicting the opening prices
		prediction = model.predict(Xtest)
		predictions = [0 if prediction[i] <=0.5 else 1 for i in range(len(prediction))]
		st.write("ACCURACY : ",accuracy_score(self.y_test, predictions))
		st.write("CONFUSION MATRIX : ")
		st.write(confusion_matrix(self.y_test, predictions))
		st.write("CLASSIFICATION REPORT : ")
		st.write(classification_report(self.y_test, predictions, target_names = ["b","s"], output_dict=True))

if file is not None:

	dataset = pd.read_csv(file)
	# flag is set to true as data has been successfully read
	flag = "True"
	st.header('**HIGGS BOSON DATA**')
	st.write(dataset.head())
	dataset.drop("EventId", axis = 1, inplace = True)
	
	st.write("PERFORM EXPLORATORY DATA ANALYSIS")
	st.subheader("Correlation plot of the features")
	corr = dataset.corr()
	fig, ax = plt.subplots()
	sb.heatmap(corr,cmap="Blues", ax=ax)
	st.write(fig)
	st.subheader("Labels distribution")
	st.bar_chart(dataset["Label"].value_counts())
	st.subheader("Finding no. of null values per column in the dataset")
	st.write(dataset.isna().sum())
	st.subheader("Statistical information about the datset")
	st.write(dataset.describe())	
	data = dataset.iloc[:,:-1] # Extracting dependent variables
	imp_mean = SimpleImputer(missing_values = -999.0, strategy='mean') 
	# Imputation transformer for completing missing values.
	imp_mean.fit(data)
	X = imp_mean.transform(data)
	y = dataset.iloc[:,-1].values # extracting the labels/independent variables
	train_label = y.tolist()
	class_names = list(set(train_label))
	class_dist = Counter(train_label)
	le = LabelEncoder()  
	y = le.fit_transform(y) # Encoding categorical data to numeric data
	st.success("Data cleaned!")
	st.subheader('Test size split of users choice:')
	st.text('Default is set to 20%')
	k = st.number_input('',step = 5,min_value=10, value = 20)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = k * 0.01, random_state = 0)
	st.write("Data is being split into testing and training data!")
	# Splitting the data into 20% test and 80% training data
	# Outlier detection and removal
	iso = IsolationForest(contamination=0.1)
	train_hat = iso.fit_predict(X_train)
	test_hat = iso.fit_predict(X_test)
	st.success("Data split successfuly")
	st.write("No of Training data outliers :",Counter(train_hat)[-1],"out of ",X_train.shape[0],"data points")
	st.write("No of Testing data outliers :",Counter(test_hat)[-1],"out of ",X_test.shape[0],"data points")
	# select all rows that are not outliers
	mask_train = train_hat != -1
	mask_test = test_hat != -1
	X_train, y_train = X_train[mask_train, :], y_train[mask_train]
	X_test, y_test = X_test[mask_test, :], y_test[mask_test]
	st.success("Outliers removed successfully!")
	std_sc = StandardScaler()
	X_train = std_sc.fit_transform(X_train) # Scaling the training data
	X_test = std_sc.transform(X_test) # Scaling the testing data

	dl = DL_models(X_train,y_train,X_test,y_test)
	st.subheader('Which Deep Learning model would you like to train? :')
	mopt = st.multiselect("Select :",["MLP","Basic ANN","Autoencoder","RNN","All"])
	# "Click to select",
	if(st.button("START TRAINING AND TESTING THE MODEL(S) SELECTED")):
		if "MLP" in mopt:
			dl.MLP()
		if "Basic ANN" in mopt:
			dl.basic_ANN()

		if "Autoencoder" in mopt:
			dl.autoencoder()

		if "RNN" in mopt:
			dl.RNN()

		if "All" in mopt:
			dl.MLP()
			dl.basic_ANN()
			dl.autoencoder()
			dl.RNN()

	if(st.button("FINISH")):
		st.info("YAY! WE ARE DONE TRAINING AND TESTING THE MODELS! HOPE WE MEET AGAIN SOON")
		st.balloons()

else:
	st.warning("No file has been chosen yet")

