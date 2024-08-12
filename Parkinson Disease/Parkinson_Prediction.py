"""

Parkinsson Disease Prediction

Importing the libraries for the model
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and Processing"""

# Loading the csv file data to Pandas Data Frame
p_data = pd.read_csv('Parkinsson disease.csv')

p_data.head()

p_data.shape

p_data.info()

# Checking for missing values
p_data.isnull().sum()

#Statistical Measure
p_data.describe()

# Distribution of target variable
p_data['status'].value_counts()

"""1 --> Parkinsson Positive

0 --> Negative
"""

# Select only numeric columns before calculating the mean
numeric_data = p_data.select_dtypes(include=['number'])
numeric_data.groupby('status').mean()

"""Seprating Target Data and Features"""

X = numeric_data.drop(columns=['status'], axis=1)
Y = numeric_data['status']

print(X)

print(Y)

"""Splitting The data to Training and Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape, X_train.shape, X_test.shape)

print(Y.shape, Y_train.shape, Y_test.shape)

"""Data Standardization"""

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Trainig the model
model = svm.SVC(kernel = 'linear')

model.fit(X_train, Y_train)

#Accuracy Data on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print(training_data_accuracy)

#Accuracy Data on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print(test_data_accuracy)

"""Building a predictive system"""

input_data = (119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)

# Changing the data to numpy array
nparray = np.array(input_data)

# Reshape the data
rshape = nparray.reshape(1,-1)

# Standardize the data
std_data = scaler.transform(rshape)

prediction = model.predict(std_data)

if (prediction[0] == 1):
  print("The person has Parkinsson Disease")
else:
  print("The person does not have Parkinsson Disease")

