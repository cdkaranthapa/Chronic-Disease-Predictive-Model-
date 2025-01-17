
"""
Importing Libraries
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data

PIMA Diabestes dataset
"""

# loading the dataset in pandas dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

diabetes_dataset.head()

# Number of rows and coluns in dataframe
diabetes_dataset.shape

# Getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 = Nondiabetic
1 = Diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

# Seperating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Data Standardistion"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 1)

print(X.shape, X_train.shape, X_test.shape)

"""Trainig the Model"""

classifier = svm.SVC(kernel = 'linear')

#Training the svm classifier
classifier.fit(X_train, Y_train)

"""Model Evaluation"""

# Accuracy Score on training data
X_train_accuracy = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_accuracy, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy Score on test data
X_test_accuracy = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_accuracy, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""Making A Predictive System"""

input_data = (6,148,72,35,0,33.6,0.627,50)

# Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
  print('The person is not diabetic')
else:
  print('The person is diabetic')

