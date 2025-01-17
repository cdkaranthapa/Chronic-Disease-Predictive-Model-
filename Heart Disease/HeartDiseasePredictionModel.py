# -*- coding: utf-8 -*-
"""HeartDiseasePrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/182jx1QLk03WNwXtV8hJstBL58t9dyaCd

# Heart Disease Prediction

## First we will import the different Libraries for the model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""## Data collection and Processing"""

# Lodaing the csv data to Pandas Data Frame
heart_data = pd.read_csv('heart_disease_data.csv')

heart_data.head()

heart_data.shape

# Info of data
heart_data.info()

"""### Checking For Missing Values"""

# Checking for missing values
heart_data.isnull().sum()

"""### Statistical measures of data"""

heart_data.describe()

"""### Checking the distribution of Target Variable"""

heart_data['target'].value_counts()

"""1 Represents Disease in heart

0 Represents Healthy Person

### Spliting the features and target
"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

"""### Splitting the data into Training data and Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state = 3)

print(X.shape, X_train.shape, X_test.shape)

"""## Model Training

### Training the data with Logistic Regression model
"""

model = LogisticRegression()

model.fit(X_train, Y_train)

"""### Model Evaluation"""

# Accuracy on trainig data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy on Traing Data: ", training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy on Test Data: ", test_data_accuracy)

"""### Building a Predictive System"""

input_data = (58,1,0,114,318,0,2,140,0,4.4,0,3,1)

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print("The Person does not have a Heart Disease")
else:
    print("The Person has Heart Disease")

"""This model Works with both type of the target data as test by using data from the CSV file"""
