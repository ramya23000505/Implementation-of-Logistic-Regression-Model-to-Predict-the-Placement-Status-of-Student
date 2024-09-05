# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
### Date:
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## STEP 1:
start the program
## STEP 2:
Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.
## STEP 3:
Split the data into training and test sets using train_test_split.
## STEP 4:
Create and fit a logistic regression model to the training data.
## STEP 5:
Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.
## STEP 6:
Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.
## STEP 7:
End the program
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAMYA R
RegisterNumber: 212223230169

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#load the california housing dataset
data=fetch_california_housing()

#use the first 3 feature as inputs
X=data.data[:,:3] #features: 'MedInc' , 'HouseAge' , 'AveRooms'

#use 'MedHouseVal' and 'AveOccup' as output variables
Y=np.column_stack((data.target, data.data[:,6])) #targets: 'MedHouseVal' , 'AveOccup'

#split the data into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

#initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

#use multioutputregressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

#train the model
multi_output_sgd.fit(X_train,Y_train)

#predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

#inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

#evaluate the model using mean squared error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

#optionally, print some predictions
print("\nPredictions:\n",Y_pred[:5]) #print first 5 predictions 
*/
```

## Output:
![out](https://github.com/user-attachments/assets/36e5320d-d300-450b-80d5-b91d7411b990)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
