# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
### Date:
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: start the program

Step 2: Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

Step 3: Split the data into training and test sets using train_test_split.

Step 4: Create and fit a logistic regression model to the training data.

Step 5: Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

Step 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

Step 7:End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAMYA R
RegisterNumber: 212223230169

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") #libraryfor large linear classificiation
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
### Accuracy:
![l5smrs3d](https://github.com/user-attachments/assets/4a3bcb2f-e93b-4944-9ea0-2b645cd425c9)

### Prediction:
![Screenshot 2024-09-05 094738](https://github.com/user-attachments/assets/5d61b80f-cf7a-43b7-bcd4-82b0cb1b76bb)

![Screenshot 2024-09-05 094832](https://github.com/user-attachments/assets/5def2ad7-11c8-40fe-9a38-11a8f582c22c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
