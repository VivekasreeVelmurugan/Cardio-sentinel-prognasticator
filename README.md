
Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
Data Collection and Processing
NameError
Traceback (most recent call last)
<ipython-input-9-c219b185e440> in <cell line: 1>()
heart_data = pd.read_csv('/content/heart.csv')
heart_data.tail()
age sex cp tres tb ps chol fbs restecg thalach exang oldpeak ca thal target
52 1 0 125 212 0 1 168 0 1.0 2 2 3 0
53 1 0 140 203 1 0 155 1 3.1 0 0 3 0
70 1 0 145 174 0 1 125 1 2.6 0 0 3 0
61 1 0 148 203 0 1 161 0 0.0 2 1 3 0
62 0 0 138 294 1 1 106 0 1.9 2 3 2 0

heart_data.shape
(1025, 14)
heart_data.info()
<class 'pandas.core.frame.DataFrame'>

RangeIndex: 1025 entries, 0 to 1024
Data columns (total 14 columns):

--- ------ -------------- -----
0 age 1025 non-null int64
1 sex 1025 non-null int64
2 cp 1025 non-null int64
3 tres tb ps 1025 non-null int64
4 chol 1025 non-null int64
5 fbs 1025 non-null int64
6 restecg 1025 non-null int64
7 thalach 1025 non-null int64
8 ex ang 1025 non-null int64
9 oldpeak 1025 non-null float64
10 slope 1025 non-null int64
11 ca 1025 non-null int64
12 thal 1025 non-null int64
13 target 1025 non-null int64
data types: float64(1), int64(13)
memory usage: 112.2 KB
heart_data.isnull().sum()
age 0
sex 0
cp 0
Trestbps 0
chol 0
fbs 0
restecg 0
exang 0
oldpeak 0
slope 0
ca 0
thal 0

target 0
D type: int64

#statistical measure of the data
heart_data.describe()
heart_data['target'].value_counts()
1 526
0 499
Name: target, d type: int64
1--> Defective heart
0--> Healthy heart
Splitting the target and the features
X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']
age sex cp tres tb ps chol fbs restecg thalach oldpeak
0 52 1 0 125 212 0 1 168 0 1.0
1 53 1 0 140 203 1 0 155 1 3.1
2 70 1 0 145 174 0 1 125 1 2.6
3 61 1 0 148 203 0 1 161 0 0.0
4 62 0 0 138 294 1 1 106 0 1.9
... ... ... .. ... ... ... ... ... ... ...
1020 59 1 1 140 221 0 1 164 1 0.0
1021 60 1 0 125 258 0 0 141 1 2.8
1022 47 1 0 110 275 0 0 118 1 1.0
1023 50 0 0 110 254 0 0 159 0 0.0
1024 54 1 0 120 188 0 1 113 0 1.4

slope ca thal

0 2 2 3
1 0 0 3
2 0 0 3
Splitting the data into Training data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.2, stratify = Y,
random_state=2)
print(X.shape, X_train.shape,X_test.shape)
(1025, 13) (820, 13) (205, 13)
Model training
Logistic Regression
model = LogisticRegression()

model.fit(X_train, Y_train)
/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458:
ConvergenceWarning: failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
Increase the number of iterations (max_iter) or scale the data as shown in:
https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
n_iter_i = _check_optimize_result(
LogisticRegression())
Model Evaluation
Accuracy Score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data:', training_data_accuracy)
Accuracy on Training data: 0.8524390243902439

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data:', test_data_accuracy)
Accuracy on Test data: 0.8048780487804879
Building a predictive System
input_data=(57,1,2,128,229,0,0,150,0,0.4,1,1,3)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]== 0):
print('The Person does not have heart disease')
else:
print('The person has heart disease')
