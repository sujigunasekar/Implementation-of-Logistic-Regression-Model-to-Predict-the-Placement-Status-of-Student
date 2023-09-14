# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2.Load the dataset and check for null data values and duplicate data values in the dataframe.
3.Import label encoder from sklearn.preprocessing to encode the dataset.
4.Apply Logistic Regression on to the model.
5.Predict the y values.
6.Calculate the Accuracy,Confusion and Classsification report. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Suji.G
RegisterNumber: 212222230152
import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/52d21019-5ba7-4bdb-bb35-0bd1b907c388)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/d7d01689-8896-40f4-b638-781f8dd88130)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/0cbf55b3-ebac-4414-81dc-227d3a59ac13)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/69934acc-b2a1-446d-b5e0-1d3dd3782c3f)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/fc7f31c4-ea69-4246-9d02-ddb53f520657)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/72098a32-9cc0-4cdd-81b3-332747a1ef68)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/0cfa67d6-9789-4ab1-9ec2-4ca926be831f)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/e74b4b46-5c91-4a54-bd5e-1e216db250fa)

![image](https://github.com/sujigunasekar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559822/53a89f3d-7d1c-4c43-9c36-d88d5732b61b)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
