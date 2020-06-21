import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataset=pd.read_csv('advertising.csv')
dataset.head()
dataset.info()
des=dataset.describe()

# EDA

sns.set_style('whitegrid')
sns.countplot(x='Age',data=dataset)
sns.jointplot(x='Age', y='Area Income', data=dataset ,kind='hex' )
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=dataset ,kind='kde' )
sns.jointplot(y='Daily Internet Usage', x='Daily Time Spent on Site', data=dataset ,kind='kde' )
sns.pairplot(dataset)


# model training


X=dataset.drop(['Clicked on Ad','Ad Topic Line','City','Country','Timestamp'] ,axis=1).values
y=dataset['Clicked on Ad'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
log.score(X_test,y_test)


# Prediction And Evaluation


pred=log.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
matrix=confusion_matrix(y_test,pred)

report=classification_report(y_test, pred)