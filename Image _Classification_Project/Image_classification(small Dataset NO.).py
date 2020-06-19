import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
dataset=fetch_openml('mnist_784')
X=dataset.data
y=dataset.target
y=y.astype('int32')

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,random_state=0,test_size=0.3)


#plot image(1-25)


for i in range(25):
    plt.subplot(5,5,i+1)
    X_plot=X.reshape((70000,28,28))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('label {}'.format(y[i]))
    plt.imshow(X_plot[i],"binary")
    plt.axis('off')
    
# Model Selection
    
    
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
from sklearn.svm import SVC
sv=SVC()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

# Data Fit on Model 

log_reg.fit(X_train,y_train)
dtf.fit(X_train,y_train)
nb.fit(X_train,y_train)
sv.fit(X_train,y_train)
knn.fit(X_train,y_train)

# Lets Check Accuracy on Test Data 
log_reg.score(X_test,y_test)
nb.score(X_test,y_test)
dtf.score(X_test,y_test)
sv.score(X_test,y_test)
knn.score(X_test,y_test)

#Lets Check Accuracy of model on Train Data set
log_reg.score(X_train,y_train)
nb.score(X_train,y_train)
dtf.score(X_train,y_train)
sv.score(X_train,y_train)
knn.score(X_train,y_train)



# Confusion Metrices

y_pred_log=log_reg.predict(X_test)
y_pred_nb=nb.predict(X_test)
y_pred_dtf=dtf.predict(X_test)
y_pred_sv=sv.predict(X_test)
y_pred_knn=knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_log=confusion_matrix(y_test,y_pred_log)
cm_nb=confusion_matrix(y_test,y_pred_nb)
cm_dtf=confusion_matrix(y_test,y_pred_dtf)
cm_sv=confusion_matrix(y_test,y_pred_sv)
cm_knn=confusion_matrix(y_test,y_pred_knn)


from sklearn.metrics import precision_score,recall_score,f1_score
# Logistic Regression
precision_score(y_test,y_pred_log,average='micro')
recall_score(y_test,y_pred_log,average='micro')
f1_score(y_test,y_pred_log,average='micro')
#Decision Tree
precision_score(y_test,y_pred_dtf,average='micro')
recall_score(y_test,y_pred_dtf,average='micro')
f1_score(y_test,y_pred_dtf,average='micro')
# Naives Baiyes
precision_score(y_test,y_pred_nb,average='micro')
recall_score(y_test,y_pred_nb,average='micro')
f1_score(y_test,y_pred_nb,average='micro')
#knn
precision_score(y_test,y_pred_knn,average='micro')
recall_score(y_test,y_pred_knn,average='micro')
f1_score(y_test,y_pred_knn,average='micro')
#SVM
precision_score(y_test,y_pred_sv,average='micro')
recall_score(y_test,y_pred_sv,average='micro')
f1_score(y_test,y_pred_sv,average='micro')





