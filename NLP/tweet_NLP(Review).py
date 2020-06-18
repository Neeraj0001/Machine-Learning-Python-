import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()
df = pd.read_csv('train.csv',engine='python')
clean_review=[]
l=len(df.index)
for i in range(l):
    word=re.sub('[^a-zA-Z]',' ',df['tweet'][i])
    word=word.lower()
    word=word.split()
    word=[ps.stem(text) for text in word if not text in set(stopwords.words('english'))]
    word=' '.join(word)
    clean_review.append(word)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
X=cv.fit_transform(clean_review)
X=X.toarray()
y=df['label']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)




from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

from sklearn.svm import SVC
svm=SVC()
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()


log_reg.fit(X_train,y_train)
knn.fit(X_train,y_train)
svm.fit(X_train,y_train)
nb.fit(X_train,y_train)
dtf.fit(X_train,y_train)

log_reg.score(X_train,y_train)
knn.score(X_train,y_train)
svm.score(X_train,y_train)
nb.score(X_train,y_train)
dtf.score(X_train,y_train)


log_reg.score(X_test,y_test)
knn.score(X_test,y_test)
svm.score(X_test,y_test)
nb.score(X_test,y_test)
dtf.score(X_test,y_test)