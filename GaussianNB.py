# -*- coding: utf-8 -*-
"""
Gaussian Naive Bayes Classifier
高斯貝氏分類器
iris dataset
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

iris =load_iris()
X =iris.data
y =iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

model =GaussianNB()
model.fit(X_train,y_train)
model.predict(X_test)

model.predict_proba(X_test)
print('Training set score:',model.score(X_train,y_train))
print('Testing set score:',model.score(X_test,y_test))