# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:43:52 2019

@author: Anonymous
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
df = pd.read_csv(r"D:\ML and NN courses\Machine Learning by Andrew Ng\machine-learning-programming-assignments-coursera-andrew-ng-master\machine-learning-ex2\ex2\ex2data1.txt")
X = df.iloc[:, :-1]
Y = df.iloc[:,-1]
admitted = df.loc[Y==1]
notadmitted = df.loc[Y==0]
plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1],s=10,label='Admitted')
plt.scatter(notadmitted.iloc[:,0],notadmitted.iloc[:,1],s=10,label='NotAdmitted')
plt.legend()
plt.show()
model = LogisticRegression()
model.fit(X, Y)
y_predicted = model.predict(X)
print(model.predict_proba(X))
model.score(X,Y)
print(model.score(X,Y))
X_test = ([[45,85]])
y_predicted = model.predict(X_test)
print(y_predicted)
model.predict_proba(X_test)
print(model.predict_proba(X_test))