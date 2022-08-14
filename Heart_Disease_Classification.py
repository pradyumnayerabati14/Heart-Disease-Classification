# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:51:16 2022

@author: Pradyumna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\ramp1\Desktop\ML\heart disease classification dataset.csv")

#FIlling the values with mean
df['trestbps'].fillna(np.mean(df['trestbps']),inplace=True)
df['chol'].fillna(np.mean(df['chol']),inplace=True)
df['thalach'].fillna(np.mean(df['thalach']),inplace=True)

df.drop('Unnamed: 0',axis=1,inplace=True)

# onehotencoding in sex column and label encoding in target column
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('onehote',OneHotEncoder(handle_unknown='ignore'),['sex']),], remainder='drop')
d=ct.fit_transform(df)
new=pd.DataFrame(d)
df=pd.concat([df,new],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['target']=le.fit_transform(df['target'])


df.drop(['sex'],axis=1,inplace=True)


X = df.drop('target',axis=1)
y = df['target']

X.drop('fbs',axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# model Selection
accuracies = {}

lr = LogisticRegression()
lr.fit(x_train,y_train)
acc = lr.score(x_test,y_test)*100

accuracies['Logistic Regression'] = acc
print("Test Accuracy of logistic Regression {:.2f}%".format(acc))   #83.61

#KNN
# try ro find best k value
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracies['KNN']=knn.score(x_test, y_test)*100

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test, y_test)*100))  #75.41

#svm
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)

acc = svm.score(x_test,y_test)*100
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))  #86.69

#Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train, y_train)

acc = rf.score(x_test,y_test)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))    #83.61
    
#Comaring Models
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()    


#Confusion matrix
# Predicted values
y_head_lr = lr.predict(x_test)
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(x_train, y_train)
y_head_knn = knn3.predict(x_test)
y_head_svm = svm.predict(x_test)

y_head_rf = rf.predict(x_test)
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_svm = confusion_matrix(y_test,y_head_svm)
cm_rf = confusion_matrix(y_test,y_head_rf)
plt.title("Random Forest Algorithm")
sns.heatmap(cm_rf,annot=True)


