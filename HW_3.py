#!/usr/bin/env python
# coding: utf-8

# In[393]:


# Anthony Perales
# 801150315
#=============================== Homework 2 ======================================
# PROBLEM 1

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn import metrics

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

outcome = []
cancer = load_breast_cancer()

X = cancer.data[:]
#X = X.reshape(-1,1)
Y = cancer.pop('target')

for i in range(len(Y)):
        if (Y[i] == 1):
            outcome.append('malignant')
        else:
            outcome.append('benign')
        

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 42)

Y_test = Y_test.reshape(-1,1)

Y = Y.reshape(-1,1)


# In[394]:


cancer = load_breast_cancer()
cancer_data = cancer.data

sc =StandardScaler()

# ======================================= Gaussian Model =============================
model = GaussianNB()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

 # num = K -----------------  K = 1

param_grid_nb = {
    'var_smoothing': np.logspace(0,-9,num = 1)
}

model_grid = GridSearchCV(estimator = GaussianNB(), param_grid = param_grid_nb, verbose = 1, cv = 10, n_jobs = -1)
model_grid.fit(X_train_std, Y_train)
predicted = model_grid.predict(X_test_std)
print(model, 'K = 10')
print(metrics.classification_report(Y_test, predicted))
acc = accuracy_score(Y_test, predicted)
pre = precision_score(Y_test, predicted)
rec = recall_score(Y_test, predicted)

clfd = {'Accuracy':acc, 'Precision': pre, 'Recall': rec}
names = list(clfd.keys())
values = list(clfd.values())

 # num = K -----------------  K = 10

param_grid_nb = {
    'var_smoothing': np.logspace(0,-9,num = 10)
}

model_grid = GridSearchCV(estimator = GaussianNB(), param_grid = param_grid_nb, verbose = 1, cv = 10, n_jobs = -1)
model_grid.fit(X_train_std, Y_train)
predicted = model_grid.predict(X_test_std)
print(model, 'K = 5')
print(metrics.classification_report(Y_test, predicted))
acc = accuracy_score(Y_test, predicted)
pre = precision_score(Y_test, predicted)
rec = recall_score(Y_test, predicted)

clfd = {'Accuracy':acc, 'Precision': pre, 'Recall': rec}
names = list(clfd.keys())
values = list(clfd.values())

 # num = K -----------------  K = 3

param_grid_nb = {
    'var_smoothing': np.logspace(0,-9,num = 3)
}

model_grid = GridSearchCV(estimator = GaussianNB(), param_grid = param_grid_nb, verbose = 1, cv = 10, n_jobs = -1)
model_grid.fit(X_train_std, Y_train)
predicted = model_grid.predict(X_test_std)
print(model, 'K = 20')
print(metrics.classification_report(Y_test, predicted))
acc = accuracy_score(Y_test, predicted)
pre = precision_score(Y_test, predicted)
rec = recall_score(Y_test, predicted)

clfd = {'Accuracy':acc, 'Precision': pre, 'Recall': rec}
names = list(clfd.keys())
values = list(clfd.values())


#======================================== Linear Regression Model

LRmodel = LogisticRegression(C = 5, solver = 'liblinear')
LRmodel.fit(X_train_std, Y_train)
predictedLR = LRmodel.predict(X_test_std)
print(LRmodel)
print(metrics.classification_report(Y_test, predictedLR))

LRacc = accuracy_score(Y_test, predictedLR)
LRpre = precision_score(Y_test, predictedLR)
LRrec = recall_score(Y_test, predictedLR)

LRclfd = {'Accuracy':LRacc, 'Precision': LRpre, 'Recall': LRrec}
LRnames = list(LRclfd.keys())
LRvalues = list(LRclfd.values())

X_axis = np.arange(len(clfd))
fig = plt.figure(figsize = (8,3))
plt.bar(X_axis - 0.2, values, 0.3, label = 'GNB')
plt.bar(X_axis + 0.2, LRvalues, 0.3, label = 'LR')
plt.xticks(X_axis, names)
plt.xlabel('Classification Report')
plt.ylabel('%')
plt.legend()
plt.show()


# In[441]:


#=================================== PROBLEM 2 =======================
from sklearn.decomposition import PCA
pca = PCA()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

C = [1, 8, 500, 0.1, 0.5, 0.01, 10]


for i in range(len(C)):
    PCAmodel = LogisticRegression(C=C[i], solver = 'liblinear')
    PCAmodel.fit(X_train_pca, Y_train)
    predictedPCA = PCAmodel.predict(X_test_pca)
    acc = accuracy_score(Y_test, predictedPCA)
    pre = precision_score(Y_test, predictedPCA)
    rec = recall_score(Y_test, predictedPCA)
    PCAclfd = {'Accuracy':acc, 'Precision': pre, 'Recall': rec}
    print('C:', C[i])
    print(metrics.classification_report(Y_test, predictedPCA))

PCAnames = list(PCAclfd.keys())
PCAvalues = list(PCAclfd.values())
X_axis = np.arange(len(PCAclfd))

fig = plt.figure(figsize = (16,9))
plt.bar(X_axis, PCAvalues, 0.1, label = C[0])
plt.bar(X_axis + 0.1, PCAvalues, 0.1, label = C[1])
plt.bar(X_axis - 0.1, PCAvalues, 0.1, label = C[2])
plt.bar(X_axis + 0.2, PCAvalues, 0.1, label = C[3])
plt.bar(X_axis - 0.2, PCAvalues, 0.1, label = C[4])
plt.bar(X_axis + 0.3, PCAvalues, 0.1, label = C[5])
plt.bar(X_axis - 0.3, PCAvalues, 0.1, label = C[6])

plt.xticks(X_axis, PCAnames)
plt.xlabel('Classification Report')
plt.ylabel('%')
plt.legend()
plt.show()


# In[396]:


# ======================== Problem 3 =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
cancer = load_breast_cancer()
cancer_data = cancer.data

X = cancer_data[:,0]
X = X.reshape(-1,1)
Y = cancer.pop('target')


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, random_state = 42)

sc =StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

model = LogisticRegression(solver='liblinear')
model.fit(X_train_std, Y_train)
predicted = model.predict(X_test_std)
report = classification_report(Y_test, predicted)
print(report)

acc = accuracy_score(predicted, Y_test)
pre = precision_score(predicted, Y_test)
rec = recall_score(predicted, Y_test)

clfd = {'Accuracy':acc, 'Precision': pre, 'Recall': rec}
names = list(clfd.keys())
values = list(clfd.values())

fig = plt.figure(figsize = (8,3))
plt.bar(names, values, width = 0.5)
plt.show()

C = [0.01, 0.04, 0.09, 0.1]

for c in C:
    clfd = LogisticRegression(penalty = 'l1', C=c, solver='liblinear')
    clfd.fit(X_train_std, Y_train)
    print('C:', c)
    print('Training accuracy:', clfd.score(X_train_std, Y_train))
    print('Test accuracy:', clfd.score(X_test_std,Y_test))
    print('')



# In[ ]:





# In[ ]:




