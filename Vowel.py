#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.decomposition import PCA
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier


# In[ ]:


#data pre-processing vowels:
dire = 'Vowels'
label = np.concatenate(([0]*100,[1]*100,[2]*100), axis = 0)
filenames = (os.listdir(dire))
for name in filenames:
    f_name = dire + '/' + name
    subject = name.split('_')
    data = loadmat(f_name)
    data_IS = data[list(data.keys())[-1]]
    data_tensor = [data_IS[0][0]]
    for j in range(len(data_IS)):
        if j == 0:
            k = 1
        else:
            k = 0
        for i in range(k,len(data_IS[j])):
            temp = [data_IS[j][i]]
            data_tensor = np.concatenate((data_tensor,temp), axis = 0)
    cov = Covariances(estimator = 'lwf')
    ts = TangentSpace()
    cov.fit(data_tensor,label)
    cov_train = cov.transform(data_tensor)
    ts.fit(cov_train,label)
    ts_train = ts.transform(cov_train)
    ts_shape = (np.shape(ts_train))
    pca = PCA()
    ann = MLPClassifier(max_iter = 5000)
    clf = BaggingClassifier(base_estimator=ann, bootstrap = True)
    pipe = Pipeline(steps = [('pca',pca),('clf',clf)])
    param_grid = {'pca__n_components' : [20, 30, 40, 50, 60, 70, 80, 90, 100],
                  'clf__base_estimator__hidden_layer_sizes' : [(10), (20), (30),(40),(50), (60),(70),(80), (90),(100),(110), (120),(130),(140), (150),(160),(170), (180)],
                  'clf__n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]}
    search = GridSearchCV(pipe, param_grid, cv = 10, n_jobs = -1, return_train_score = True)
    grid = search.fit(ts_train, label)
    write_file = dire + '.txt'
    with open(write_file, 'w+') as file:
        file.write(subject[1])
        file.write('test result:')
        for i in search.cv_results_['mean_test_score']:
            file.write(str(i))
            file.write("\n")
        for i in search.cv_results_['std_test_score']:
            file.write(str(i))
            file.write("\n")
        file.write('train result:')
        for i in search.cv_results_['mean_train_score']:
            file.write(str(i))
            file.write("\n")
        for i in search.cv_results_['std_train_score']:
            file.write(str(i))
            file.write("\n")

