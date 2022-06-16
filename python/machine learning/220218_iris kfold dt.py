# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:06:04 2022

@author: junho
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
import pdb

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size = 0.2)
cv = KFold(n_splits = 5, shuffle=True)


acc_depth = []
for depth in range(1, 20):
    model = DecisionTreeClassifier(max_depth = depth)
    acc_fold = []
    for tx, vx in cv.split(x_train):
        xf_train, xf_eval = x_train[tx], x_train[vx]
        yf_train, yf_eval = y_train[tx], y_train[vx]
        
        pdb.set_trace()
        model.fit(xf_train, yf_train)
        acc_fold.append(model.score(xf_eval, yf_eval))
    
    print(f'Depth-{depth:2d} : fold acculate = {np.round(acc_fold,3)}', end='')
    acc_depth.append(np.mean(acc_fold))
    print(f'-->평균 = {acc_depth[-1]:0.4f}')

opt_depth = np.argmax(acc_depth) + 1
model = DecisionTreeClassifier(max_depth = opt_depth)
model.fit(x_train, y_train)

print(f'\n최적 depth = {opt_depth}')
print(f'시험 데이터 정확도 = {model.score(x_test, y_test):0.4f}')


