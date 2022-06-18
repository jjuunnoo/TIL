# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:13:38 2022

@author: junho
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# iris data set을 읽어온다
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2)

# Gaussian model로 Train 데이터 세트를 학습한다.
model = GaussianNB()
model.fit(x_train, y_train)

print('\n* Gaussian model :')
print(f'* 학습용 데이터로 측정한 정확도 = {model.score(x_train, y_train):0.2f}')
print(f'* 시험용 데이터로 측정한 정확도 = {model.score(x_test, y_test):0.2f}' )
