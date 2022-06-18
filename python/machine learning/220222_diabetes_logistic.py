# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:24:45 2022

@author: junho
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

data = pd.read_csv('C:/Users/junho/Desktop/study/py/data/diabetes.csv')
f_data = np.array(data.drop('Outcome', axis=1))
t_data = np.array(data['Outcome'])
data.head()


# 데이터를 표준화한다. train과 test 데이터를 동시에 표준화 했다.
# 분류의 경우에서는 target 표준화 필요 없음 
# 괜찮을까? 문제라면 무엇이 문제일까?
f_scale = StandardScaler()
f_scaled = f_scale.fit_transform(f_data)

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x_train, x_test, y_train, y_test = train_test_split(f_scaled, t_data, test_size = 0.2)

# Logistic Regression으로 Train 데이터 세트를 학습한다.
model = LogisticRegression()
model.fit(x_train, y_train)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
print(f'* 학습용 데이터로 측정한 정확도 = {model.score(x_train, y_train):0.2f}')
print(f'* 시험용 데이터로 측정한 정확도 = {model.score(x_test, y_test):0.2f}')

# 학습된 w, b를 확인해 본다.
print('\nw :')
print(model.coef_)
print('\nb :')
print(model.intercept_)
print('\nclass :')
print(model.classes_)


# x_test[n]의 class를 추정한다.
n = 1
y_pred = model.predict(x_test[n].reshape(1, -1))[0]
print(f'y_test[{n}] = {y_test[n]}, y_pred = {y_pred}')
# 클래스가 0일확률과 1일확률을 표시 
print('probability = ', model.predict_proba(x_test[n].reshape(1, -1))[0])


# 수동으로 확률 계산 
# manual로 x_test[n]의 class를 추정해 본다. 각 파라메터의 기능을 확인한다.
theta = np.dot(model.coef_[0], x_test[n]) + model.intercept_
prob = 1.0 / (1.0 + np.exp(-theta))
print('probability = ', prob)
