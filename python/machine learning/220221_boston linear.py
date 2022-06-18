# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:51:08 2022

@author: junho
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.2)

linear = LinearRegression()
linear.fit(x_train,y_train)
y_pred = linear.predict(x_test)

print(f'\n시험 데이터 전체 오류 (R2-score) = {linear.score(x_test, y_test):0.4f}')
# R-square를 manual로 계산하고, model.score() 결과와 비교한다.
# SSE : explained sum of square
# SSR : residual sum of square (not explained)
# SST : total sum of square
# R-square : SSE / SST or 1 - (SSR / SST)
ssr = np.sum(np.square(y_pred - y_test))
sst = np.sum(np.square(y_test - y_test.mean()))
R2 = 1 - ssr / sst
print(f'R-square = {R2:0.4f}')







############################ 표준화로 진행#########################


# 데이터를 표준화한다, train과 test 데이터를 동시에 표준화
# 문제점이 있을지 고민
# 표준화 > split FM이 아님, 학습(표준화)에 test데이터가 사용됨 
# but 문제점을 알고 있으며 실무적으로 data 많으면 대체적으로 분포가 유사하다는 것을 알고 그냥 사용함
# split > 표준화 과정이 FM이지만 분포가 다를때 문제가 생길 수 있음
boston = load_boston()
f_scale = StandardScaler()
t_scale = StandardScaler()

f_scaled = f_scale.fit_transform(boston.data)
t_scaled = t_scale.fit_transform(boston.target.reshape(-1,1))

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x_train, x_test, y_train, y_test = train_test_split(f_scaled, t_scaled, test_size = 0.2)

# Logistic Regression으로 Train 데이터 세트를 학습한다.
model = LinearRegression()
model.fit(x_train, y_train)

# x_test[n]에 해당하는 target (price)을 추정한다.
# n = 1
# y_pred = model.predict(x_test[n].reshape(1, -1))
# y_pred
# # 복원
# y_pred = t_scale.inverse_transform(y_pred)
# y_true = t_scale.inverse_transform(y_test[n].reshape(-1, 1))

# print(f'test{n}의 추정 price = {y_pred[0][0]:0.2f}')
# print(f'test{n}의 실제 price = {y_true[0][0]:0.2f}')
# print(f'추정 오류 = rmse(추정 price - 실제 price) = {np.sqrt(np.square(y_pred[0][0] - y_true[0][0])):0.2f}' )
# 시험 데이터 전체의 오류를 MSE로 표시한다.
# MSE는 값의 범위가 크다는 단점이 있다.


y_pred = model.predict(x_test)
y_pred = t_scale.inverse_transform(y_pred)
y_true = t_scale.inverse_transform(y_test)

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(f'시험 데이터 전체 오류 (rmse) = {rmse:0.4f}')

# 시험 데이터 전체의 오류를 R-square로 표시한다.
# 범위가 한정되어 MSE보다 좋은 척도다.
print(f'시험 데이터 전체 오류 (R2-score) = {model.score(x_test, y_test):0.4f}')


# 추정 결과를 시각화 한다.
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, c='red', s=30, alpha=0.5)
plt.xlabel("house price")
plt.ylabel("predicted price")
plt.show()












