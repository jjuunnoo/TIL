# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 08:55:34 2022

@author: junho
"""
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression

# 샘플데이터 생성
def createData(a, b, n):
    resultX = []
    resultY = []
    for i in range(n):
        x = np.random.normal(0.0, 0.5)
        y = a*x + b + np.random.normal(0.0, 0.05)
        resultX.append(x)
        resultY.append(y)
    return np.array(resultX).reshape(-1, 1), np.array(resultY).reshape(-1, 1)

# Train 데이터 세트를 구성한다
X, Y = createData(0.1, 0.3, 1000)
fig = plt.figure(figsize=(5,5))
plt.plot(X, Y, 'ro', markersize=1.5)
plt.show()


model = LinearRegression()
model.fit(X,Y)

# 기울기
a = model.coef_[0][0]
# 절편 
b = model.intercept_[0]
yHat = model.predict(X)
print(f'n* 회귀직선의 방정식 (OLS) : y = {a:0.4f} * x + {b:0.4f}')

fig = plt.figure(figsize = (5,5))
plt.plot(X, Y, 'ro', markersize=1.5)
plt.plot(X, yHat)
plt.show()




# 시험 데이터 전체의 오류를 R-square로 표시한다.
print(f'\n시험 데이터 전체 오류 (R2-score) = {model.score(X, Y):0.4f}')

# R-square를 manual로 계산하고, model.score() 결과와 비교한다.
# SSE : explained sum of square
# SSR : residual sum of square (not explained)
# SST : total sum of square
# R-square : SSE / SST or 1 - (SSR / SST)
ssr = np.sum(np.square(yHat - Y))
sst = np.sum(np.square(Y - Y.mean()))
R2 = 1 - ssr / sst
print(f'R-square = {R2:0.4f}')
