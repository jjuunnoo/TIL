# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:22:50 2022

@author: junho
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mglearn
import numpy as np
import pandas as pd

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size = 0.2)

# Radial Basis Function(guess 분포), 곡선으로 분리하겠다 rbf
# gamma 가우스 분포의 표준편차 (분포의 폭) 과 관련된 / 곡선의 곡률 관련됨 
# c 허용할 요류의 정도와 관련됨 
model = SVC(kernel = 'rbf', gamma = 0.2, C = 1.0)
model.fit(x_train, y_train)
model.score(x_test, y_test)



# 시각화를 위해 sepal length와 sepal width 만 사용한다.
x = iris.data[:, [2, 3]] # colume 0과 1만 사용함.
y = iris.target

df = pd.DataFrame(data= np.c_[x, y], columns= ['x', 'y', 'target'])
df.head()

plt.scatter(df[df['target'] == 0.0]['x'], df[df['target'] == 0.0]['y'], marker='o', c='blue')
plt.scatter(df[df['target'] == 1.0]['x'], df[df['target'] == 1.0]['y'], marker='x', c='red')
plt.scatter(df[df['target'] == 2.0]['x'], df[df['target'] == 2.0]['y'], marker='v', c='green')
plt.show()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# SVM으로 Train 데이터 세트를 학습한다.
model = SVC(kernel='rbf', gamma=0.2, C=1.0)
model.fit(x_train, y_train)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
print()
print(f' 학습용 데이터로 측정한 정확도 = {model.score(x_train, y_train):0.2f}%' )
print(f' 시험용 데이터로 측정한 정확도 = {model.score(x_test, y_test):0.2f}%' )

# 시각화
plt.figure(figsize=[7,7])
mglearn.plots.plot_2d_classification(model, x_train, alpha=0.3)
mglearn.discrete_scatter(x_train[:,0], x_train[:,1], y_train)
plt.legend(iris.target_names)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()


# gamma와 C의 조합을 바꿔가며 학습데이터의 정확도가 최대인 조합을 찾는다 
optAcc = -999
optG = 0
optC = 0
for gamma in np.arange(0.1, 5.0, 0.1):
    for c in np.arange(0.1, 5.0, 0.1):
        model = SVC(kernel='rbf', gamma=gamma, C=c)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        
        if acc > optAcc:
            optG = gamma
            optC = c
            optAcc = acc
print(f'optG = {optG:0.2f}')
print(f'optC = {optC:0.2f}')
print(f'optAcc = {optAcc:0.2f}')


# 최적 조건으로 다시 학습한다
model = SVC(kernel='rbf', gamma=optG, C=optC)
model.fit(x_train, y_train)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
print()
print(f' 학습용 데이터로 측정한 정확도 = {model.score(x_train, y_train):0.2f}%' )
print(f' 시험용 데이터로 측정한 정확도 = {model.score(x_test, y_test):0.2f}%' )

# 시각화
plt.figure(figsize=[7,7])
mglearn.plots.plot_2d_classification(model, x_train, alpha=0.3)
mglearn.discrete_scatter(x_train[:,0], x_train[:,1], y_train)
plt.legend(iris.target_names)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()



############### support vector regressor ##############

from sklearn.svm import SVR

x = np.arange(1, 19).reshape(-1, 1)
y = np.array([4.14, 3.37, 11.7, 6.3, 10.25, 10.32, 
              5.67, 7.12, 12.47, 17.97, 19.34, 21.62, 
              15.64, 25.83, 29.28, 21.47, 26.3, 32.48])

# reg_svm = SVR(kernel='rbf', gamma=0.1, C=2.0)
reg_svm = SVR(kernel='linear',C=2.0)
reg_svm.fit(x,y)

y_hat = reg_svm.predict(x)
plt.scatter(x,y,c='red', s=100, alpha=0.7)
plt.plot(x, y_hat, marker='o', c='blue', alpha=0.7)
plt.show()


# 실습
# boston house price data
# SVR로 분석 
# dt 와 비교 
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from datetime import datetime

boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['PRICE'] = boston.target


# 데이터 스케일 조정
# 이 부분은 금요일 다시 설명 드리겠습니다.
df['AGE'] /= 10
df['TAX'] /= 100
df['PTRATIO'] /= 10
df['B'] /= 100
df['PRICE'] /= 10


x_train, x_test, y_train, y_test = train_test_split(df.drop('PRICE', axis=1), df['PRICE'], test_size=0.2)

d_max = 20
r2_test = []

for i in range(1, d_max+1):
    reg_dt = DecisionTreeRegressor(max_depth=i)
    reg_dt.fit(x_train, y_train)
    r2_test.append(reg_dt.score(x_test, y_test))
    
opt_depth = np.argmax(r2_test)+1
model_dt = DecisionTreeRegressor(max_depth=opt_depth)
model_dt.fit(x_train, y_train)
print(f'DT_R2 = {model_dt.score(x_test, y_test):0.4f}')


optAcc = -999
optG = 0
optC = 0

# 이 부분도 금요일 다시 설명 드리겠습니다.
EP = [0.5, 1, 5]
C = [10000, 11000, 12000, 20000, 50000, 100000]

# for ep in EP:
#     for c in C:
#         # print(ep, c)
#         model = SVR(kernel='rbf', C=c, epsilon=ep)
#         model.fit(x_train, y_train)
#         acc = model.score(x_test, y_test)
        
#         if acc > opt_acc:
#             opt_ep = ep
#             opt_c = c
#             opt_acc = acc


# for i in np.arange(0.1, 5.0, 0.1):
for c in np.arange(0.1, 5.0, 0.1):
    # model = SVR(kernel='rbf', gamma=i, C=c)
    model = SVR(kernel='linear', C=c)
    model.fit(x_train, y_train)
    r2 = model.score(x_test, y_test)

    print(c, datetime.now().strftime('%H:%M:%S'))
    if r2 > optAcc:
        # optG = gamma
        optC = c
        optAcc = r2


svr = SVR(kernel='rbf',gamma=optG, C=optC)
svr.fit(x_train, y_train)
print(f'R2 = {svr.score(x_test, y_test):0.4f}')


# r2 score 비교 
print(f'DT의 R2 = {model_dt.score(x_test, y_test):0.4f}')
print(f'SVR의 R2 = {svr.score(x_test, y_test):0.4f}')