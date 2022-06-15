# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:22:33 2022

@author: junho
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['PRICE'] = boston.target

x_train, x_test, y_train, y_test = train_test_split(df.drop('PRICE', axis=1), df['PRICE'], test_size=0.2)
# x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

# x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

d_max = 20
r2_test = []
# 여기선 test 데이터로 depth를 찾는데 사용했으나
# 원래는 evaluation 데이터도 나눠서 depth를 찾고 test데이터는 최종 모델 평가에서만 사용
# cross validation test로 train eval test 데이터도 여러 간격으로 나눠서 진
for i in range(1, d_max+1):
    reg_dt = DecisionTreeRegressor(max_depth=i)
    reg_dt.fit(x_train, y_train)
    r2_test.append(reg_dt.score(x_test, y_test))

plt.figure()
plt.plot(r2_test, label='test data')
plt.xlabel('max_depth')
plt.ylabel('R2')
plt.legend()
plt.xticks(np.arange(d_max), np.arange(1,d_max+1))
plt.show()

opt_depth = np.argmax(r2_test) + 1
opt_acc = r2_test[opt_depth]
print(f'max_depth = {opt_depth}, R2 = {opt_acc}')

# 최종 모델
model = DecisionTreeRegressor(max_depth=opt_depth)
model.fit(x_train, y_train)



# 최종 모델 성능 확인
# R2 : 방법-1
print(f'R2 = {model.score(x_test, y_test):.4f}')

# R2 : 방법-2
y_pred = model.predict(x_test)
print(f'R2 = {r2_score(y_test, y_pred):.4f}')

# MSE
y_pred = model.predict(x_test)
print(f'MSE = {mean_squared_error(y_test, y_pred):.4f}')


# feature 들으 ㅣ중요도 분석
feature_importance = model.feature_importances_
feature_importance

n_feature = x_train.shape[1]
idx = np.arange(n_feature)

plt.barh(idx, feature_importance, align='center')
plt.yticks(idx, boston['feature_names'], size=12)
plt.xlabel('importance',size=15)
plt.ylabel('feature', size=15)
plt.show()









