# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:02:21 2022

@author: junho
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as  plt
import os
import pickle



# mnist = fetch_openml('mnist_784')
# with open(os.getcwd()+'\\mnist.pkl', 'wb') as f:
#     pickle.dump(mnist,f)
    
# mnist data >> logistic regression / l2 regulaization  = C , C에 따라서 조정 
# 트레인 80  x_test, y_test
# 잘못맞춘 이미지 몇장 출력해보기 
# 1. 비교성능, 정확도
# 2. C에 따라서
# 3. 잘못맞춘 이미지 몇장만 출력 


with open('C:/Users/junho/Desktop/study/py/data/mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)
    
print(mnist.keys())

x_feat = np.array(mnist['data'])
# 표준화
x_feat = np.array(mnist['data']) / 255
y_target = np.array(mnist['target'])

x_train, x_test, y_train, y_test = train_test_split(x_feat, y_target, test_size=0.2)

# logi = LogisticRegression()
# logi.fit(x_train, y_train)
# print(f'* 학습용 데이터로 측정한 정확도 = {logi.score(x_train, y_train):0.2f}')
# print(f'* 시험용 데이터로 측정한 정확도 = {logi.score(x_test, y_test):0.2f}')

logi = LogisticRegression()
param_logi = [{'C':np.arange(0.1, 3, 0.5)}]
grid = GridSearchCV(estimator = logi, param_grid = param_logi, cv = 5, refit=True)
grid.fit(x_train, y_train)

logi_best = grid.best_estimator_
print('LogisticRegression: Best paramter =', grid.best_params_)
print(f'LogisticRegression: train data의 score = {logi_best.score(x_train, y_train):0.3f}')
print(f'LogisticRegression: test data의 score = {logi_best.score(x_test, y_test):0.3f}')


y_pred = logi_best.predict(x_test)
n=10
miss_cls = np.where(y_pred!=y_test)[0]
img = x_test[miss_cls[n]].reshape(28,28)
plt.imshow(img)
plt.title(f'y_pred = {y_pred[miss_cls[n]]}, y_test = {y_test[miss_cls[n]]}')
plt.show()
# y_test와 y_pred가 다른 이미지 출력 
y_pred = logi_best.predict(x_test)
n_sample = 10
miss_cls = np.where(y_pred!=y_test)[0]
miss_sam = np.random.choice(miss_cls, n_sample)
# arr_dif 의 n번째 이미지 선택 

fig, ax = plt.subplots(1, n_sample, figsize=(12,4))
for i, miss in enumerate(miss_sam):
    x = x_test[miss] * 255  # 표준화 단계에서 255로 나누었으므로, 여기서는 곱해준다.
    x = x.reshape(28, 28)   # 이미지 확인을 위해 (28 x 28) 형태로 변환한다.
    ax[i].imshow(x)
    ax[i].axis('off')
    ax[i].set_title(f'y_pred = {y_pred[miss]}, y_test = {y_test[miss]}')




