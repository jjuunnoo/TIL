# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:02:56 2022

@author: junho
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys
import pandas as pd


####################### iris 데이터 knn 적용하기 ########################

# iris 읽어옴
iris = load_iris()

# sample / new 데이터 분리
x_train = iris['data'][:-1]
y_train = iris['target'][:-1]
x_test = iris['data'][-1]

distance_target= [[np.sqrt(np.sum(np.square(loc-x_test))), y_train[i]] for i, loc in enumerate(x_train)]
distance_target.sort()

# k input
k=int(sys.stdin.readline().rstrip())
k_dis = distance_target[:k]


# target count 
k_dis_count = {}
for i in k_dis:
    if f'{i[1]}' in k_dis_count:
        k_dis_count[f'{i[1]}'] += 1
    else:
        k_dis_count[f'{i[1]}'] = 1

print('new_data_target:',max(k_dis_count))



################################### 강사님 코딩 ##############################

# iris 읽어옴
iris = load_iris()

x_train = iris['data'][:-1]
y_train = iris['target'][:-1]
x_test = iris['data'][-1]

distance = np.sqrt(np.sum(np.square(x_train-x_test),axis=1))

# np.c_ : concatenate 하겠다 
df = pd.DataFrame(data = np.c_[distance, y_train], columns = ['distance', 'target'])
# 기본적으로 ascending= True 오름차순 
df.sort_values(by='distance', inplace = True)

K=10
K_dis = df[:K]['target'].to_numpy().astype('int')
counts = np.bincount(K_dis)

# np.max(counts) 하면 8이 나온다 갯수, np.argmax하면 해당 갯수를 가진 값이 나옴 
majority = np.argmax(counts)
print(majority)








###################### weighted로 업데이트 #######################
# iris 읽어옴
iris = load_iris()

# sample / new 데이터 분리
x_train = iris['data'][:-1]
y_train = iris['target'][:-1]
x_test = iris['data'][-1]

distance = np.sqrt(np.sum(np.square(x_train-x_test),axis=1))

df = pd.DataFrame(data = np.c_[distance, y_train], columns = ['distance', 'target'])
df.sort_values(by='distance', inplace = True)

K=10
K_dis = df[:K]
K_dis_weighted = K_dis.groupby('target')['weight'].sum() / K_dis['weight'].sum()
print(int(K_dis_weighted.index[K_dis_weighted==K_dis_weighted.max()][0]))
# int(max(dict(K_dis_weighted)))









###################### KNeighborsClassifier 사용 ############################

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris = load_iris()
train_x, test_x, train_y, test_y = train_test_split(iris['data'], iris['target'], test_size = 0.2)

acc_train = []
acc_test = []


def knn_acc_func(k_max):
    for k in range(1, k_max):
        knn = KNeighborsClassifier(n_neighbors = k, p=2, weights = 'distance')
        knn.fit(train_x, train_y)    
        train_pred_y = knn.predict(train_x)
        accuracy_train = (train_y == train_pred_y).mean()
        
        test_pred_y = knn.predict(test_x)
        accuracy_test = (test_y == test_pred_y).mean()
        
        # accuracy_train = knn.score(train_x, train_y)
        # accuracy_test = knn.score(test_x, test_y)
        acc_train.append(accuracy_train)
        acc_test.append(accuracy_test)
        print(f'k={k} 학습용 데이터 정확도:{accuracy_train:0.4f}')
        print(f'k={k} 시험용 데이터 정확도:{accuracy_test:0.4f}')
k_max = 30
acc_train = []
acc_test = []
knn_acc_func(k_max)

plt.figure()
plt.plot(acc_train, marker = 'o', label = 'train data')
plt.plot(acc_test, marker = 'o', label = 'test data')
plt.legend()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.xticks(np.arange(k_max), np.arange(1,k_max+1))
plt.show()

opt_k = np.argmax(acc_test)
opt_acc = acc_test[opt_k]
print(f'optimal k: {opt_k}, opt_acc: {opt_acc:0.4f}')





####################### knn regression#######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import math
from sklearn.metrics import mean_squared_error 

x = np.array([1.25, 3.52, 3.0, 4.13, 6.51, 6.27, 7.3, 8.42, 8.81, 10.14,10.68,12.6 , 13.42, 14.01, 14.82, 15.96, 17.77, 17.85]).reshape(-1,1)

y = np.array([ 5.64,  2.36, 11.08,  6.62,  9.18, 11.91,  5.61,  7.75, 12.16, 18.18, 18.73, 20.43, 14.86, 26.75, 29.9 , 20.32, 25.04, 31.59])

x_test = np.array([ 1.98,  4.16,  3.43,  4.49,  6.51,  6.76,  7.31,  8.55,  9.69, 10.52, 10.85, 13.29, 13.63, 14.09, 15.28, 16.94, 18.01, 18.7 ]).reshape(-1,1)

plt.scatter(x, y, c='red', s=100)
plt.xticks(x, rotation=90)
plt.show()

# KNN regressor를 생성한다.
knn = KNeighborsRegressor(n_neighbors=5, weights='uniform')
knn.fit(x, y)

# y를 추정한다.
y_hat = knn.predict(x_test)

# 추정된 y를 시각화하고, 육안으로 성능을 확인한다.
plt.scatter(x, y, c='red', s=100, alpha=0.7)
plt.plot(x, y_hat, marker='o', c='blue', alpha=0.7, 
         drawstyle="steps-post")
plt.show()

# x = 12.0일 때 y의 추정치는?
y_hat = knn.predict(np.array([12.0]).reshape(-1,1))
print(f'\nx = 12.0 --> predicted y = {y_hat[0]:0.4f}')








##################### regressor 로 최적 k값 찾기 ###########################
for k in range(1,len(x)):
    knn = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    knn.fit(x,y)
    knn.score(x,y)
    y_pred = knn.predict(x)
    accuracy = (y==y_pred).mean()
    print(accuracy)
    # print(f'mse:{mse}')
    # print(f'rmse:{rmse}')
    
    