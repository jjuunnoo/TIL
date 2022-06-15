import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

# 당뇨병 데이터 읽기
diabetes = pd.read_csv(r'C:\Users\junho\Desktop\study\py\data\diabetes.csv',encoding = 'utf-8')

# train, test 데이터 분리 8:2
x_train, x_test, y_train, y_test = train_test_split(diabetes[diabetes.columns[:-1]], diabetes[diabetes.columns[-1]], test_size = 0.2)

# 최적의 knn 찾기
max_neighbor = 20
acc_knn_test = []
acc_knn_train = []
for i in range(1, max_neighbor+1):
    knn = KNeighborsClassifier(n_neighbors = i, p=2, weights='distance')
    knn.fit(x_train, y_train)
    acc_knn_test.append(knn.score(x_test, y_test))
    acc_knn_train.append(knn.score(x_train, y_train))

plt.figure()
plt.plot(acc_knn_test, label='test data')
plt.plot(acc_knn_train, label='train data')
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.xticks(np.arange(max_neighbor), np.arange(1, max_neighbor+1))
plt.legend()
plt.show()

opt_knn = np.argmax(acc_knn_test)
opt_acc_knn = acc_knn_test[opt_knn]
print(f'n_neighbor: {opt_knn+1}, accuracy: {opt_acc_knn}')


# 최적의 decision tree depth 찾기 
d_max = 30
acc_dt_test = []
acc_dt_train = []
for i in range(1, d_max+1):
    dt = DecisionTreeClassifier(max_depth = i)
    dt.fit(x_train, y_train)
    acc_dt_test.append(dt.score(x_test, y_test))
    acc_dt_train.append(dt.score(x_train, y_train))
    
plt.figure(figsize=(12,8))
plt.plot(acc_dt_test, label='test data')
plt.plot(acc_dt_train, label='train data')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(np.arange(d_max), np.arange(1, d_max+1))
plt.legend()
plt.show()


opt_dt = np.argmax(acc_dt_test)
opt_acc_dt = acc_dt_test[opt_dt]
print(f'max_depth: {opt_dt+1}, accuracy: {opt_acc_dt}')



x_new = np.array([3, 127, 85, 25, 473, 27, 25, 37]).reshape(-1, 8)

knn = KNeighborsClassifier(n_neighbors=opt_knn+1, p=2, weights='distance')
knn.fit(x_train, y_train)
y_knn_pred = knn.predict(x_new)
if y_knn_pred[0] == 0:
    print('"이 환자는 당뇨병이 아님"으로 진단함')
else:
    print('"이 환자는 당뇨병임"으로 진단함')
print(f'추정의 정확도 = {opt_acc_knn*100:.2f}%')

dt = DecisionTreeClassifier(max_depth = opt_dt+1)
dt.fit(x_train, y_train)
y_dt_pred = dt.predict(x_new)
if y_dt_pred[0] == 0:
    print('"이 환자는 당뇨병이 아님"으로 진단함')
else:
    print('"이 환자는 당뇨병임"으로 진단함')
print(f'추정의 정확도 = {opt_acc_dt*100:.2f}%')
