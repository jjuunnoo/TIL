# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:30:32 2022

@author: junho
"""
# AdaBoost에 의한 앙상블 방법을 연습한다.
# --------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# iris 데이터를 읽어온다.
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size = 0.2)

svm = SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True)
aboost = AdaBoostClassifier(base_estimator=svm, n_estimators=100)
aboost.fit(x_train, y_train)

# 시험데이터의 confusion matrix를 작성하고, (row : actual, col : predict),
# 4개 score를 확인한다.
y_pred = aboost.predict(x_test)

print('\nConfusion matrix :')
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=iris.target_names))
