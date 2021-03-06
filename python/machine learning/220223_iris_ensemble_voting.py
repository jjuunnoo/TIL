# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:40:11 2022

@author: junho
"""
# Majority voting의 앙상블 방법을 연습한다.
# ----------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

# iris 데이터를 읽어온다.
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size = 0.2)

# 4가지 모델을 생성한다 (KNN, Decision tree, SVM, Logistic Regression).
# 각 모델은 최적 조건으로 생성한다. (knn의 k개수, dtree의 max_depth 등)
# sklearn 문서 : Recommended for an ensemble of well-calibrated classifiers.
knn = KNeighborsClassifier(n_neighbors=5, p=2)
dtree = DecisionTreeClassifier(criterion='gini', max_depth=8)
svm = SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True)
lreg = LogisticRegression(max_iter=500)

# 4가지 모델로 앙상블을 구성한다.
base_model = [('knn', knn), ('dtree', dtree), ('svm', svm), ('lr', lreg)]
ensemble = VotingClassifier(estimators=base_model, voting='soft')

ensemble

# 4가지 모델을 각각 학습하고, 결과를 종합한다.
# VotingClassifier()의 voting = ('hard' or 'soft')에 따라 아래와 같이 종합한다.
# hard (default) : 4가지 모델에서 추정한 class = (0, 1, 2)중 가장 많은 것으로 판정.
# soft : 4가지 모델에서 추정한 각 class의 확률값의 평균 (혹은 합)을 계산한 후,
#        확률이 가장 높은 (argmax(P)) class로 판정한다.
ensemble.fit(x_train, y_train)

# 학습데이터와 시험데이터의 정확도를 측정한다.
print(f'\n학습데이터의 정확도 = {ensemble.score(x_train, y_train):0.3f}')
print(f'시험데이터의 정확도 = {ensemble.score(x_test, y_test):0.3f}')

# 시험데이터의 confusion matrix를 작성하고, (row : actual, col : predict),
# 4개 score를 확인한다.
y_pred = ensemble.predict(x_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=iris.target_names))
