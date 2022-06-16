# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:02:02 2022

@author: junho
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size = 0.2)


dt = DecisionTreeClassifier()
# dt 의 파라미터들을 적어준다 
dt_params = [{'max_depth': np.arange(1, 20),
              'criterion':['gini', 'entropy']
              }]

# GridSearchCv : 사용 모델, 모델의 파라미터, kfold, 최적모델 생성까지 한번에 
# refit : 최적 파라미터로 최종 모델 생성
grid = GridSearchCV(estimator=dt, param_grid=dt_params, cv=5, refit=True)
grid.fit(x_train, y_train)



# grid.cv_results_     #: K-fold cross validation test 결과 dictionary
# grid.best_params_    #: best parameter ==> {'max_depth': 5}
# grid.best_estimator_ #: best parameter로 생성한 tree

grid.cv_results_
grid.cv_results_.keys()

# eval 0 일때 depth별 점수 
grid.cv_results_['split0_test_score']


# depth별 평균, 가장 높은것을 찾은게 최적 depth
grid.cv_results_['mean_test_score']

grid.best_params_



# 최적 모델로 시험 데이터의 성능을 평가한다.
best_model = grid.best_estimator_

print('\n최적 파라메터 =', grid.best_params_)
print(f'시험 데이터 정확도 = {best_model.score(x_test, y_test):0.4f}')







