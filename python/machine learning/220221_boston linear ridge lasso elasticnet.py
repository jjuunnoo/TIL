# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:48:32 2022

@author: junho
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline

boston = load_boston()

# x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.2)


# 표준화 
f_scale = StandardScaler()
t_scale = StandardScaler()

f_scaled = f_scale.fit_transform(boston.data)
t_scaled = t_scale.fit_transform(boston.target.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(f_scaled, t_scaled, test_size = 0.2)

# rd = Ridge()
# ridge_params = [{'alpha':np.arange(0.1,10,0.1)}]
# grid_ridge = GridSearchCV(estimator=rd, param_grid = ridge_params, cv=5,refit=True)
# grid_ridge.fit(x_train, y_train)
# grid_ridge.best_estimator_.score(x_test, y_test)

# lasso = Lasso()
# lasso_params = [{'alpha':np.arange(0.1,10,0.1)}]
# grid_lasso =  GridSearchCV(estimator=lasso, param_grid = lasso_params, cv=5,refit=True)
# grid_lasso.fit(x_train, y_train)
# grid_lasso.best_estimator_.score(x_test, y_test)

# elas = ElasticNet()
# elas_params = [{'alpha':np.arange(0.1, 10, 0.1),
#                'l1_ratio':np.arange(0.1, 0.9, 0.1)}]
# grid_elas =  GridSearchCV(estimator=elas, param_grid = elas_params, cv=5,refit=True)
# grid_elas.fit(x_train, y_train)
# grid_elas.best_estimator_.score(x_test, y_test)

pipe = Pipeline(steps=[('model', Ridge())])
grid_params = [{'model':[LinearRegression()],
                'model__fit_intercept':[True, False]},
               
               {'model':[Ridge()],
                'model__alpha':np.arange(0.1, 10, 0.1)},
               
               {'model':[Lasso()],
                'model__alpha':np.arange(0.1, 10, 0.1)},
               
               {'model':[ElasticNet()],
                'model__alpha':np.arange(0.1, 10, 0.1),
                'model__l1_ratio':np.arange(0.1, 0.9, 0.1)}
               ]

grid = GridSearchCV(estimator=pipe, param_grid=grid_params, cv=5, refit=True)
grid.fit(x_train, y_train)
best_model = grid.best_estimator_
print('Best parameter = ', grid.best_params_)
print('Best test score = ', best_model.score(x_test, y_test))


y_pred = best_model.predict(x_test)

# 표준화 복구
y_pred = t_scale.inverse_transform(y_pred)
y_test = t_scale.inverse_transform(y_test)

df_pred = pd.DataFrame(np.c_[y_test, y_pred], columns=['price', 'prediction'])

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, c='red', s=30, alpha=0.5)
plt.xlabel('house price')
plt.ylabel('predicted price')
plt.show()