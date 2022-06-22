# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:20:25 2022

@author: junho
"""
# K-Means Clustering 연습
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 시험용 데이터 세트를 구성한다
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

print(X[:10, :])

# 시험용 데이터를 2차원 좌표에 표시한다
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], marker='o', s=100, alpha=0.5)
plt.grid()
plt.show()

# K-means 알고리즘으로 시험용 데이터를 3 그룹으로 분류한다 (k = 3)
# n_init = 10 (default) : Local min에 빠지는 것을 완화함.
# 초기 중심을 다르게 설정하여 (10 번) error가 작은 값을 최종 학습 결과로 선택한다.
km = KMeans(n_clusters=3, init='random', n_init=100, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(X)
y_km = km.predict(X)

# 분류 결과를 표시한다
plt.figure(figsize=(8, 6))
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=100, c='green', marker='s', alpha=0.5, label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=100, c='orange', marker='o', alpha=0.5, label='cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=100, c='blue', marker='v', alpha=0.5, label='cluster 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker='+', c='red', label='centroids')
plt.legend()
plt.grid()
plt.show()