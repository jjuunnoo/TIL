# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:21:14 2022

@author: junho
"""
# K-Means Clustering : Silrouette 계수 확인
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# 시험용 데이터 세트를 구성한다
X, y = make_blobs(n_samples=300, n_features=2, centers=5, cluster_std=1.4, shuffle=True, random_state=10)

# 시험용 데이터를 2차원 좌표에 표시한다
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], marker='o', s=100, alpha=0.5)
plt.grid()
plt.show()

silhouette_list = []
for n in range(2, 10):
    # K-means 알고리즘으로 시험용 데이터를 3 그룹으로 분류한다 (k = 3)
    km = KMeans(n_clusters=n, init='random', n_init=100, max_iter=300, tol=1e-04, random_state=0)
    km = km.fit(X)
    y_km = km.predict(X)

    vals = silhouette_samples(X, y_km, metric='euclidean')
    mean_vals = silhouette_score(X, y_km, metric='euclidean')     # np.mean(vals)
    silhouette_list.append(mean_vals)
    print('n_cluster = {}, 실루엣 스코어 = {:.3f}'.format(n, mean_vals))

plt.plot(range(2, 10), silhouette_list, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()