# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:40:30 2022

@author: junho
"""
# H-Clustering 연습
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np

# 시험용 데이터 세트를 구성한다
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

# 시험용 데이터를 2차원 좌표에 표시한다
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], marker='o', s=100, alpha=0.5)
plt.grid()
plt.show()

# linkage를 계산한다.
mergings = linkage(X, method='complete')

# Dendrogram을 그린다.
plt.figure(figsize=(8,6))
dendrogram(mergings)
plt.show()

# 분류 결과를 표시한다
y_clust = fcluster(Z=mergings, t=3, criterion='distance')

# 중점의 좌표를 계산한다.
c1 = np.mean(X[y_clust == 1], axis=0)
c2 = np.mean(X[y_clust == 2], axis=0)
c3 = np.mean(X[y_clust == 3], axis=0)
centroid = np.vstack([c1, c2, c3])

plt.figure(figsize=(8, 6))
plt.scatter(X[y_clust == 1, 0], X[y_clust == 1, 1], s=100, c='green', marker='s', alpha=0.5, label='cluster 1')
plt.scatter(X[y_clust == 2, 0], X[y_clust == 2, 1], s=100, c='orange', marker='o', alpha=0.5, label='cluster 2')
plt.scatter(X[y_clust == 3, 0], X[y_clust == 3, 1], s=100, c='blue', marker='v', alpha=0.5, label='cluster 3')
plt.scatter(centroid[:,0], centroid[:,1], s=250, marker='+', c='red', label='centroids')
plt.legend()
plt.grid()
plt.show()