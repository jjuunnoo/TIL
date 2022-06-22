# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:44:04 2022

@author: junho
"""
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# make_blobs() 로 300개의 데이터 셋, 3개의 cluster 셋, cluster_std=0.5 을 만듬. 
X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=0)

# 길게 늘어난 타원형의 데이터 셋을 생성하기 위해 변환함. 
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)

# feature 데이터 셋과 make_blobs( ) 의 y 결과 값을 DataFrame으로 저장
clusterDF = pd.DataFrame(data=X_aniso, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y

# 데이터 시각화
plt.scatter(clusterDF['ftr1'], clusterDF['ftr2'])
plt.show()

# 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 인자로 받아 클러스터링 결과를 시각화하는 함수  
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*']

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        cluster_legend = 'Cluster ' + str(label)
        
        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70, marker=markers[label], label=cluster_legend)
        
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white', alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', marker='$%d$' % label)
    
    legend_loc='upper right'
    plt.legend(loc=legend_loc)
    plt.show()

# KMeans clustering
kmeans = KMeans(3, random_state=0)
kmeans_label = kmeans.fit_predict(X_aniso)
clusterDF['kmeans_label'] = kmeans_label

visualize_cluster_plot(kmeans, clusterDF, 'kmeans_label',iscenter=True)

# GMM clustering
gmm = GaussianMixture(n_components=3, random_state=0).fit(X_aniso)
gmm_cluster_labels = gmm.predict(X_aniso)
clusterDF['gmm_cluster'] = gmm_cluster_labels

visualize_cluster_plot(None, clusterDF, 'target', iscenter=False)

p = gmm.predict_proba(X_aniso)
cluster = np.argmax(p, axis=1)
(gmm_cluster_labels != cluster).sum()