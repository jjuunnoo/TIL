# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:11:12 2022

@author: junho
"""
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# mnist['data'] 같은애들끼리 모아보자 
with open('C:/Users/junho/Desktop/study/py/data/mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)

X = np.array(mnist['data'][:1000])/255
y_real = np.array(mnist['target'])


np.random.choice(X, 5)

km = KMeans(n_clusters=10, init='random', n_init=100, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(X)
y_km = km.predict(X)

fig = plt.figure(figsize=(12,8))
for cluster in range(0,10):
    for i in range(1, 11):
        plt.subplot(10, 10, i+(cluster*10))
        plt.imshow(X[np.where(y_km==cluster)][i].reshape(28,28))
        # plt.title(f'{i}')
        plt.axis('off')
plt.tight_layout()
plt.show()
    
