#!/usr/bin/env python3
"""
Performing K-means on a dataset.
"""
import sklearn.cluster as Sc 


def kmeans(X, k):
    """    
Function that performs K-means on a dataset:
Function that performs K-means on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid
means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of
the cluster in C
that each data point belongs to.
    """

    kmeans = Sc.KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    return kmeans.cluster_centers_, y_kmeans
