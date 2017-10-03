import matplotlib.pyplot as plt 
import numpy as np 
import random
import math
import pdb
from scipy.sparse import csr_matrix
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D


def read_3d_data():
	"""
	returns 3d dataset array (np.array)
	"""
	data = open("/Users/kunal/Documents/2016-17/Spring/kondor/hw2/3Ddata.txt", "r")
	data = np.array([line.split() for line in data.readlines()], dtype=np.float64)
	return data[:,0:3], data[:,3]

def plot_3d(data, cols):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:,0],data[:,1],data[:,2], c=cols)

def center_data(data):
	'''
	centers data
	'''
	return (data.T-data.mean(axis=1)).T

def pca(data, k=2):
	'''
	returns first k principal components of data
	'''
	cov = np.cov(data.T)
	evals, evecs = np.linalg.eigh(cov)
	return evecs.T[-k:,:][::-1]

def pca_plot(data, cols, k=2):
	'''
	performs PCA and plots data
	'''
	evecs = pca(data, k)
	proj_data = np.dot(data, evecs.T)
	plt.scatter(proj_data[:,0], proj_data[:,1], c=cols)

def dist_matrix(data):
	'''
	constructs distance matrix using data
	'''
	n = data.shape[0]
	mat = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			mat[i,j] = np.sqrt(((data[i]-data[j])**2).sum())
	return mat

def isomap(data, k=10, d=2, D=None):
	'''
	performs isomap on data, returning the reduced dimensional points
	'''
	if D is None:
		D = k_nn_dist_matrix(data, k)
	n = D.shape[0]
	P = np.identity(n) - (np.ones((n,n)) / n)
	G = - (1/2) * P.dot(D**2).dot(P)
	evals, evecs = np.linalg.eigh(G)
	evals = evals[::-1][:d]
	evecs = evecs.T[::-1][:d]
	return evecs.T * np.sqrt(evals)

def isomap_plot(data, cols, k=10, d=2, D=None):
	'''
	performs isomap and plots data
	'''
	ps = isomap(data, k, d, D)
	plt.scatter(ps[:,0], ps[:,1], c=cols)

def lle_weights(data, k=10, d=2):
	'''
	finds lle_weights
	'''
	mat = dist_matrix(data)
	n = mat.shape[0]
	ind = mat.argsort()[:,1:k+1]
	B = np.zeros((n, k))
	Z = data[ind].transpose(0, 2, 1)
	# Algorithm from http://www.cs.nyu.edu/~roweis/lle/algorithm.html
	# Some shortcuts taken from 
	# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/locally_linear.py
	for i in range(n):
		A = Z[i]
		A_n = A.T - data[i]
		C = np.dot(A_n, A_n.T)
		C += 1e-3 * np.identity(k)
		w = solve(C, np.ones(k), sym_pos=True)
		B[i, :] = w / np.sum(w)
	indptr = np.arange(0, n * k + 1, k)
	#Use csr_matrix as a convenience to put things into the right spots
	return csr_matrix((B.ravel(), ind.ravel(), indptr), shape=(n, n)).todense()

def lle(data, k=10, d=2):
	'''
	performs lle, returning the reduced dimension points
	'''
	W = lle_weights(data, k, d)
	n = W.shape[0]
	A = np.identity(n) - W
	M = A.T * A
	evals, evecs = np.linalg.eigh(M)
	idx = np.argsort(np.abs(evals))
	return evecs.T[idx][1:d+1].T

def lle_plot(data, cols, k=10, d=2):
	'''
	performs lle and plots the new data
	'''
	ps = lle(data, k, d)
	plt.scatter(ps[:,0], ps[:,1], c=cols)

def k_nn_dist_matrix(data, k=10):
	'''
	finds the k_nn graph distance matrix using Floyd-Warshall
	'''
	mat = dist_matrix(data)
	n = mat.shape[0]
	ranks = mat.argsort().argsort()
	mat[ranks > k] = math.inf
	#Floyd's
	for k in range(n):
		for i in range(n):
			for j in range(n):
				if mat[i,j] > mat[i,k] + mat[k,j]:
					mat[i,j] = mat[i,k] + mat[k,j]
	return mat

def col_map(col):
	'''
	maps int to color (as per hw assignment)
	'''
	if col == 1:
		return 'g'
	elif col == 2:
		return 'y'
	elif col == 3:
		return 'b'
	else:
		return 'r'
