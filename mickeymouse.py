import matplotlib.pyplot as plt 
import numpy as np 
import random
import kmeans

def gen_data():
	points1 = np.random.multivariate_normal(
		np.array([0,0]),
		np.array([[0.4,0],[0,0.4]]),
		80)
	points2 = np.random.multivariate_normal(
		np.array([20,20]),
		np.array([[0.1,0],[0,0.1]]),
		10)
	points3 = np.random.multivariate_normal(
		np.array([-20,20]),
		np.array([[0.1,0],[0,0.1]]),
		10)
	return np.array(list(points1) + list(points2) + list(points3))

def compare_dist(iters=100):
	pts = gen_data()
	cs = []
	dist = []
	for i in range(iters):
		c,_ = kmeans.k_means(pts, 3)
		cpp,_ = kmeans.k_means(pts, 3, kmeans.k_means_pp_initialization)
		d = kmeans.compute_avg2_distortion(pts, c)
		dpp = kmeans.compute_avg2_distortion(pts, cpp)
		dist.append([d,dpp])
		cs.append(np.array([c,cpp]))
	return np.array(dist), np.array(cs)