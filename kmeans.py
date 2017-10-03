import matplotlib.pyplot as plt 
import numpy as np 
import random

MAX_ITER = 1000

def read_toy_dataset():
	"""
	returns toy dataset array (np.array)
	"""
	text_file = open("/Users/kunal/Documents/2016-17/Spring/kondor/hw1/toydata.txt", "r")
	return np.array([line.split() for line in text_file.readlines()], dtype=np.float64)

def compute_avg2_distortion(points, centroids):
	"""
	computes average squared distortion of the given arrangement of points and centroids
	"""
	assignment = classify_points(points, centroids)	
	distortion = 0 
	for i in range(centroids.shape[0]):
		i_pts = points[assignment == i]
		distortion += ((i_pts - centroids[i])**2).sum()
	return distortion


def initialize_centroids(points, k):
	"""
	randomly initializes k points as centroids	
	"""
	return np.array(random.sample(list(points), k))

def dist_from_point(points, point):
	"""
	returns an array of respective distance of points from a given point
	"""
	return ((points-point)**2).sum(axis=1)

def dist_from_point_list(points, point_list):
	"""
	returns an array of respective minimum distances of points from a set of points
	"""
	dists = None
	for point in point_list:
		if dists is None:
			dists = dist_from_point(points, point)
		else:
			dists = np.minimum(dists, dist_from_point(points, point))
	return dists

def k_means_pp_initialization(points, k):
	"""
	uses the kmeans++ initialization procedure to return k centroids
	"""
	centroid_list = list(initialize_centroids(points, 1))
	for i in range(1,k):
		dists = dist_from_point_list(points, centroid_list)
		prob = dists**2 / np.sum(dists**2)
		choice = np.random.choice(range(points.shape[0]),p=prob)
		centroid_list.append(points[choice])
	return np.array(centroid_list)

def classify_points(points, centroids):
	"""
	returns the classification of points according to centroids
	"""
	return np.argmin(np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)

def plot_assignment(points, centroids):
	"""
	plots (using pyplot) a color coded 2D plot of assignments according to centroids
	"""
	assignment = classify_points(points, centroids)
	col_list = ['r', 'g', 'b', 'p', 'y', 'o']
	for i in range(centroids.shape[0]):
		i_pts = points[assignment == i]
		plt.scatter(i_pts[:,0], i_pts[:,1], c=col_list[i])
	plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)

def update_centroids(points, centroids):
	"""
	updates the centroids based on kmeans algorithm (mean of classified points)
	"""
	assignment = classify_points(points, centroids)
	return np.array([points[assignment==i].mean(axis=0) for i in range(centroids.shape[0])])

def k_means(points, k, initialization_fn=initialize_centroids):
	"""
	this implements kmeans using the given initialization function 
	(here we defined the uniform random one, and the kmean++ one)
	it returns the centroids, as well as an array of distortion for each iteration
	"""
	assignment_list = []
	centroids = initialization_fn(points, k)
	assignment = classify_points(points, centroids)
	distortion_trend = [compute_avg2_distortion(points, centroids)]

	for i in range(MAX_ITER):
		centroids = update_centroids(points, centroids)
		assignment_prev = assignment
		assignment = classify_points(points, centroids)
		if np.array_equal(assignment, assignment_prev):
			print("Algorithm converged after {} iterations".format(i+1))
			return centroids, distortion_trend
		distortion_trend.append(compute_avg2_distortion(points, centroids))
	return centroids, distortion_trend

def k_mean_plot(points, k, initialization_fn=initialize_centroids, iters=20):
	"""
	plots (using pyplot) the distortion trend of iters number of independe
	kmeans runs, using the defined k_means function
	"""
	for i in range(iters):
		c, d_trend = k_means(points, k ,initialization_fn)
		plt.plot(d_trend)
	plt.ylabel('$J_{avg^2}$ distortion')
	plt.xlabel('iteration')
