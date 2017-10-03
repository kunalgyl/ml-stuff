import matplotlib.pyplot as plt 
import numpy as np 
import random
import math
import pdb

def read_mnist_data():
	"""
	returns mnist dataset: train, train_lables, test (first two are randomly permuted)
	"""
	train = open("/Users/kunal/Documents/2016-17/Spring/kondor/hw2/train35.digits", "r")
	train = np.array([line.split() for line in train.readlines()], dtype=np.float64)
	train = train / np.linalg.norm(train, axis=1, ord=2)[:,None]
	test = open("/Users/kunal/Documents/2016-17/Spring/kondor/hw2/test35-1.digits", "r")
	test = np.array([line.split() for line in test.readlines()], dtype=np.float64)
	test = test / np.linalg.norm(test, axis=1, ord=2)[:,None]
	train_labels = open("/Users/kunal/Documents/2016-17/Spring/kondor/hw2/train35.labels", "r")
	train_labels = np.array([line.split()[0] for line in train_labels.readlines()], dtype=np.float64)
	p = np.random.permutation(len(train_labels))
	return train[p], train_labels[p], test

def cross_val_data(train, train_labels, k=20, m=1):
	'''
	performs cross validation of data using different values of m from 1 to 10
	does 10-fold cross validation
	'''
	splits_d = np.split(train, k)
	splits_l = np.split(train_labels, k)
	err_rates = []
	for i in range(10):
		pmod = PreceptronModel()
		splits_di = np.concatenate(splits_d[:i]+splits_d[i+1:])
		splits_li = np.concatenate(splits_l[:i]+splits_l[i+1:])
		val_di = np.array(splits_d[i])
		val_li = np.array(splits_l[i])
		pmod.batchUpdate(splits_di, splits_li, m)
		pred_li = pmod.batchPredict(val_di)
		err_rates.append((val_li == pred_li).sum() / len(pred_li))
	return np.array(err_rates).mean()

def train_model(m=7):
	'''
	return a PreceptonModel trained on m runs of the data
	Default value of 7 obtained using cross validation (see plot_m)
	'''
	train, train_labels, test = read_mnist_data()
	pmod = PreceptronModel()
	pmod.batchUpdate(train, train_labels, m)
	return pmod

def plot_mistakes(m=7, pmod=None):
	'''
	plots the mistakes against examples seen
	'''
	if pmod is None:
		pmod = train_model(m)
	ms = pmod.mistakes_seq
	plt.plot(range(len(ms)), ms)

def plot_m():
	'''
	plots the err_rates using cross validation, for different values of m
	'''
	train, train_labels, test = read_mnist_data()
	inds = []
	rates = []
	for i in range(1,10):
		inds.append(i)
		rates.append(cross_val_data(train, train_labels, m=i))
	plt.plot(inds, rates, 'ro')

def save_preds(preds):
	'''
	saves the new prediction on the test data
	'''
	test_labels = open("/Users/kunal/Documents/2016-17/Spring/kondor/hw2/test35.labels", "w+")
	for pred in preds:
		print("{}".format(pred), file=test_labels)

def plot_examples(feats, nrows=4, ncols = 5):
	'''
	plots a few sample digits
	'''
	plt.figure(figsize=(ncols, nrows))
	for i in range(nrows*ncols):
		plt.subplot(nrows, ncols, i+1)
		plt.imshow(feats[i].reshape((28,28)), cmap='gray')
		plt.axis('off')

def sign(x):
	if x >= 0:
		return 1
	return -1

class PreceptronModel:
	'''
	Class for the PreceptronModel
	'''
     def __init__(self, dim=784):
         self.weights = np.zeros(dim)
         self.t = 0
         self.mistakes = 0
         self.mistakes_seq = []

     def predict(self, x):
     	'''
     	performs prediction using Class model parameters
     	'''
     	return sign(np.dot(self.weights, x))

     def update(self, x, label):
     	'''
     	Updates the class parameters (online)
     	'''
     	self.t += 1
     	pred = self.predict(x)
     	if (pred == -1) and (label == 1):
     		self.weights += x
     		self.mistakes += 1
     	elif (pred == 1) and (label == -1):
     		self.weights += -x
     		self.mistakes += 1
     	self.mistakes_seq.append(self.mistakes)
     
     def batchUpdate(self, xs, labels, m=1):
     	'''
     	Uses the online update to batch update (m-fold times)
     	'''
     	for i in range(m):
	     	for (x, l) in zip(xs, labels):
	     		self.update(x, l)

     def batchPredict(self, xs):
     	'''
     	Uses the online prediction to predict on a batch
     	'''
     	preds = []
     	for x in xs:
     		preds.append(self.predict(x))
     	return preds

     def errRate(xs, labels):
     	'''
     	Finds error rate on batch prediction, given true labels
     	'''
     	preds = self.batchPredict(xs)
     	return (preds == labels).mean()
