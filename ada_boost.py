import matplotlib.pyplot as plt 
import numpy as np 
import random
import math
import pdb
from scipy.sparse import csr_matrix
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
import itertools
import os
import pdb
import time
import multiprocessing

# I use Pillow, as per suggestion from:
# http://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python
# http://stackoverflow.com/questions/20060096/installing-pil-with-pip
from PIL import Image

FEATURES = [(2,1), (1,2), (3,1), (1,3), (2,2)]

def load_images_sugared():
	poss = [ImageSugared(im, 1) for im in load_data('faces')]
	negs = [ImageSugared(im, -1) for im in load_data('background')]
	return poss, negs

def load_data(path):
	ims = []
	for f in os.listdir(path):
		if f.endswith('.jpg'):
			ims.append(np.array(Image.open(os.path.join(path, f)).convert('L')))
	return ims

class Cascade:
	def __init__(self, poss, negs, poss_val, negs_val ,threshold=0.3):
		self.positives = poss
		self.negatives = negs
		self.postivies_val = poss_val
		self.negatives_val = negs_val
		self.train_errs = []
		self.false_poss = []
		self.threshold = threshold
		self.strong_classifiers = []
		self.strong_classifiers_false_pos = []
		self.times = []

	def train_cascade(self, N=5):
		start = time.clock()
		self.add_strong_classifier()
		self.times.append(time.clock()-start)
		strong_classifiers_false_pos = self.cascade_class_rate()
		self.strong_classifiers_false_pos.append(strong_classifiers_false_pos)
		new_negatives = self.negatives
		for i in range(N-1):
			new_negatives = self.misclassified_negatives(new_negatives)
			start = time.clock()
			self.add_strong_classifier(self.positives, new_negatives)
			self.times.append(time.clock()-start)
			strong_classifiers_false_pos = self.cascade_class_rate()
			self.strong_classifiers_false_pos.append(strong_classifiers_false_pos)
			self.times.append()
		return

	def misclassified_negatives(self, negatives):
		misclass = []
		for img in negatives:
			if (self.predict(img) == 1):
				misclass.append(img)
		return misclass

	def add_strong_classifier(self, positives=None, negatives=None):
		if positives is None:
			positives = self.positives
		if negatives is None:
			negatives = self.negatives
		ada = AdaBoost(positives, negatives)
		ada.train()
		false_poss = []
		false_pos = ada.class_rate()
		false_poss.append(false_pos)
		n_iters = 1
		while(false_pos > self.threshold):
			print("added {} weak learner/s".format(n_iters))
			ada.train()
			false_pos = ada.class_rate()
			false_poss.append(false_pos)
			n_iters +=1
		self.strong_classifiers.append(ada)
		self.false_poss.append(false_poss)
		return ada

	def cascade_class_rate(self, type='false_pos'):
		preds = []
		if type == "false_pos":
			for im in self.negatives_val:
				preds.append(self.predict(im))
			return (np.array(preds) == 1).mean()
		elif type == "false_neg":
			for im in self.positives_val:
				preds.append(self.predict(im))
			return (np.array(preds) == -1).mean()

	def predict(self, img):
		for ada in self.strong_classifiers:
			pred = ada.predict(img)
			if (pred == -1):
				return -1
		return 1

class AdaBoost:
	def __init__(self, poss, negs, features=None, sz=64, stride=6):
		if features is None:
			features = []
			for f in FEATURES:
				for (w, h) in itertools.product(range(f[0], sz+1, stride*f[0]), range(f[1], sz+1, stride*f[1])):
					for (x, y) in itertools.product(range(1,sz+1-w, stride), range(1, sz+1-h, stride)):
						features.append((f, (x,y), w, h))
		self.features = features
		self.n_features = len(features)
		self.positives = poss
		self.negatives = negs
		self.n_pos = len(self.positives)
		self.alphas = []
		self.classifiers = []
		self.weights = None
		self.Theta = None

	def choose_opt_pt(self, lab, weights, f_val):
		t_p = weights[:self.n_pos].sum()
		t_m = weights[self.n_pos:].sum()
		idx = np.array(range(len(f_val)))
		wf = list(zip(f_val, weights, lab, idx))
		wf.sort(key=lambda tup: tup[0])
		s_p = 0
		s_m = 0
		min_acc = math.inf
		min_args = (0, 1)
		for i in range(len(f_val)):
			(f,w,t,idx) = wf[i]
			if t == 1:
				s_p += w
			else:
				s_m += w
			if (s_p+t_m-s_m) < min_acc:
				min_acc = (s_p+t_m-s_m)
				min_args = (i,1)
			if (s_m+t_p-s_p) < min_acc:
				min_acc = (s_m+t_p-s_p)
				min_args = (i,-1)
		(i_opt,p) = min_args
		try:
			small_val_idx = np.array(np.array(wf[:i_opt])[:,3], dtype=int)
		except:
			small_val_idx = np.array([])
		if i_opt == 0:
			return ((wf[0][0] - 0.1, p), min_acc, small_val_idx)
		else:
			return (((wf[i_opt][0] + wf[i_opt-1][0])/2.0, p), min_acc, small_val_idx)

	def train(self, T=1):
		positives = self.positives
		negatives = self.negatives
		if self.weights is None:
			weights = np.array(([1.0/(2*len(positives))] * len(positives)) + 
			([1.0/(2*len(negatives))] * len(negatives)))
		else:
			weights = self.weights
		images = positives + negatives
		f_vals = {}
		for i in range(self.n_features):
			print("working on {}".format(i))
			(f, ulc, w, h) = self.features[i]
			f_vals[i] = list(map(lambda x: x.compute_feature(f, ulc, w, h), images))
		classifiers = self.classifiers
		alphas = self.alphas
		lab = np.ones(len(weights))
		lab[self.n_pos:] = -1
		for t in range(T):
			weights = weights / weights.sum()
			min_err = math.inf
			min_opts = (self.features[0], (0,1))
			min_svi = []
			for i in range(self.n_features):
				(opt, err, svi) = self.choose_opt_pt(lab, weights, f_vals[i])
				if err < min_err:
					min_err = err
					min_opt = (self.features[i], opt)
					min_svi = svi
			preds = np.ones(len(weights))
			for idx in min_svi:
				preds[idx] = -1
			preds = preds * min_opt[1][1]
			e = np.array(preds != lab, dtype=int)
			classifiers.append(min_opt)
			beta = min_err / (1-min_err)
			weights = weights * beta**(1-e)
			alphas.append(-np.log(beta))
		self.alphas = alphas
		self.classifiers = classifiers
		self.weights = weights
		self.Theta = self.find_Theta()

	def find_Theta(self):
		min_val = math.inf
		for img in self.positives:
			val = self.predict_val(img)
			if val < min_val:
				min_val = val
		return min_val

	def classify(self, classifier, img):
		((f, ulc, w, h), (theta, p)) = classifier
		return img.compute_vote(f, ulc, w, h, p, theta)

	def predict_val(self, img):
		val = 0
		for (classifier, alpha) in zip(self.classifiers, self.alphas):
			val += alpha * self.classify(classifier, img)
		return val

	def predict(self, img):
		val = self.predict_val(img)
		return 1 if (val >= self.Theta) else -1

	def class_rate(self, type='false_pos'):
		preds = []
		if type == "false_pos":
			for im in self.negatives:
				preds.append(self.predict(im))
			return (np.array(preds) == 1).mean()
		elif type == "false_neg":
			for im in self.positives:
				preds.append(self.predict(im))
			return (np.array(preds) == -1).mean()

class ImageSugared:
	def __init__(self, img, label):
		self.img = img
		n = img.shape[0]
		assert (img.shape[1] == n) # images are square
		self.integral_img = np.zeros((n+1, n+1))
		self.integral_img[1:,1:] = img.cumsum(axis=0).cumsum(axis=1)
		self.label = label

	def compute_sum_rect(self, ulc, brc):
    	# rectangle with index for ulc and brc (x1,y1) and (x2,y2) consists
    	# of i(i,j) with x1<=i<x2, y1<=i<y2
		urc = (brc[0], ulc[1])
		blc = (ulc[0], brc[1])
		return self.integral_img[brc] + self.integral_img[ulc] - self.integral_img[urc] - self.integral_img[blc]

	def compute_feature(self, feature, ulc, width, height):
		(x1, y1) = ulc
		brc = (x1+ width, y1+height)
		if feature == (2,1): #Two horizontally next to each other
			sum1 = self.compute_sum_rect(ulc, (x1+width//2, y1+height))
			sum2 = self.compute_sum_rect((x1+width//2, y1), brc)
			return sum1 - sum2
		elif feature == (1,2): #Two vertically next to each other
			sum1 = self.compute_sum_rect(ulc, (x1+width, y1+height//2))
			sum2 = self.compute_sum_rect((x1, y1+height//2), brc)
			return sum1 - sum2 
		elif feature == (3,1): #Three horizontally next to each other
			sum1 = self.compute_sum_rect(ulc, (x1+width//3, y1+height))
			sum2 = self.compute_sum_rect((x1+width//3, y1), (x1+(2*width)//3, y1+height))
			sum3 = self.compute_sum_rect((x1+(2*width)//3, y1), brc)
			return sum1 - sum2 + sum3
		elif feature == (1,3): #Three vertically next to each other
			sum1 = self.compute_sum_rect(ulc, (x1+width, y1+height//3))
			sum2 = self.compute_sum_rect((x1, y1+height//3), (x1+width, y1+(2*height)//3))
			sum3 = self.compute_sum_rect((x1, y1+(2*height)//3), brc)
			return sum1 - sum2 + sum3
		elif feature == (2,2): #Four in a grid
			sum1 = self.compute_sum_rect(ulc, (x1+width//2, y1+height//2))
			sum2 = self.compute_sum_rect((x1+width//2, y1), (x1+width, y1+height//2))
			sum3 = self.compute_sum_rect((x1, y1+height//2), (x1+width//2, y1+height))
			sum4 = self.compute_sum_rect((x1+width//2, y1+height//2), brc)
			return sum1 - sum2 + sum3 - sum4
		else:
			raise Exception

	def compute_vote(self, feature, ulc, width, height, p, theta):
		feat = self.compute_feature(feature, ulc, width, height)
		return 1 if (p*feat >= p*theta) else -1

