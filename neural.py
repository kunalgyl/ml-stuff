import numpy as np
import matplotlib.pyplot as plt

def logistic(x):
    return 1 / (1 + np.exp(-x))

def der_logistic(x):
	return sigmoid(x)(1-sigmoid(x))

def der_logistic_h(x):
	return x*(1-x)

def read_data():
	X = np.loadtxt('TrainDigitX.csv.gz', delimiter=',')
	Y = np.loadtxt('TrainDigitX.csv.gz').astype(int)
	Y_oh = np.zeros((Y.shape[0],10))
	Y_oh[np.arange(Y.shape[0]), Y] = 1
	return X,Y_oh

def read_test_data():
	X = np.loadtxt('TestDigitX.csv.gz', delimiter=',')
	Y = np.loadtxt('TestDigitY.csv.gz').astype(int)
	Y_oh = np.zeros((Y.shape[0],10))
	Y_oh[np.arange(Y.shape[0]), Y] = 1
	return X,Y_oh

def plot_improvement(X,Y):
	data = list(zip(Y,X))
	data_train = data[:40000]
	data_test = data[40000:]
	nn = NeuralNetwork(784,[50],10)
	errs = []
	for i in range(20):
		print("{}".format(i))
		nn.train_epoch(data_train)
		errs.append(nn.err_examples(data_test))
	plt.plot(range(20), errs)
	plt.show()

hidden_neurons = [10,20,30,50,100,200]
learning_rates = [0.1,0.01,0.001]

def gen_nn():
	(X,Y) = read_data()
	data = list(zip(Y,X))
	data_test = list(zip(Y_test,X_test))
	nn = NeuralNetwork(input=784,hiddens=[50],output=10,learning_rate=0.01,l2=0.0001,decay=0.1)
	nn.train_n_epochs(data,20)
	return nn

def test_2_preds():
	nn = gen_nn()
	X = np.loadtxt('TestDigitX2.csv.gz', delimiter=',')
	pred = np.array(nn.predict_set(X))
	pred = pred.argmax(axis=1)
	np.savetxt('TestDigitY2.csv.gz',ps) 
	return pred

def perf_test(X,Y):
	data = list(zip(Y,X))
	data_train = data[:40000]
	data_test = data[40000:]
	d = {}
	for lr in learning_rates:
		d[lr] = []
		for N_h in hidden_neurons:
			print("{},{}".format(lr,N_h))
			nn = NeuralNetwork(input=784,hiddens=[N_h],output=10,learning_rate=lr)
			nn.train_n_epochs(data_train,10)
			d[lr].append(nn.err_examples(data_test))
	return d

def test_decay(X,Y,decays=[0]):
	data = list(zip(Y,X))
	data_train = data[:40000]
	data_test = data[40000:]
	err = []
	for decay in decays:
		nn = NeuralNetwork(input=784,hiddens=[50],output=10,learning_rate=0.01,l2=0.0001,decay=decay)
		nn.train_n_epochs(data_train,10)
		err.append(nn.err_examples(data_test))
	return err

def hidden_layer_test(X,Y,n=5,l2s=[0]):
	data = list(zip(Y,X))
	data_train = data[:40000]
	data_test = data[40000:]
	d = {}
	for i in range(1,n+1):
		d[i] = []
		for l2 in l2s:
			print("{},{}".format(i,l2))
			nn = NeuralNetwork(input=784,hiddens=[50]*i,output=10,learning_rate=0.01,l2=l2)
			nn.train_n_epochs(data_train,10)
			err = nn.err_examples(data_test)
			print("{}".format(err))
			d[i].append(err)
	return d

def final_nn(X,Y):
	data = list(zip(Y,X))
	nn = NeuralNetwork(input=784,hiddens=[50],output=10,learning_rate=0.01,l2=0.0001,decay=0.1)
	nn.train_n_epochs(data_train,10)

class NeuralNetwork(object):
	def __init__(self,input,hiddens,output,learning_rate=0.01,decay=0, l2=0):
		"""
		input is the dimension of input, not including the bias will be added to it (Int)
		hiddens will be a non-empty list of length number of hidden layers, with the ith entry
			signifying the number of hidden neurons in that layer (it is quite likely I won't
			experiment with using more than 1 hidden layer though)
		output is the dimension of the output
		"""
		self.input = input+1
		self.hiddens = hiddens
		self.output = output

		self.dims = [self.input] + self.hiddens + [self.output]
		self.activations = np.array([np.ones(d) for d in self.dims]) #as we want bias to always be 1, we never change it
		self.changes = np.array([np.zeros((first,second)) for (first,second) in zip(self.dims, self.dims[1:])])
		self.weights = np.array([np.random.randn(first,second) for (first,second) in zip(self.dims, self.dims[1:])])

		self.initial_rate = learning_rate
		self.learning_rate = learning_rate
		self.decay = 0
		self.t = 0
		self.l2 = l2

	def feed_forward(self, x):
		assert len(x) == self.input - 1
		self.activations[0][:self.input-1] = x
		for j in range(len(self.dims)-1):
			s = np.dot(self.weights[j].T, self.activations[j])
			self.activations[j+1] = logistic(s)
		return self.activations[-1]

	def back_propogate(self, y):
		assert len(y) == self.output
		delta = -1 * der_logistic_h(self.activations[-1]) * (y - self.activations[-1])
		for j in reversed(range(len(self.dims)-1)):
			err = np.dot(self.weights[j],delta)
			change = delta * np.reshape(self.activations[j], (self.activations[j].shape[0],1))
			self.weights[j] -= (self.learning_rate * (change + self.weights[j]*self.l2)) + self.changes[j]
			self.changes[j] = change
			delta = der_logistic_h(self.activations[j]) * err
		return np.sum((self.activations[-1] - y)**2)

	def train_epoch(self, examples):
		err = 0
		for example in examples:
			self.learning_rate = self.initial_rate / (self.t**self.decay)
			(y,x) = example
			self.feed_forward(x)
			err += self.back_propogate(y)
			self.t += 1
		return err

	def predict(self, x):
		return self.feed_forward(x)

	def predict_set(self, X):
		preds = []
		for x in X:
			preds.append(self.predict(x))
		return preds

	def err_examples(self, examples):
		Y = [ex[0] for ex in examples]
		X = [ex[1] for ex in examples]
		return self.err_set(X,Y)

	def err_set(self, X, Y):
		Y_hat = self.predict_set(X)
		Y_class = np.argmax(Y,axis=1)
		Y_hat_class = np.argmax(Y_hat,axis=1)
		return np.mean(Y_class == Y_hat_class)

	def train_n_epochs(self,examples,n=1):
		for i in range(n):
			print("{}".format(self.train_epoch(examples)))
			#self.train_epoch(examples)


