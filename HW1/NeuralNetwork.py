import math
import torch
from torch.utils.data import DataLoader


class Activations:
	@staticmethod
	def ReLU(z):
		return z.apply_(lambda x: max(0, x))

	@staticmethod
	def ReLU_prime(z):
		return z.apply_(lambda x: int(x > 0))

	@staticmethod
	def Sigmoid(z):
		return 1 / (1 + math.e ** (-z))

	@staticmethod
	def sigmoid_prime_(z):
		p = Activations.Sigmoid(z)
		return p * (1 - p)

	@staticmethod
	def Sigmoid_Prime(z):
		return z.apply_(Activations.sigmoid_prime_)

	@staticmethod
	def RReLU(z, a=0.2):
		return z.apply_(lambda x: (a * x) if x < 0 else x)

	@staticmethod
	def tanh(z):
		return 2 * Activations.Sigmoid(2 * z) - 1

	@staticmethod
	def tanh_prime(z):
		return 1 - z * z

	@staticmethod
	def softmax(z):
		if len(z.size()) == 1:
			z = z - torch.max(z)
			A = sum([math.e ** elem for elem in z])
			K = torch.tensor([((math.e ** elem) / A) for elem in z])
			z = K
			return z

		for i in range(z.size(0)):
			z[i] = z[i] - torch.max(z[i])
			A = sum([math.e ** elem for elem in z[i]])
			K = torch.tensor([((math.e ** elem) / A) for elem in z[i]])
			z[i, :] = K
		return z

	@staticmethod
	def softmax_prime(z):
		I = torch.eye(z.size(0))
		s = Activations.softmax(z)
		return s * (I - s.reshape(-1, 1))


class Loss:
	@staticmethod
	def ZeroOne(Y, Y_hat):
		return sum([e == e_hat for e, e_hat in zip(Y, Y_hat)]) / len(Y)

	@staticmethod
	def MSE(Y, Y_hat):
		batch_size = Y.size(0)
		return sum([torch.norm(y - y_hat) ** 2 for y, y_hat in zip(Y, Y_hat)]) / batch_size

	@staticmethod
	def CE(Y, Y_hat):
		batch_size = Y.size(0)
		return -sum([(y * (y_hat + (10 ** -10)).apply_(lambda x: math.log(x))).sum() for y, y_hat in zip(Y, Y_hat)]) / batch_size


class NeuralNetwork:
	def __init__(self, size: tuple = (28*28, 100, 100, 10), activations: tuple = (Activations.tanh,
																	 Activations.ReLU,
																	 Activations.softmax),
				 loss=Loss.MSE):

		self.Weights = []
		self.b = []
		self.loss = loss
		self.train_loss = []
		self.train_accuracy = []

		self.avg_train_loss = []
		self.avg_train_accuracy = []

		self.test_loss = []
		self.size = size

		for i in range(len(size) - 1):
			self.Weights.append(torch.randn(size[i], size[i + 1]))
			self.b.append(torch.randn(size[i + 1]))

		if len(self.Weights) != len(activations):
			self.activations = [Activations.Sigmoid] * len(size)
			print("Defaulted the activations into Sigmoids")
		else:
			self.activations = activations

	def accuracy(self, Y, Y_hat):
		return sum(Y == Y_hat) / Y.size(0)

	def train(self, train_set, batch_size=100, shuffle=True, batches=None):
		train_loader = DataLoader(dataset=train_set,
								  batch_size=batch_size,
								  shuffle=shuffle, )

		self.m = []
		self.v = []
		for i in range(len(self.size) - 1):
			self.m.append([0])
			self.v.append([0])
		self.beta1 = .9
		self.beta2 = .999
		self.eps = 10 ** -8

		l = 0
		a = 0
		b = 0
		for i, (x, y) in enumerate(train_loader):
			if (batches is not None) and (i >= batches):
				break

			Y = self.one_hot(y)
			o = self.forward(x)
			self.backward(x, Y, o)

			self.train_loss.append(self.loss(Y, o))
			self.train_accuracy.append(self.accuracy(y, self.conv_prob_pred(o)))

			a += self.train_accuracy[-1]
			l += self.train_loss[-1]
			b = i + 1
			if (i % 100) == 0:
				print(f"batch: {int((i / 100) + 1)}\tMSE loss: {self.train_loss[-1]:.2f}\tAccuracy: {self.train_accuracy[-1]:.2f}")

		self.avg_train_loss.append(l / b)
		self.avg_train_accuracy.append(a / b)

	def one_hot(self, y):
		Y = torch.zeros(y.size(0), self.size[-1])

		for j, e in enumerate(y):
			Y[j, e.item() - 1] = 1
		return Y

	def forward(self, X):
		self.layers_output = []
		Z = X
		for j, e in enumerate(self.Weights):
			Z = self.activations[j](torch.matmul(Z, e) + self.b[j])
			self.layers_output.append(Z.clone())

		return Z.clone()

	def conv_prob_pred(self, Y_hat):
		return torch.tensor([(torch.argmax(e).item() + 1) for e in Y_hat])

	def predict(self, X):
		return torch.tensor([(torch.argmax(e).item() + 1) for e in self.forward(X)])

	def backward(self, X, Y, Y_hat, lr=.1):
		# grad wrt W3, b3
		batch = Y.size(0)

		H2 = self.layers_output[-2]
		# Z = (Y_hat - Y)

		dl_dx3 = (1/batch) * (Y_hat - Y)  # * Y_hat * (1 - Y_hat)

		# for i, row in enumerate(dl_dx3):
		# 	dl_dx3[i] = row @ (row.reshape(1, -1) * (I - row.reshape(-1, 1)))

		gradW3 = H2.T @ dl_dx3
		self.m[0].append(self.beta1 * self.m[0][-1] + (1 - self.beta1) * gradW3)
		self.v[0].append(self.beta2 * self.v[0][-1] + (1 - self.beta2) * (torch.norm(gradW3)))

		m_hat = self.m[0][-1] / (1 - (self.beta1 ** len(self.m[0])))
		v_hat = self.v[0][-1] / (1 - (self.beta2 ** len(self.v[0])))

		self.Weights[-1] = self.Weights[-1] - (lr / (math.sqrt(v_hat) + self.eps)) * m_hat
		self.b[-1] = self.b[-1] - lr * (dl_dx3.T @ torch.ones(batch))

		# grad wrt W2, b2
		H1 = self.layers_output[-3]
		p = Activations.ReLU_prime(self.layers_output[1])
		dl_dx2 = (dl_dx3 @ self.Weights[2].T) * p

		gradW2 = H1.T @ dl_dx2
		self.m[1].append(self.beta1 * self.m[1][-1] + (1 - self.beta1) * gradW2)
		self.v[1].append(self.beta2 * self.v[1][-1] + (1 - self.beta2) * (torch.norm(gradW2)))

		m_hat = self.m[1][-1] / (1 - (self.beta1 ** len(self.m[1])))
		v_hat = self.v[1][-1] / (1 - (self.beta2 ** len(self.v[1])))

		self.Weights[1] = self.Weights[1] - (lr / (math.sqrt(v_hat) + self.eps)) * m_hat
		self.b[1] = self.b[1] - lr * (dl_dx2.T @ torch.ones(batch))

		# dl_dxi1 = dl_dx3
		#
		# for i in range(len(self.layers_output) - 3, -1, -1):
		# 	Hi = self.layers_output[i]
		# 	p = Activations.Sigmoid_Prime(self.layers_output[i + 1])
		# 	dl_dxi = (1 / batch) * (dl_dxi1 @ self.Weights[i + 2].T) * p
		# 	self.Weights[i + 1] = self.Weights[i + 1] - lr * Hi.T @ dl_dxi
		# 	self.b[i + 1] = self.b[i + 1] - lr * (dl_dxi.T @ torch.ones(batch))
		# 	dl_dxi1 = dl_dxi.clone()
		#
		# dl_dx2 = dl_dxi1

		# grad wrt W1, b1

		p = Activations.tanh_prime(self.layers_output[0])
		dl_dx1 = (dl_dx2 @ self.Weights[1].T) * p

		gradW1 = X.T @ dl_dx1
		self.m[2].append(self.beta1 * self.m[2][-1] + (1 - self.beta1) * gradW1)
		self.v[2].append(self.beta2 * self.v[2][-1] + (1 - self.beta2) * (torch.norm(gradW1)))

		m_hat = self.m[2][-1] / (1 - (self.beta1 ** len(self.m[2])))
		v_hat = self.v[2][-1] / (1 - (self.beta2 ** len(self.v[2])))

		self.Weights[0] = self.Weights[0] - (lr / (math.sqrt(v_hat) + self.eps)) * m_hat
		self.b[0] = self.b[0] - lr * (dl_dx1.T @ torch.ones(batch))


