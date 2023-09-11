import pickle

import torch

from train_model_q1 import *
from train_model_q1 import CNN


def evaluate_model_q1():
	import __main__
	setattr(__main__, "CNN", CNN)
	cnn = pickle.load(open('model_q1.pkl', 'rb'))

	# the load_state_dict returns error 'invalid magic number...'
	# cnn = CNN()
	# cnn.load_state_dict(torch.load('model_q1.pkl', map_location=lambda storage, loc: storage))

	_, test_loader = load_data()

	cnn.eval()
	with torch.no_grad():
		correct = 0
		total = 0

		for x, y in test_loader:
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			o = cnn(x)
			_, predicted = torch.max(o.data, 1)
			total += y.size(0)
			correct += (predicted == y).sum()

		return 1 - (correct / total)


