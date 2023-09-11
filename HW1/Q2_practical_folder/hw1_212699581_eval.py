import torch
from NeuralNetwork import NeuralNetwork
from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import pickle


def evaluate_hw1():
	p = 0.5

	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Lambda(lambda x: torch.flatten(x))])

	test_set = ds.MNIST(root='../data/',
						train=False,
						transform=transform,
						download=True)

	batch_size = len(test_set)
	ber_y = torch.bernoulli(torch.tensor([p] * batch_size))
	test_set.targets[:batch_size] = ber_y

	test_loader = DataLoader(dataset=test_set,
							 batch_size=batch_size,
							 shuffle=False, )

	model = pickle.load(open('model.pkl', 'rb'))

	x, y = next(iter(test_loader))
	return model.accuracy(y, model.predict(x)).item()


# print(f'{evaluate_hw1():.3f}')



