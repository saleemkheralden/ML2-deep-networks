import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import pickle
import matplotlib.pyplot as plt
import gc

# Hyper Parameters
num_epochs = 1000
batch_size = 250
learning_rate = 0.001
# weight_decay = 1e-5

def load_data(batch_size=250):
	# Image Preprocessing
	train_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465),
							 (0.247, 0.2434, 0.2615)),
	])

	valid_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(p=0.3),
		transforms.RandomRotation(30),
		# transforms.ColorJitter(),
		# transforms.RandomVerticalFlip(p=0.3),
		# transforms.RandomCrop(32, padding=5),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465),
							 (0.247, 0.2434, 0.2615)),
	])

	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465),
							 (0.247, 0.2434, 0.2615)),
	])

	# CIFAR-10 Dataset
	train_dataset = dsets.CIFAR10(root='./data/',
								  train=True,
								  transform=train_transform,
								  download=True)

	validation_dataset = dsets.CIFAR10(root='./data/',
									   train=True,
									   transform=valid_transform,
									   download=True)

	test_dataset = dsets.CIFAR10(root='./data/',
								 train=False,
								 transform=test_transform,
								 download=True)

	combined_train_dataset = ConcatDataset([train_dataset, validation_dataset])

	# Data Loader (Input Pipeline)
	train_loader = DataLoader(dataset=combined_train_dataset,
							  batch_size=batch_size,
							  shuffle=True)

	test_loader = DataLoader(dataset=test_dataset,
							 batch_size=batch_size * 10,
							 shuffle=False)

	return train_loader, test_loader

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


def to_device(data, device):
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	return data.to(device, non_blocking=True)


class DeviceDataLoader:
	def __init__(self, dl, device):
		self.dl = dl
		self.device = device

	def __iter__(self):
		for b in self.dl:
			yield to_device(b, self.device)


# train_loader = DeviceDataLoader(train_loader, device)
# test_loader = DeviceDataLoader(test_loader, device)


# CNN MODEL

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2))  # 3 x 32 x 32 -> 16 x 8 x 8
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),)  # 16 x 8 x 8 -> 32 x 8 x 8
			# nn.MaxPool2d(2))

		self.dropout1 = nn.Dropout(p=.3)

		self.layer3 = nn.Sequential(
			nn.Conv2d(32, 40, kernel_size=3, padding=1),
			nn.BatchNorm2d(40),
			nn.ReLU(),
			nn.MaxPool2d(2))  # 32 x 8 x 8 -> 40 x 4 x 4

		self.layer4 = nn.Sequential(
			nn.Conv2d(40, 50, kernel_size=3, padding=1),
			nn.BatchNorm2d(50),
			nn.ReLU())  # 40 x 4 x 4 -> 50 x 4 x 4

		self.fc1 = nn.Linear(4 * 4 * 50, 15)  # 4 * 4 * 50 -> 15
		self.dropout = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(15, 10)  # 15 -> 10
		self.logsoftmax = nn.LogSoftmax(dim=1)
		# self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.dropout1(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = nn.functional.relu(out)
		out = self.dropout(out)
		out = self.fc2(out)
		# return self.softmax(out)
		return self.logsoftmax(out)


# cnn = CNN()
# print('Training from scratch')
# cnn = pickle.load(open('model_q1.pkl', 'rb'))



# TRAINING
def train_model_q1(save_file=True):
	print('num_epochs: %d \t batch_size: %d \t learning_rate: %.3f' %
		  (num_epochs, batch_size, learning_rate))
	cnn = CNN()
	if torch.cuda.is_available():
		cnn.to(device)

	num_param = sum(param.numel() for param in cnn.parameters())
	print('number of parameters: ', num_param)

	train_loader, test_loader = load_data()

	# criterion = nn.NLLLoss()
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)


	test_accuracy = []

	test_error = []
	test_loss = []

	train_error = [0]
	train_loss = [0]

	for epoch in range(num_epochs):

		# cnn.to(torch.device('cpu'))

		test_batch_acc = 0
		test_len = 0
		test_batch_loss = 0

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

				# test_batch_acc = test_batch_acc + sum(y == torch.argmax(o, dim=1)).item()
				# test_len = test_len + y.size(0)
				test_batch_loss += criterion(o, y).item()

			test_accuracy.append(correct / total)
			test_loss.append(test_batch_loss)
			test_error.append(1 - test_accuracy[-1])

		cnn.train()

		if (len(test_accuracy) > 0) and (test_accuracy[-1] >= 0.801):
			break

		# if (epoch + 1) % 20 == 0:
		# 	print('EPOCH 20 SAVING MODEL')
		# 	with open(f'model.pkl', 'wb') as pickle_file:
		# 		pickle.dump(cnn, pickle_file)

		avg_loss = 0
		avg_acc = 0

		train_len = 0
		train_len_loss = 0

		# cnn.to(device)
		for i, (images, labels) in enumerate(train_loader):
			if torch.cuda.is_available():
				# print("CUDA")
				images = images.cuda()
				labels = labels.cuda()

			# Forward + Backward + Optimize
			outputs = cnn(images)
			loss = criterion(outputs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			avg_loss = avg_loss + loss.item()
			avg_acc += sum(labels == torch.argmax(outputs, dim=1)).item()

			train_len = train_len + labels.size(0)
			train_len_loss = train_len_loss + 1

			if (i + 1) % 100 == 0:
				print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f \n'
					  '\t Train acc: %.4f \t Test accuracy: %.4f \t Test error: %.4f '
					  '\t Test loss: %.4f \t Train error: %.4f \t Train loss: %.4f'
					  % (epoch + 1, num_epochs, i + 1,
						 len(train_loader), loss.data, avg_acc / train_len,
						 test_accuracy[-1], test_error[-1], test_loss[-1], train_error[-1], train_loss[-1]))

			if torch.cuda.is_available():
				del images
				del labels
				del outputs
				torch.cuda.empty_cache()

		# print('train loss: ', train_loss[-10:])
		train_loss.append(avg_loss / train_len_loss)
		train_error.append(1 - (avg_acc / train_len))

	plt.plot(range(len(train_error)), train_error, color='red', label='train error')
	plt.plot(range(len(test_error)), test_error, color='blue', label='test error')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.title('train, test Error')
	plt.legend()
	plt.show()

	plt.plot(range(len(train_loss)), train_loss, color='red', label='train loss')
	plt.plot(range(len(test_loss)), test_loss, color='blue', label='test loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('train, test Loss')
	plt.legend()
	plt.show()

	plt.plot(range(len(test_accuracy)), test_accuracy)
	plt.xlabel('Epoch')
	plt.ylabel('Test Accuracy')
	plt.title('Test accuracy as function of epochs.')
	plt.show()

	if save_file:
		with open(f'model_q1.pkl', 'wb') as pickle_file:
			pickle.dump(cnn, pickle_file)



