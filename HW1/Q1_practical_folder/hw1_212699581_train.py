import torch
from NeuralNetwork import NeuralNetwork as NN
from torchvision import transforms, datasets as ds
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle


transform = transforms.Compose([transforms.ToTensor(),
								transforms.Lambda(lambda x: torch.flatten(x))])

train_set = ds.MNIST(root='../data/',
					 train=True,
					 transform=transform,
					 download=True)

test_set = ds.MNIST(root='../data/',
					 train=False,
					 transform=transform,
					 download=True)

test_loader = DataLoader(dataset=test_set,
						  batch_size=len(test_set),
						  shuffle=False, )

test_loss = []
test_accuracy = []

nn = NN()
for epoch in range(50):
	nn.train(train_set, batch_size=100)

	x, y = next(iter(test_loader))
	o = nn.forward(x)
	Y = nn.one_hot(y)
	test_loss.append(nn.loss(Y, o))
	test_accuracy.append(nn.accuracy(y, nn.conv_prob_pred(o)))

	print(f"epoch: {epoch + 1}"
		  f"\tepoch train loss: {nn.avg_train_loss[-1]:.2f}"
		  f"\tepoch train accuracy: {nn.avg_train_accuracy[-1]:.2f}"
		  f"\tepoch test loss: {test_loss[-1]:.2f}"
		  f"\tepoch test accuracy: {test_accuracy[-1]:.2f}\n")
	if test_accuracy[-1] >= 0.8:
		break

with open('model.pkl', 'wb') as pickle_file:
	pickle.dump(nn, pickle_file)

plt.plot(range(len(nn.train_loss)), nn.train_loss, label="train loss")
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title("Networks Loss with each batch")
plt.legend()
plt.show()

plt.plot(range(len(nn.train_accuracy)), nn.train_accuracy, label="train accuracy")
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.title("Networks Accuracy with each batch")
plt.legend()
plt.show()

plt.plot(range(len(nn.avg_train_loss)), nn.avg_train_loss, label="train loss")
plt.plot(range(len(test_loss)), test_loss, label="test loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Networks Loss with each Epoch")
plt.legend()
plt.show()

plt.plot(range(len(nn.avg_train_accuracy)), nn.avg_train_accuracy, label="train accuracy")
plt.plot(range(len(test_accuracy)), test_accuracy, label="test accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Networks Accuracy with each Epoch")
plt.legend()
plt.show()




