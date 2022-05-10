#import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn import preprocessing
import os

#define cnn class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(10240, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 21)

    #forward propogation
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

net = Net()
#move model to cuda here - for later version

#cross entropy loss w SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

#load data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

#shuffle data so it is randomly distributed
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X_train, y_train = unison_shuffled_copies(X_train, y_train)
X_val, y_val = unison_shuffled_copies(X_val, y_val)

print(X_train.shape)
print(y_train.shape)

print(X_val.shape)
print(y_val.shape)

print(X_test.shape)
print(y_test.shape)

#convert numpy arrays to tensors, encode labels
X_train = torch.from_numpy(X_train)
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = le.transform(y_test)
y_test = torch.from_numpy(y_test)

X_val = torch.from_numpy(X_val)
y_val = le.transform(y_val)
y_val = torch.from_numpy(y_val)

#create dataloaders for all 3 datasets
train_set = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_set, batch_size = 32)

val_set = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_set, batch_size = 32)

test_set = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_set, batch_size = 32)

#print cnn structure
print(net)

#start with 2 epochs for test
for epoch in range(2):

	#set loss for each epoch to 0
	running_loss = 0.0

	#loop through training set in batches
	for i, data in enumerate(train_dataloader):
		#get input data and labels
		inputs, labels = data

		#labels need to be LongTensor type for some reason
		labels = labels.type(torch.LongTensor)

		#inputs need to be floats
		inputs = inputs.float()
		
		#reshape inputs as (channels, batch size, 20, 32)
		#(20, 32) at end should be constant due to our preprocessing of the
		#MFCCs
		inputs = torch.reshape(inputs, (1, inputs.shape[0], 20, 32))
		inputs = torch.permute(inputs, (1, 0, 2, 3))

		#zero the gradients in optimizer
		optimizer.zero_grad()

		#create predictions, calculate loss, backward propogation, then
		#update weights of model
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		#add loss
		running_loss += loss.item()

		#print update every 32 batches
		if i % 32 == 31:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 32:.3f}')
			running_loss = 0.0

print('Finished training')


######Testing#######
correct = 0
total = 0 

#use no_grad as we are testing
with torch.no_grad():

	#loop through test data
	for data in test_dataloader:
		#gets inputs and labels
		inputs, labels = data 

		#convert labels to long tensors
		labels = labels.type(torch.LongTensor)

		#convert inputs to float
		inputs = inputs.float()
		
		#reshape inputs as (channels, batch size, 20, 32)
		inputs = torch.reshape(inputs, (1, inputs.shape[0], 20, 32))
		inputs = torch.permute(inputs, (1, 0, 2, 3))

		#predict outputs
		outputs = net(inputs)

		#calculate class with highest probability
		_, predicted = torch.max(outputs.data, 1)

		#update total and correct counts
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
print(f'Total count: {total}, correct: {correct} %')

PATH = os.getcwd()
torch.save(net.state_dict(), PATH)