#import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn import preprocessing
from sklearn.utils import class_weight
import os
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #1st conv block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.dropout_conv = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)

        #2nd conv block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        #3rd conv block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)

        #4th conv block
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(32)

        #5th conv block
        self.conv5 = nn.Conv2d(256, 128, 3, padding=1)

        #6th conv block
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1)

        #7th conv block
        self.conv7 = nn.Conv2d(64, 32, 3, padding=1)

        #dropout and fully connected layer
        self.dropout_fc = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4960, 21)
        

    #forward propogation
    def forward(self, x):
    	#1st conv block
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #2nd conv block
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)

        #3rd conv block
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = self.pool(x)

        #4th conv block
        x = F.relu(self.conv4(x))
        x = self.dropout_conv(x)

        #5th conv block
        x = F.relu(self.conv5(x))
        x = self.dropout_conv(x)

        #6th conv block
        x = F.relu(self.conv6(x))
        x = self.dropout_conv(x)

        #7th conv block
        x = F.relu(self.conv7(x))
        x = self.dropout_conv(x)

        #dropout and fully connected layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout_fc(x)
        x = self.fc1(x)
        return x


net = Net()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('param count:', count_parameters(net))

#set device to gpu if one exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
	net = nn.DataParallel(net)
net.to(device)

#load data
X_train = np.load('X_train2.npy')
y_train = np.load('y_train2.npy')
print(X_train.shape)

X_test = np.load('X_test2.npy')
y_test = np.load('y_test2.npy')

X_val = np.load('X_val2.npy')
y_val = np.load('y_val2.npy')

def scale_data3d(train, test, val=None):
    scaler = preprocessing.StandardScaler()
    train_dim0, train_dim1, train_dim2 = train.shape[0], train.shape[1], train.shape[2]
    test_dim0, test_dim1, test_dim2 = test.shape[0], test.shape[1], test.shape[2]
    
    train = np.reshape(train, newshape=(train_dim0 * train_dim1, train_dim2))
    test = np.reshape(test, newshape=(test_dim0 * test_dim1, test_dim2))
    
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    
    train = np.reshape(train, newshape=(train_dim0, train_dim1, train_dim2))
    test = np.reshape(test, newshape=(test_dim0, test_dim1, test_dim2))
    
    if (val is not None):
        val_dim0, val_dim1, val_dim2 = val.shape[0], val.shape[1], val.shape[2]
        val = np.reshape(val, newshape=(val_dim0 * val_dim1, val_dim2))
        val = scaler.transform(val)
        val = np.reshape(val, newshape=(val_dim0, val_dim1, val_dim2))
    
    return train, test, val

## uncomment next line to scaler data, it seems to make performance/convergence much worse in this case
# X_train, X_test, X_val = scale_data3d(X_train, X_test, X_val) 

# class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
# class_weights = class_weights / class_weights[1]
# class_weights = torch.from_numpy(class_weights)
# class_weights = class_weights.to(device).float()

#convert numpy arrays to tensors, encode labels
X_train = torch.from_numpy(X_train)
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = torch.from_numpy(y_train)

# print(np.unique(y_train))
# print(le.inverse_transform(np.unique(y_train)))
# print(class_weights)
# sys.exit()

X_test = torch.from_numpy(X_test)
y_test = le.transform(y_test)
y_test = torch.from_numpy(y_test)

X_val = torch.from_numpy(X_val)
y_val = le.transform(y_val)
y_val = torch.from_numpy(y_val)

print(X_train.shape)

#cross entropy loss w SGD optimizer
criterion = nn.CrossEntropyLoss()#weight = class_weights)
optimizer = optim.Adam(net.parameters(), lr=3e-4)

#create dataloaders for all 3 datasets
train_set = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_set, batch_size = 32, shuffle=True)

val_set = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_set, batch_size = 32)

test_set = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_set, batch_size = len(y_train))

y_pred = []

for epoch in range(100):
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

		#reshape inputs as (channels, batch size, 20, 126)
		#(20, 126) at end should be constant due to our preprocessing of the
		#MFCCs
		inputs = torch.reshape(inputs, (1, inputs.shape[0], 20, 126))
		inputs = torch.permute(inputs, (1, 0, 2, 3))

		#send data to gpus
		inputs = inputs.to(device)
		labels = labels.to(device)

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
		inputs = torch.reshape(inputs, (1, inputs.shape[0], 20, 126))
		inputs = torch.permute(inputs, (1, 0, 2, 3))

		#send data to gpus
		inputs = inputs.to(device)
		labels = labels.to(device)

		#predict outputs
		outputs = net(inputs)

		#calculate class with highest probability
		_, predicted = torch.max(outputs.data, 1)
		y_pred = predicted
		# y_pred.extend(predicted)

		#update total and correct counts
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
print(f'Total count: {total}, correct: {correct}')
y_test = y_test.cpu()
y_pred = np.array(y_pred.cpu())

cf = confusion_matrix(y_test, y_pred)
print(cf)

# plt.figure(figsize=(10,8), dpi=120)
# sns.heatmap(cf, cmap='coolwarm', annot=True, fmt='g')
# plt.savefig('no_cw_heatmap.png')

PATH = os.getcwd()
file_name = 'nn05.pt'
torch.save(net.state_dict(), os.path.join(PATH, file_name))