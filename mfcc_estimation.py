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
from models import CNNQuantized

n_mfcc = 20
mfcc_length = 126
sr = 16000
batch_size = 32

class MFCCNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(512, 200, 1, padding=0)
		self.conv2 = nn.Conv1d(200, 50, 1, padding=0)
		self.conv3 = nn.Conv1d(50, 50, 1, padding=0)
		self.conv4 = nn.Conv1d(50, 16, 1, padding=0)

	#forward propogation
	def forward(self, x):
		x = F.hardtanh(self.conv1(x))
		x = F.hardtanh(self.conv2(x))
		x = F.hardtanh(self.conv3(x))
		x = F.hardtanh(self.conv3(x))
		x = F.hardtanh(self.conv3(x))
		x = F.hardtanh(self.conv4(x))
		# print(x.shape)
		# print(len(x) / (n_mfcc * mfcc_length))
		# print(x.shape[1] / (n_mfcc * mfcc_length))
		# x = torch.reshape(x, (x.shape[1], n_mfcc, mfcc_length))
		# x = torch.reshape(x, ((len(x) / (n_mfcc * mfcc_length), n_mfcc, mfcc_length)))
		return x


net = MFCCNet()
conv_net = CNNQuantized()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('param count:', count_parameters(net))

#set device to gpu if one exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
	net = nn.DataParallel(net)
net.to(device)
conv_net.to(device)

#load data
X_train = np.load('train_audio.npy')
y_train = np.load('train_mfcc.npy')


X_test = np.load('test_audio.npy')
y_test = np.load('test_mfcc.npy')
test_label = np.load('test_label.npy')

X_val = np.load('val_audio.npy')
y_val = np.load('val_mfcc.npy')
val_label = np.load('val_label.npy')

def reshape_data(data):
	#data is array with shape (n, 16000)
	npad = ((0,0), (0, 384))
	new_data = np.pad(data, npad, mode='constant', constant_values=0)
	# new_data = np.reshape(new_data, (new_data.shape[0], 512, 64))
	as_strided = np.lib.stride_tricks.as_strided
	new_data = as_strided(new_data, (new_data.shape[0], 512, 64))
	return new_data

X_train = reshape_data(X_train)
X_test = reshape_data(X_test)
X_val = reshape_data(X_val)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

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
le.classes_ = np.load('encoder.npy')
# y_train = le.fit_transform(y_train)
y_train = torch.from_numpy(y_train)

# print(np.unique(y_train))
# print(le.inverse_transform(np.unique(y_train)))
# print(class_weights)
# sys.exit()

X_test = torch.from_numpy(X_test)
test_label = le.transform(test_label)
y_test = torch.from_numpy(y_test)

X_val = torch.from_numpy(X_val)
val_label = le.transform(val_label)
val_label = torch.from_numpy(val_label)
y_val = torch.from_numpy(y_val)

print('1')

def matrix_similarity_loss(output, target):
	loss = abs(torch.cdist(output, target, p=2.0))
	return loss

#cross entropy loss w SGD optimizer
# criterion = nn.MSELoss()#weight = class_weights)
# criterion = matrix_similarity_loss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

#create dataloaders for all 3 datasets
train_set = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_set, batch_size = 32, shuffle=True)

val_set = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_set, batch_size = 32, shuffle=False)

test_set = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_set, batch_size = len(y_test))

y_pred = []

checkpoint = torch.load('nn05_quantized.pt')
try:
	conv_net.load_state_dict(checkpoint)
except RuntimeError:
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in checkpoint.items():
		name = k[7:]
		new_state_dict[name] = v
	conv_net.load_state_dict(new_state_dict)
conv_net.eval()


for epoch in range(5):
	#set loss for each epoch to 0
	print(epoch)	
	running_loss = 0.0

	#loop through training set in batches
	for i, data in enumerate(train_dataloader):
		#get input data and labels
		inputs, labels = data

		#labels need to be LongTensor type for some reason
		labels = labels.type(torch.FloatTensor)

		#inputs need to be floats
		inputs = inputs.float()

		#reshape inputs as (channels, batch size, 20, 126)
		#(20, 126) at end should be constant due to our preprocessing of the
		#MFCCs
		print(inputs.shape)
		# inputs = torch.reshape(inputs, (1, inputs.shape[0], 16000))
		# inputs = torch.permute(inputs, (1, 0))

		#send data to gpus
		inputs = inputs.to(device)
		labels = labels.to(device)

		#zero the gradients in optimizer
		optimizer.zero_grad()

		#create predictions, calculate loss, backward propogation, then
		#update weights of model
		outputs = net(inputs)
		print(outputs.shape)
		# loss = matrix_similarity_loss(outputs, labels)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		#add loss
		running_loss += float(loss.item())

		# total = 0
		# outputs = torch.reshape(outputs, (1, outputs.shape[0], outputs.shape[1], outputs.shape[2]))
		# final_outputs = conv_net(outputs)

		# #calculate class with highest probability
		# _, predicted = torch.max(final_outputs.data, 1)
		# y_pred = predicted
		# # y_pred.extend(predicted)

		# #update total and correct counts
		# total += labels.size(0)
		# correct += (predicted == labels).sum().item()

		# print(f'Accuracy of the network on the {total} validation images: {100 * correct // total} %')
		# print(f'Total count: {total}, correct: {correct}')

		#print update every 32 batches
		if i % 32 == 31:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 32:.3f}')
			running_loss = 0.0

	running_vloss = 0.0
	total = 0
	correct = 0
	with torch.no_grad():
		for i, vdata in enumerate(val_dataloader):
			# print(i*32 + 31)
			vinputs, vlabels = vdata
			vinputs = vinputs.float()
			vlabels = vlabels.type(torch.FloatTensor)

			vinputs = vinputs.to(device)
			vlabels = vlabels.to(device)
			vinputs = torch.reshape(vinputs, (1, vinputs.shape[0], 16000))

			voutputs = net(vinputs)
			vloss = criterion(voutputs, vlabels)
			running_vloss += float(vloss)

			voutputs = torch.reshape(voutputs, (1, voutputs.shape[0], voutputs.shape[1], voutputs.shape[2]))
			# voutputs = torch.reshape(voutputs, (voutputs.shape[0], 1, voutputs.shape[1], voutputs.shape[2]))
			voutputs = torch.permute(voutputs, (1, 0, 2, 3))
			# print(voutputs.shape)

			voutputs_final = conv_net(voutputs)
			#calculate class with highest probability
			_, y_pred = torch.max(voutputs_final.data, 1)
			val_len = len(y_pred)
			vwords = val_label[i*32: i*32 + val_len]
			vwords = vwords.to(device)

			# print(y_pred)
			# print(vwords)
			# y_pred.extend(predicted)

			#update total and correct counts
			total += vlabels.size(0)
			correct += (y_pred == vwords).sum().item()

	print(f'Accuracy of the network on the {total} validation audio clips: {100 * correct // total} %')
	print(f'Total count: {total}, correct: {correct}')

	avg_vloss = running_vloss / (i + 1)
	print('Validation loss {}'.format(avg_vloss))

print('Finished training')

print('3')
del inputs, labels, vinputs, vlabels, loss, vloss

######Testing#######
correct = 0
total = 0 
test_loss = 0.0
#use no_grad as we are testing
with torch.no_grad():

	#loop through test data
	for data in test_dataloader:
		#gets inputs and labels
		inputs, labels = data 

		#convert labels to long tensors
		labels = labels.type(torch.FloatTensor)

		#convert inputs to float
		inputs = inputs.float()
		
		#reshape inputs as (channels, batch size, 16000)
		inputs = torch.reshape(inputs, (1, inputs.shape[0], 16000))
		# inputs = torch.permute(inputs, (1, 0, 2, 3))

		#send data to gpus
		inputs = inputs.to(device)
		labels = labels.to(device)

		#predict outputs
		outputs = net(inputs)

		outputs = torch.reshape(outputs, (1, outputs.shape[0], outputs.shape[1], outputs.shape[2]))
		# voutputs = torch.reshape(voutputs, (voutputs.shape[0], 1, voutputs.shape[1], voutputs.shape[2]))
		outputs = torch.permute(outputs, (1, 0, 2, 3))
		# print(outputs.shape)
			
		outputs_final = conv_net(outputs)
		# print(outputs.shape)
		#calculate class with highest probability
		_, predicted = torch.max(outputs_final.data, 1)
		# y_pred = predicted
		# y_pred.extend(predicted)

		#update total and correct counts
		total += labels.size(0)
		# loss = criterion(outputs, labels)
		# test_loss += loss.item()
		print(predicted)
		print(test_label)
		test_label = torch.from_numpy(test_label)
		test_label = test_label.to(device)
		correct += (predicted == test_label).sum().item()

# print(f'MSE of the network on the {total} test images: ')#{test_loss}')
print(f'Accuracy of the network on the {total} test audio clips: {100 * correct // total}%')
print(f'Total count: {total}, correct: {correct}')
# y_test = y_test.cpu()
# y_pred = np.array(y_pred.cpu())

# cf = confusion_matrix(y_test, y_pred)
# print(cf)
outputs = outputs.cpu()
np.save("outputs", outputs)
# plt.figure(figsize=(10,8), dpi=120)
# plt.hist(outputs.cpu().flatten())
# plt.savefig('mfcc_pred_hist.png')
# sns.heatmap(cf, cmap='coolwarm', annot=True, fmt='g')
# plt.savefig('no_cw_heatmap.png')

PATH = os.getcwd()
file_name = 'mfcc_estimation.pt'
# torch.save(net.state_dict(), os.path.join(PATH, file_name))