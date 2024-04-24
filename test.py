#!/usr/bin/env python3
import pykitti
from config import *
from data_utils import KittiOdomDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import DeepVOLite,DeepVO



def se3_to_position(mat):
    t = mat[:, -1][:-1] # last 3 element of the last column
    return t
def se3_to_rot(mat):
    return mat[:3, :3]

odom = pykitti.odometry(DATA_PATH, "00")

train_dataset =  KittiOdomDataset("03", DATA_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
# optimizer = optim.Adagrad(model.parameters(), lr=0.001)

def conv(in_channel, out_channel, kernel_size, stride, padding, dropout):
	"""
	bn : Batch normalization
	"""

	sequential_layer = nn.Sequential(
		nn.Conv2d(
			in_channel,
			out_channel,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			bias=False
			),

		nn.BatchNorm2d(out_channel),
		nn.ReLU(inplace=True),
		nn.MaxPool2d(kernel_size= (2,2), stride = (2,2)),
		nn.Dropout(dropout),

		)

	return sequential_layer



class CNN(nn.Module):
	def __init__(self, num_classes=6):
		super(CNN, self).__init__()
		# self.conv1 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(3,3), stride=(1,1) ,padding=(1,1) )
		# self.pool = nn.MaxPool2d(kernel_size= (2,2), stride = (2,2))
		# self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1) ,padding=(1,1) )

		self.conv1 = conv(6, 32, 7, 2, 3, 0.2)
		self.conv2 = conv(32, 64, 5, 2, 2, 0.2)
		self.conv3 = conv(32, 16, 5, 2, 2, 0.2)

		# self.conv3 = conv(64, 16, 5, 2, 2, 0.2)
		# 64, 94, 311
		self.rnn = nn.LSTM(
			input_size = 64*94*311, hidden_size=100, num_layers=2, batch_first=True
			)
		self.lstm_dropout = nn.Dropout(0.5)
		self.fc = nn.Linear(100, 6)

		# initialization (LSTM n stuff)
		# print(m for m in self.modules())
		for m in self.modules():
			if isinstance(m, nn.LSTM):
				# print(m)
				# print(dir(m))

				kaiming_normal_(m.weight_ih_l0) # input to hidden layer for first 10 layers of lstm
				kaiming_normal_(m.weight_hh_l0) # hidden to hidden layer for 10 layers
				# Initialize biases
				m.bias_ih_l0.data.zero_()
				m.bias_hh_l0.data.zero_()
				# Set specific values for some biases
				n = m.bias_hh_l0.size(0)
				start, end = n // 4, n // 2
				m.bias_hh_l0.data[start:end].fill_(1.0)

				kaiming_normal_(m.weight_ih_l1)
				kaiming_normal_(m.weight_hh_l1)
				m.bias_ih_l1.data.zero_()
				m.bias_hh_l1.data.zero_()
				n = m.bias_hh_l1.size(0)
				start, end = n // 4, n // 2
				m.bias_hh_l1.data[start:end].fill_(1.0)

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self,x):
		batch_size = x.size(0)
		seq_len = x.size(1) # 
		print("Seq Len <model.py>: ", seq_len)
		# print("x size :",sys.getsizeof(x))
		x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		
		print("before flat : ", x.shape)
		x = x.view(batch_size, seq_len, -1) # Flattening

		print("Flattened",x.shape)
		# x,_= self.rnn(x)
		# x = self.lstm_dropout(x)
		x = self.fc(x)
		
		print("Done forward")

		return x


# seq_arr=[]
# pos_arr=[]
# ang_arr=[]
# for i, (seq, pos, ang) in enumerate(train_dataset):
# 	print(i)
	

# 	seq_arr.append(seq)
# 	pos_arr.append(pos)
# 	ang_arr.append(ang)

	
# 	if (i+1)%6==0:

# 		seq_arr = np.array(seq_arr)
# 		pos_arr = np.array(pos_arr)
# 		ang_arr = np.array(ang_arr)

# 		seq_arr = torch.from_numpy(seq_arr)
# 		pos_arr = torch.from_numpy(pos_arr)
# 		ang_arr = torch.from_numpy(ang_arr)


# 		print(seq_arr.shape)
# 		print(pos_arr.shape)
# 		print(ang_arr.shape)	

# 		# yield seq_arr, pos_arr, ang_arr
# 		seq_arr=[]
# 		pos_arr=[]
# 		ang_arr=[]

# print(train_loader)

model = DeepVOLite()
model = model.to(DEVICE)
model.load_state_dict(torch.load("./trained_models/1.pth"))
model.eval()
