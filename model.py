import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
# from config import *
import sys


"""
Stupid questions:
	- What exactly is Batch Normaliztion : https://www.youtube.com/watch?v=DtEq44FTPM4
	- Wtf is Sequential for NNs ? : 
	- kaiming_normal_ ? : Kaiming Normal initialization is a technique used to initialize the weights of neural networks in a way that helps with training. It sets the initial values of weights based on the number of input and output units, making sure that the variance remains the same across layers
	- How to build custom datasets ? : https://www.youtube.com/watch?v=ZoZHd0Zm3RY
	"""

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


class DeepVOLite(nn.Module):
	def __init__(self):
		super(DeepVOLite, self).__init__()

		self.conv1 = conv(6, 32, 7, 2, 3, 0.2)
		self.conv2 = conv(32, 32, 5, 2, 2, 0.2)
		self.conv3 = conv(32, 16, 5, 2, 2, 0.2)

		#64, 94, 311

		self.rnn = nn.LSTM(
			input_size = 16*6*19, hidden_size=100, num_layers=2, batch_first=True
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

	def forward(self, x):
		batch_size = x.size(0)
		seq_len = x.size(1) # 
		# print("Seq Len <model.py>: ", seq_len)
		# print("x size :",sys.getsizeof(x))
		x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		
		# print("before flat : ", x.shape)
		x = x.view(batch_size, seq_len, -1) # Flattening

		# print("Flattened",x.shape)
		x,_= self.rnn(x)
		x = self.lstm_dropout(x)
		x = self.fc(x)
		

		return x

	def get_loss(self, seq, pos, ang):
		pos = pos[:, 1:, :]
		ang = ang[:, 1:, :]
		y_hat = self.forward(seq)
		pos_loss = nn.functional.mse_loss(y_hat[:, :, 3:], pos)
		ang_loss = nn.functional.mse_loss(y_hat[:, :, :3], ang)
		return 100 * ang_loss + pos_loss

	def save_model(self, epoch, save_path):
		torch.save(
			{
			"epoch":epoch,
			"model_state_dict":self.state_dict(),
			"optimizer_state_dict": optimizer.state_dict()
			},
			f"{save_path}/{epoch}.pth"
		)

class DeepVO(nn.Module):
	def __init__(self):
		super(DeepVO, self).__init__()

		self.conv1 = conv(6, 64, 7, 2, 3, 0.2)
		self.conv2 = conv(64, 128, 5, 2, 2, 0.2)
		self.conv3 = conv(128, 256, 5, 2, 2, 0.2)
		self.conv3_1 = conv(256, 256, 3, 1, 1, 0.2)
		self.conv4 = conv(256, 512, 3, 2, 1, 0.2)
		self.conv4_1 = conv(512, 512, 3, 1, 1, 0.2)
		self.conv5 = conv(512, 512, 3, 2, 1, 0.2)
		self.conv5_1 = conv(512, 512, 3, 1, 1, 0.2)
		self.conv6 = conv(512, 1024, 3, 2, 1, 0.2)

		self.rnn = nn.LSTM(
			input_size = 6*20*1024, hidden_size=1000, num_layers=2, batch_first=True
			)
		self.lstm_dropout = nn.Dropout(0.5)
		self.fc = nn.Linear(1000, 6)

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

	def forward(self, x):
		batch_size = x.size(0)
		seq_len = x.size(1) # 
		print("Seq Len <model.py>: ", seq_len)
		# print("x size :",sys.getsizeof(x))
		x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv3_1(x)
		x = self.conv4(x)
		x = self.conv4_1(x)
		x = self.conv5(x)
		x = self.conv5_1(x)
		x = self.conv6(x)
		print("before flat : ", x.shape)
		x = x.view(batch_size, seq_len, -1) # Flattening

		print("Flattened",x.shape)
		x,_= self.rnn(x)
		print(x.shape)
		x = self.lstm_dropout(x)
		print(x.shape)
		x = self.fc(x)
		print(x.shape)
		
		print("Done forward")

		return x

	
	def get_loss(self, seq, pos, ang):
		pos = pos[:, 1:, :]
		ang = ang[:, 1:, :]
		y_hat = self.forward(seq)
		pos_loss = nn.functional.mse_loss(y_hat[:, :, 3:], pos)
		ang_loss = nn.functional.mse_loss(y_hat[:, :, :3], ang)
		return 100 * ang_loss + pos_loss

	def save_model(self, epoch, save_path):
		torch.save(
			{
			"epoch":epoch,
			"model_state_dict":self.state_dict(),
			"optimizer_state_dict": optimizer.state_dict()
			},
			f"{save_path}/{epoch}.pth"
		)



if __name__ == "__main__":
	model = DeepVOLite()
	# model=DeepVO()
	print("Num params DeepVOLite : ",sum(p.numel() for p in model.parameters()))


