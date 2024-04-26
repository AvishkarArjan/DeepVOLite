import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DeepVOLite
from utils import *
from data_utils import KittiOdomDataset
from config import *
import os
import argparse
import time

parser = argparse.ArgumentParser(description ='DeepVO paths and stuff')
parser.add_argument('--data', help='Dataset Path')
parser.add_argument('--mode', help='Train or Test')
parser.add_argument('--trained', help='Trained folder Path')
parser.add_argument('--loss', help='Loss folder Path')
parser.add_argument('--workers', help='Number of workers')
parser.add_argument('--batch', help='Batch Size')
parser.add_argument('--epochs', help='Number of Epochs')
parser.add_argument('--scene', help="The scene")

"""
python3 main.py --mode train --data "/home/avishkar/Desktop/research/KITTI_VO_RGB" --scene "00" --trained "trained_models" --loss "loss_trained" --workers 2 --batch 6 --epochs 1
"""

args = parser.parse_args()

mode = args.mode
loss_folder = args.loss
trained_folder = args.trained
data_path = args.data
num_workers = int(args.workers)
batch_size = int(args.batch)
num_epochs = int(args.epochs)
scene = args.scene
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not os.path.exists(loss_folder):
	os.mkdir(loss_folder)

if not os.path.exists(trained_folder):
	os.mkdir(trained_folder)

model = DeepVOLite()
model = model.to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)
cur=0

train_dataset =  KittiOdomDataset(scene, data_path)
test_dataset = KittiOdomDataset("03", data_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

epoch_losses = []

start = time.time()

if mode =="train":
	for epoch in range(cur+1, cur+num_epochs+1):
		print("Epoch:",epoch)
		epoch_loss = train(model, train_loader, optimizer, epoch, device, loss_folder)
		# epoch_loss = train_better(model, train_dataset, optimizer, epoch)
		plt.clf()
		plt.plot(epoch_losses)
		
		plt.clf()
		
		if epoch % 10 == 0 or (len(epoch_losses) == 0 or epoch_loss < min(epoch_losses)):
			plt.savefig(f"{loss_folder}/train_{epoch}.png")
			print(f"{epoch}th epoch saved")
			model.save_model(epoch, optimizer, trained_folder)
			test(model, data_path, test_loader, epoch, trained_folder, "04", device)

		epoch_losses.append(epoch_loss)

		end = time.time()

	print("Total training time : ", (end-start)/60, "min")

if mode=="test":
	model.load_state_dict(torch.load("./trained_models/8.pth")["model_state_dict"])
	test(model, data_path, test_loader, 1, trained_folder, "03", device)


"""
test loss

2.84
1.33
0.99
0.74
0.59
0.54
0.51
0.5
0.47
0.48
0.43
0.39
0.39
0.40

"""