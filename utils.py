from data_utils import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


def train(model, train_loader, optimizer, epoch, device, loss_folder):
	start = time.time()

	model.train()
	train_losses = 0.0
	immediate_losses = 0.0
	loss_to_plot = 0.0
	losses = []

	i=0
	for seq, pos, ang in train_loader:
		# print("Got a batch from loader")
		# print(sys.getsizeof(seq))
		# print(sys.getsizeof(pos))
		# print(sys.getsizeof(ang))

		# print("please work")
		# print(seq.shape)
		# print(pos.shape)
		# print(ang.shape)

		optimizer.zero_grad()
		i = i+1
		seq = seq.to(device)
		pos = pos.to(device)
		ang = ang.to(device)

		# print("Starting loss calc")
		loss = model.get_loss(seq, pos, ang)
		train_losses += loss.item()
		immediate_losses += loss.item()
		print("Loss:",loss.item())
		loss_to_plot += loss.item()
		loss.backward()
		loss=0
		optimizer.step()
		if i % 10 == 0:
			losses.append(loss_to_plot / 10)
			print(losses)
			loss_to_plot = 0.0

	plt.clf()
	plt.plot(losses)
	plt.savefig(f"{loss_folder}/{epoch}.png")
	train_losses /= len(train_loader)
	print(f"Train Epoch {epoch}th loss: {train_losses}")

	end = time.time()
	print(f"Epoch {epoch} trainng time : ", (end-start)/60 , "min")
	return train_losses

def train_better(model, train_loader, optimizer, epoch, device, loss_folder):
	model.train()
	train_losses = 0.0
	immediate_losses = 0.0
	loss_to_plot = 0.0
	losses = []

	i=0
	for seq, pos, ang in train_loader:
		print("Got a batch from loader")
		
		# print(sys.getsizeof(seq))
		# print(sys.getsizeof(pos))
		# print(sys.getsizeof(ang))

		# print("please work")
		# print(seq.shape)
		# print(pos.shape)
		# print(ang.shape)

		optimizer.zero_grad()
		i = i+1
		seq = seq.to(device)
		pos = pos.to(device)
		ang = ang.to(device)

		print("Getting output")
		output = model(seq)
		print("Output shape :", output.shape)
		print("Getting pos loss")
		pos_loss = nn.functional.mse_loss(output[:, :, 3:], pos)
		print("Getting ang loss")
		ang_loss = nn.functional.mse_loss(output[:, :, :3], ang)
		print("Calc total loss")
		loss = 100 * ang_loss + pos_loss
		print("loss:", loss)
		# print("Starting loss calc")
		# loss = model.get_loss(seq, pos, ang)
		# print("Loss size : ",sys.getsizeof(loss))
		train_losses += loss.item()
		# immediate_losses += loss.item()
		loss_to_plot += loss.item()
		print("Starting backpropogation")
		loss.backward()
		print("Back P complete")
		loss=0
		optimizer.step()
		if i % 20 == 0:
			losses.append(loss_to_plot / 20)
			loss_to_plot = 0.0

	plt.clf()
	plt.plot(losses)
	plt.savefig(f"{loss_folder}/{epoch}.png")
	train_losses /= len(train_loader)
	print(f"Train Epoch {epoch}th loss: {train_losses}")
	return train_losses


def test(model,path, test_loader, epoch, trained_folder, scene, device):
	model.eval()
	odom = pykitti.odometry(path, scene)
	pose_estimates = [[0.0] * 6]  # Initial pose
	current_rotation = np.eye(3)  # Initial rotation matrix
	current_translation = np.zeros((3, 1))  # Initial translation vector
	gt=[]
	trajectory = []
	for i in range(len(odom)):
		gt.append(se3_to_position(odom.poses[i]))

	for i, batch in enumerate(test_loader):
		if i%10==0:
			print("Testing 03 batch : ", i)
		seq, _, _ = batch # rgb, pos, ang
		seq = seq.to(device)
		predicted = model(seq)
		predicted = predicted.data.cpu().numpy()

		if i==0:
			for pose in predicted[0]:
				pose += pose_estimates[-1]
				pose_estimates.append(pose.tolist())

			# Store the current trajectory
			initial_pose = np.concatenate((current_rotation.copy(), current_translation.copy()), axis=1).flatten()
			trajectory.append(initial_pose)
            
			predicted = predicted[1:]  # Skip the first prediction (already processed)


		for poses in predicted:
			rotation_angle = eulerAnglesToRotationMatrix([0, pose_estimates[-1][0], 0])
			location = rotation_angle.dot(poses[-1][3:])
			poses[-1][3:] = location[:]

			last_pose = poses[-1]
			new_rotation = eulerAnglesToRotationMatrix([last_pose[1], last_pose[0], last_pose[2]])
			current_rotation = new_rotation @ current_rotation


			for j in range(len(last_pose)):
				last_pose[j] = last_pose[j] + trajectory[-1][j]

			current_translation = np.array(last_pose[3:]).reshape((3, 1))
			last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi

			pose_estimates.append(last_pose.tolist())
			final_pose = np.concatenate((current_rotation.copy(), current_translation.copy()), axis=1).flatten()
			trajectory.append(final_pose)

		np.savetxt(f"{trained_folder}/{scene}_{epoch}_pred.txt", trajectory, fmt="%1.8f")

        




