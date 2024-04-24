import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/research/KITTI_VO_RGB') 
SCENE = "03"
IMG_PATH = os.path.join(DATA_PATH, "sequences" ,SCENE, "image_2")
POSE_PATH = os.path.join(DATA_PATH, "poses", f"{SCENE}.txt")

SEQ_LENGTH = 6
TRAIN_SEQ = ["03"] #, "02", "08", "09"]
BATCH_SIZE = 6
NUM_WORKERS = 2
NUM_EPOCHS = 1

TRAINED_FOLDER="./trained_models"
LOSS_FOLDER="./loss_trained"

