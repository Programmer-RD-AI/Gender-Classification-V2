from models.imports import *
from torchvision import transforms
from helper_funtions import *
from models.clf_and_conv1d import *
from models.clf import *
from models.clf import *
from models.tl import *
import os
import random

from PIL import Image
import face_recognition
import random
import cv2
from torch.nn import *
from torch.optim import *
import wandb
from ray import tune
from torchvision import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from helper_funtions import *
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
IMG_SIZE = 84
DATA_DIR = "./data/raw/"
PROJECT_NAME = "Gender-Clf"
device = torch.device("cuda")
lr = 0.001
criterion = BCELoss
batch_size = 32
epochs = 12
optimizer = Adam
config = {
    "IMG_SIZE": 84,
    "DATA_DIR": "./data/raw/",
    "PROJECT_NAME": "Gender-Clf",
    "device": torch.device("cuda"),
    "lr": 0.001,
    "criterion": BCELoss,
    "batch_size": 32,
    "epochs": 12,
    "optimizer": Adam,
    "input_size": (3, 84, 84),
    "output": 1,
    "output_ac": Sigmoid(),
}
