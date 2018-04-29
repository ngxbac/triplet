import math
import numpy as np
import pandas as pd
from torchvision import transforms
import torch


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    TRAIN_DIR = "/home/deeplearning/Kaggle/whale-categorization-playground/input/train"
    TEST_DIR = "/home/deeplearning/Kaggle/whale-categorization-playground/input/test"

    TRAIN_CSV = "/home/deeplearning/Kaggle/whale-categorization-playground/input/train.csv"
    TEST_CSV = "/home/deeplearning/Kaggle/whale-categorization-playground/input/sample_submission.csv"

    TRAIN_DIR_2 = "/home/deeplearning/Kaggle/whale-categorization-playground/input/dlib_gen"

    SZ = (224, 224)
    BS = 10
    N_SAME = 7
    START_EPOCH = 0
    EPOCHS = 100
    INPUT_SHAPE = (3,) + SZ
    LR = 0.1
    LR_DECAY = 1e-4

    TRANSFORM = transforms.Compose([
        transforms.Resize(SZ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    MARGIN = 0.2
    DISTANCE_TH = 0.6
    MODEL_NAME = "whale_resnet50"
    MODEL_CKP = f"ckp_{MODEL_NAME}.pth.tar"
    MODEL_CKP_BEST = f"ckp_best_{MODEL_NAME}.pth.tar"
    RESUME = f"runs/{MODEL_NAME}/{MODEL_CKP}" #MODEL_CKP_BEST
    LOG_INTERVAL = 100
    VISDOM_NAME = "WhaleTriplet"

    def __init__(self):
        self.TRAIN_DF = pd.read_csv(self.TRAIN_CSV)
        self.N_TRAIN = len(self.TRAIN_DF)
        self.N_CLASS = self.TRAIN_DF["Id"].nunique()
        self.CLASSES = self.TRAIN_DF["Id"].unique()
        self.CLS_TO_IDX = {}
        self.IDX_TO_CLS = {}
        self.CLS_TO_INDICATES = {}
        for i in range(self.N_CLASS):
            cls = self.CLASSES[i]
            self.CLS_TO_IDX[cls] = i
            self.IDX_TO_CLS[i] = cls

            df_indicates = self.TRAIN_DF[self.TRAIN_DF["Id"] == cls]
            self.CLS_TO_INDICATES[cls] = df_indicates["Image"].tolist()

        self.TEST_DF = pd.read_csv(self.TEST_CSV)
        self.N_TEST = len(self.TEST_DF)

        self.USE_GPU = torch.cuda.is_available()
        if self.USE_GPU:
            print("Using GPU")

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")