import os
import shutil

import torch
import datasets
import torch.optim as optim
from torchvision.models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import config as cfg
import models
import losses as loss
import utils

config = cfg.Config()

# training
# Create embedding net
embedding_size = 256
model = models.TripletModel(embedding_size)
# Create criterion
criterion = loss.TripletMarginLoss(config.MARGIN)
optimizer = optim.Adam(model.parameters(), lr=config.LR)
if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

best_acc = 0
start_epoch = 0
if config.RESUME:
    ckp = utils.load_checkpoint(config.RESUME)
    model.load_state_dict(ckp["state_dict"])
    start_epoch = ckp["epoch"]

print(f"Start epoch {start_epoch}")

for epoch in range(start_epoch, config.EPOCHS):
    # Train loader
    train_dataset = datasets.WhaleDataset(config, n_triplets=5000, transform=config.TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=config.BS,
                              shuffle=False)
    # Test loader
    test_dataset = datasets.WhaleDataset(config, n_triplets=1000, transform=config.TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=config.BS,
                             shuffle=False)

    # train for one epoch
    utils.train_triplet(train_loader, model, criterion, optimizer, epoch)
    # evaluate on validation set
    val_loss = utils.test_triplet(test_loader, model, criterion, epoch)
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
    }, False)