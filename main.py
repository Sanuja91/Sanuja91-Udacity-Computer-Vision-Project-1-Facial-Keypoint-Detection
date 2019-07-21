import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import FCN, NaimishNet, initialize_weights_advance_
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from train import train_net


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    offline = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 20

    print("AI Running on", device)

    net = FCN().apply(initialize_weights_advance_).to(device)
    print(net)

    data_transform = transforms.Compose(
        [Rescale(250), RandomCrop(224), Normalize(), ToTensor()])

    transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    # load training data in batches
    batch_size = 128

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    losses = train_net(n_epochs, net, train_loader,
                       device, optimizer, criterion)

    plt.xlabel("Steps")
    plt.ylabel("MSE Loss")
    plt.plot(losses, "g-")
