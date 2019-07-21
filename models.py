import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()

        CONV_1 = 32
        CONV_2 = 64
        CONV_3 = 128
        CONV_4 = 256

        FC_1 = 43264
        FC_2 = 1000
        FC_3 = 1000
        OUTPUT = 136

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, CONV_1, 4)
        self.conv2 = nn.Conv2d(CONV_1, CONV_2, 3)
        self.conv3 = nn.Conv2d(CONV_2, CONV_3, 2)
        self.conv4 = nn.Conv2d(CONV_3, CONV_4, 1)

        # Maxpool layers
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Dropout layers
        self.do1 = nn.Dropout(p=0.1)
        self.do2 = nn.Dropout(p=0.2)
        self.do3 = nn.Dropout(p=0.3)
        self.do4 = nn.Dropout(p=0.4)
        self.do5 = nn.Dropout(p=0.5)
        self.do6 = nn.Dropout(p=0.6)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(CONV_1)
        self.bn2 = nn.BatchNorm2d(CONV_2)
        self.bn3 = nn.BatchNorm2d(CONV_3)
        self.bn4 = nn.BatchNorm2d(CONV_4)
        self.bn5 = nn.BatchNorm1d(FC_2)
        self.bn6 = nn.BatchNorm1d(FC_3)

        # Fully connected layers
        self.fc1 = nn.Linear(FC_1, FC_2)
        self.fc2 = nn.Linear(FC_2, FC_3)
        self.fc3 = nn.Linear(FC_3, OUTPUT)

    def forward(self, x):
        # print("INPUT", x.shape)
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        # print("CONV 1", x.shape)
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        # print("CONV 2", x.shape)
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        # print("CONV 3", x.shape)
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))
        # print("CONV 4", x.shape)

        x = x.view(x.size(0), -1)
        # print("FLATTEN", x.shape)

        x = self.do5(F.relu(self.bn5(self.fc1(x))))
        # print("FC 1", x.shape)
        x = self.do6(F.relu(self.bn6(self.fc2(x))))
        # print("FC 2", x.shape)
        x = self.fc3(x)
        # print("OUTPUT", x.shape)

        return x


class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()

        CONV_1 = 32
        CONV_2 = 64
        CONV_3 = 128
        CONV_4 = 256
        CONV_5 = 512
        CONV_6 = 1024
        CONV_7 = 2048
        CONV_8 = 1024
        CONV_9 = 512
        CONV_10 = 256
        OUTPUT = 136

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=CONV_1,
                      kernel_size=4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_1, out_channels=CONV_2,
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_2, out_channels=CONV_3,
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_3, out_channels=CONV_4,
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_4)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_4, out_channels=CONV_5,
                      kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_5)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_5, out_channels=CONV_6,
                      kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_6)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_6, out_channels=CONV_7,
                      kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_7)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_7, out_channels=CONV_8,
                      kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_8)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_8, out_channels=CONV_9,
                      kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_9)
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_9, out_channels=CONV_10,
                      kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(CONV_10)
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=CONV_10, out_channels=OUTPUT,
                      kernel_size=2, stride=1, padding=0),
        )

    def forward(self, x):
        # print("INPUT", x.shape)
        x = self.conv1(x)
        # print("CONV 1", x.shape)
        x = self.conv2(x)
        # print("CONV 2", x.shape)
        x = self.conv3(x)
        # print("CONV 3", x.shape)
        x = self.conv4(x)
        # print("CONV 4", x.shape)
        x = self.conv5(x)
        # print("CONV 5", x.shape)
        x = self.conv6(x)
        # print("CONV 6", x.shape)
        x = self.conv7(x)
        # print("CONV 7", x.shape)
        x = self.conv8(x)
        # print("CONV 8", x.shape)
        x = self.conv9(x)
        # print("CONV 9", x.shape)
        x = self.conv10(x)
        # print("CONV 10", x.shape)
        x = self.conv11(x)

        x = x.view(x.size(0), -1)
        # print("FLATTEN", x.shape)

        return x


def initialize_weights_advance_(model):
    '''Initializes weight depending on type of neuron'''
    if isinstance(model, nn.Conv1d):
        init.normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.Conv2d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.Conv3d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.ConvTranspose1d):
        init.normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.ConvTranspose2d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.ConvTranspose3d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.BatchNorm1d):
        init.normal_(model.weight.data, mean=1, std=0.02)
        init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        init.normal_(model.weight.data, mean=1, std=0.02)
        init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        init.normal_(model.weight.data, mean=1, std=0.02)
        init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.Linear):
        init.xavier_normal_(model.weight.data)
        init.normal_(model.bias.data)
    elif isinstance(model, nn.LSTM):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(model, nn.LSTMCell):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(model, nn.GRU):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(model, nn.GRUCell):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
