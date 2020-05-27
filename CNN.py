from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
# Ignore warnings
import warnings
import torchvision.transforms as tvt


warnings.filterwarnings("ignore")

state = 1

class RetinalDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = os.path.join(self.root_dir, self.frame.iloc[item, 0])
        image = Image.open(img_name + ".jpeg")
        label = self.frame.iloc[item, 1:]
        label = np.array([label], dtype=np.int8)

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label, dtype=torch.long)

        return image, label



class RetNet(nn.Module):
    def __init__(self):
        super(RetNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fcl1 = nn.Linear(35344, 240)
        self.fcl2 = nn.Linear(240, 10)
        self.fcl3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # the number beluw for input is equal to sqrt(errored input size/(in_channels * out_channels))
        # view input = sqrt(357216/6/16) = 61, use this number to convert to linear layers above too
        x = x.view(-1, 35344)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        return x


if state == 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load into a dataLoader
    training_data = RetinalDataset('Training_values.csv', 'Train',
                                   transform=tvt.ToTensor())
    train_loader = DataLoader(training_data, batch_size=8,
                             shuffle=True , num_workers=0)
    net = RetNet()
    net.to(device)
    number_outputs = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum= 0.8)

    for epoch in range(6):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    PATH = 'Model'
    torch.save(net.state_dict(), PATH)
    state = 2

if state == 2:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PATH = 'Model'
    valiation_data = RetinalDataset('Validation_values.csv', 'Validation', transform=tvt.ToTensor())
    valid_loader = DataLoader(valiation_data, batch_size=2,
                              shuffle=False, num_workers=0)
    net = RetNet()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    net.to(device)
    gcorrect = 0
    correct = 0
    total = 0
    healthy = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            for i in range(len(labels)):
                if predicted[i] == 0:
                    healthy += 1
                if predicted[i] == labels[i]:
                    correct += 1
                    if predicted[i] == 1:
                        gcorrect += 1

    print('Accuracy of the network on the 60k test images: %d %%' % (100 * correct / total))
    print(gcorrect)
    print(healthy)
    print(total)




