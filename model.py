import torch
import torch.nn as nn
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================
# Model
# ============================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = nn.MaxPool2d(kernel_size=2)(x)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)
        # x = torch.where(x>0.5, torch.tensor(1), torch.tensor(0))
        return x

# ============================
# Loss
# ============================

def dice_coefficient(y_pred, y_true):
    eps = 1.0
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def dice_loss(y_pred, y_true):
    return 1 - dice_coefficient(y_pred, y_true)

def get_optimiazer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)


# ============================
# Trainer
# ============================

def trainer(train_loader, model, optimizer, loss_fn, num_epochs, device, save_path: str):
    if not save_path.endswith('pth'):
        raise Exception('Invalid save path')
    
    best_loss = 1
    for epoch in range(num_epochs):
        total_loss = 0

        loop = tqdm(enumerate(train_loader), ncols=100, total=len(train_loader))
        loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')

        for i, (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            loop.set_postfix_str(f'loss={loss.item():.6f}')
                
        total_loss /= len(train_loader)
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), save_path)
            print(f'Save checkpoint in Epoch[{epoch+1}/{num_epochs}].')