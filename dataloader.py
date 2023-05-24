import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================
# DRIVE
# ==========================

# label_folder为None就是获取测试数据，不为None就是获取训练数据
class DRIVE_Dataset(Dataset):
    def __init__(self, input_folder, label_folder=None):
        self.input_list = [os.path.join(input_folder, name) for name in sorted(os.listdir(input_folder))]
        self.label_list = None if label_folder is None else [os.path.join(label_folder, name) for name in sorted(os.listdir(label_folder))]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        input = Image.open(self.input_list[index])
        if self.label_list is not None:
            label = Image.open(self.label_list[index])
            return self.transform(input), self.transform(label)
        else:
            return self.transform(input)
    def __len__(self):
        return len(self.input_list)

# label_folder为None就是获取测试数据，不为None就是获取训练数据
def get_DRIVE_Dataloader(input_folder, label_folder=None, batch_size=4):
    dataset = DRIVE_Dataset(input_folder=input_folder, label_folder=label_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# ==========================
# CHASEDB1
# ==========================

# train为True表示获取训练集数据，为False表示获取测试数据
class CHASEDB1_Dataset(Dataset):
    def __init__(self, folder, train=True, *, train_num=20):
        base_list = sorted(os.listdir(folder))
        input_list = [os.path.join(folder, base_list[i]) for i in range(0, len(base_list), 3)]
        label1st_list = [os.path.join(folder, base_list[i]) for i in range(1, len(base_list), 3)]
        label2nd_list = [os.path.join(folder, base_list[i]) for i in range(2, len(base_list), 3)]

        self.train=train
        if train:
            assert train_num<= len(input_list)
            # 将1st label和2nd label都用作训练集，对训练集做数据增强
            self.input_list = input_list[0:train_num] + input_list[0:train_num]
            self.label_list = label1st_list[0:train_num] + label2nd_list[0:train_num]
        else:
            self.input_list = input_list[train_num:]
            self.label_list = [label1st_list[train_num:], label2nd_list[train_num:]]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        input = Image.open(self.input_list[index])
        if self.train:
            label = Image.open(self.label_list[index])
            return self.transform(input), self.transform(label)
        else:
            label1st = Image.open(self.label_list[0][index])
            label2nd = Image.open(self.label_list[1][index])
            return self.transform(input), self.transform(label1st), self.transform(label2nd)
    def __len__(self):
        return len(self.input_list)

# train为True表示获取训练集数据，为False表示获取测试数据
def get_CHASEDB1_Dataloader(folder, train=True, batch_size=4):
    dataset = CHASEDB1_Dataset(folder=folder, train=train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# ==========================
# Helper Function
# ==========================

def show_DRIVE_result(input: torch.Tensor, output: torch.Tensor, title: str=None):
    input = np.moveaxis(input.detach().cpu().numpy(), 1, -1)
    output = np.moveaxis(output.detach().cpu().numpy(), 1, -1)
    
    num = input.shape[0]
    subtitle = ['input', 'output']
    plt.figure(figsize=(num*5+2, 10))
    if title is not None:
        plt.suptitle(title, fontsize=20)
    for row, data in enumerate([input, output]):
        for col, item in enumerate(data):
            plt.subplot(2, num, (row)*num+(col+1))
            plt.title(f'{subtitle[row]} {col+1}')
            plt.imshow(item, cmap='gray')
    plt.show()

def show_CHASEDB1_result(input: torch.Tensor, label1st: torch.Tensor, label2nd:torch.Tensor, output: torch.Tensor, title: str=None):
    input = np.moveaxis(input.detach().cpu().numpy(), 1, -1)
    output = np.moveaxis(output.detach().cpu().numpy(), 1, -1)
    label1st = np.moveaxis(label1st.detach().cpu().numpy(), 1, -1)
    label2nd = np.moveaxis(label2nd.detach().cpu().numpy(), 1, -1)
    
    num = input.shape[0]
    subtitle = ['input', 'label1st', 'label2nd', 'output']
    plt.figure(figsize=(25, num*5))
    if title is not None:
        plt.suptitle(title, fontsize=20)
    for row in range(num):
        for col, data in enumerate([input, label1st, label2nd, output]):
            plt.subplot(num, 4, (row)*4+(col+1))
            plt.title(f'{subtitle[col]} {row+1}')
            plt.imshow(data[row], cmap='gray')
    plt.show()