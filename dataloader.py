import os
from PIL import Image
from torchvision import transforms

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
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

def getDataloader(input_folder, label_folder=None, batch_size=4):
    dataset = MyDataset(input_folder=input_folder, label_folder=label_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader
