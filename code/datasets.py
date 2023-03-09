import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

class OmegaDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        for dir in next(os.walk(f'../data/omega'))[1]:
            print(dir)
            for file in os.listdir(f'../data/omega/{dir}'):
                if file == ".DS_Store":
                    continue
                img = Image.open(f'../data/omega/{dir}/{file}')
                img = img.convert('RGB')
                # img = img / 255.0
                # img = np.reshape(img, (-1, 28, 28, 1)).astype(np.float32)
                self.data.append(np.array(img))
                self.labels.append(dir)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label



class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, transform=None):
        if training==True:
            f = open('../data/MNIST/raw/train-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('../data/MNIST/raw/train-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        else:
            f = open('../data/MNIST/raw/t10k-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('../data/MNIST/raw/t10k-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
        ys = ys.astype(np.int)
        self.x_data = xs
        self.y_data = ys
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_data[idx].reshape(28, 28))
        y = torch.tensor(np.array(self.y_data[idx]))
        if self.transform:
            x = self.transform(x)
        x = transforms.ToTensor()(np.array(x)/255)
        return x, y

