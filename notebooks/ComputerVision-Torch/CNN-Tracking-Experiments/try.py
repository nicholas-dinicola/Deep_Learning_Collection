import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader

trainset = MNIST("data", train=True, download=False, transform=T.ToTensor())
dataloader = DataLoader(trainset, batch_size=5, shuffle=True)

if __name__ == "__main__":
    print(dataloader.dataset.classes)

