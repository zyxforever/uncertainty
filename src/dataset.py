import os 
import torch 
import numpy as np
import scipy.io as sio
import torchvision 
from torchvision import transforms

class Dataset:
    def __init__(self,config):
        self.config=config
        
    def load_dataloader(self):
        if self.config.data_set=='mnist':
            return self._load_mnist()
        elif self.config.data_set=='cifar10':
            return self._load_cifar10()

    def _load_cifar10(self):
        pass 
    def _load_mnist(self):
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/home/zyx/datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((,), (0.3081,))
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/home/zyx/datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((,), (0.3081,))
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=512, shuffle=False)
        return train_loader,test_loader


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,data,targets,transform):
        self.data=data
        self.targets=targets
        self.tranform=transform
        super(SimpleDataset,self).__init__()
    def __getitem__(self,idx):
        img=self.data[idx]
        target=self.targets[idx]
        if self.tranform is not None:
            img=self.transform(img)
        return img,target
    def __len__(self):
        return len(self.targets)
        
if __name__=='__main__':
    config=Config().get_config()
    dataset=Dataset(config)
