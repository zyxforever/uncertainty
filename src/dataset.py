import os 
import torch 
import numpy as np
import scipy.io as sio
import torchvision 
from config import Config
from torchvision import transforms

class Dataset:
    def __init__(self,config):
        self.config=config
        self.dataset=self.config["dataset"]
        self.data_dir=self.config["data_dir"]
        self.filepath = os.path.join(self.data_dir, self.dataset + ".MAT")
        self.train_dataset=self.load_data()
        self.train_loader=self.load_dataloader()
    def load_data(self):
        if self.dataset=='mnist':
            self._load_mnist()
    def load_dataloader(self):
        return self.train_loader
    def get_dataloader(self):

    def _load_mnist(self):
        mean, std = 0.13092539, 0.3084483
        train_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
        vec=sio.loadmat(dataset_path)
        labels=vec['']
        data=vec['']

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
