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
    def load_data(self):
        if self.dataset=='mnist':
            self._load_mnist()
    def _load_mnist(self):
        mean, std = 0.13092539, 0.3084483
        train_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    def load_data(self):
        vec=sio.loadmat()

if __name__=='__main__':
    config=Config().get_config()
    dataset=Dataset(config)
