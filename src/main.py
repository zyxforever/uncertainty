
from tqdm import trange,tqdm 
from dataset import Dataset
from mc_dropout_model import MCDropoutModel

import torch 
import torch.nn as nn 
import logging 
import argparse

logging.basicConfig(level = logging.INFO,format = '%(asctime)s-%(name)s -%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)

class Trainer:
    def __init__(self):
        self.cfg=self.config()
        self.model=MCDropoutModel()
        self.criterion=nn.CrossEntropyLoss()
        self.train_loader,self.test_loader=Dataset(self.cfg).load_dataloader()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.01)
        if self.cfg.cuda:
            self.model=self.model.cuda()
    def config(self):
        parser = argparse.ArgumentParser(description='uncertainty')
        parser.add_argument('--dataset_path',default='/home/zyx/datasets')
        parser.add_argument('--data_set',default='mnist')
        parser.add_argument('--train_batch_size',default=128)
        parser.add_argument('--epoch',default=100)
        parser.add_argument('--cuda',default=True)
        return parser.parse_args()
    def evaluate(self):
        for images,labels in self.test_loader:
            if self.cfg.cuda:
                output=self.model(images)
                
    def run(self):
        pbar=trange(self.cfg.epoch)

        for i in pbar:
            for images,labels in self.train_loader:
                if self.cfg.cuda:
                    images=images.cuda()
                    labels=labels.cuda()
                output=self.model(images)
                loss=self.criterion(output,labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description("Loss:%-20s"%loss.item())
if __name__=='__main__':
    trainer=Trainer()
    trainer.run()