
from net import CNN
from tqdm import trange,tqdm 
from dataset import Dataset
from sklearn.metrics import accuracy_score
from mc_dropout_model import MCDropoutModel

import torch 
import numpy as np 
import torch.nn as nn 
import logging 
import argparse

logging.basicConfig(level = logging.INFO,format = '%(asctime)s-%(name)s -%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)

class Trainer:
    def __init__(self):
        self.cfg=self.config()
        self.pbar=trange(self.cfg.epoch)
        print(self.cfg.model)
        self.model=self.get_model(self.cfg.model)()
        self.criterion=nn.CrossEntropyLoss()
        self.train_loader,self.test_loader=Dataset(self.cfg).load_dataloader()
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=self.cfg.learning_rate)
        if self.cfg.cuda:
            self.model=self.model.cuda()
    def get_model(self,name):
        return {
            "cnn":CNN,
            "dropout":MCDropoutModel
        }[name]
    def config(self):
        parser = argparse.ArgumentParser(description='uncertainty')
        parser.add_argument('--dataset_path',default='/home/zyx/datasets')
        parser.add_argument('--model',default='cnn', choices=['cnn', 'dropout', 'scissors'])
        parser.add_argument('--data_set',default='mnist')
        parser.add_argument('--train_batch_size',default=128)
        parser.add_argument('--epoch',default=20)
        parser.add_argument('--learning_rate',default=1e-2)
        parser.add_argument('--cuda',default=True)
        return parser.parse_args()
    def evaluate(self):
        self.model.eval()
        pred_list=np.array([])
        label_list=np.array([])
        for images,labels in self.test_loader:
            if self.cfg.cuda:
                images=images.cuda()
                labels=labels.cuda()
            output=self.model(images)
            loss=self.criterion(output,labels)
            _,pred=torch.max(output,1)
            #logger.info(_)
            #pred_list.append(pred.cpu().numpy().resize(-1))
            pred_list=np.concatenate((pred_list,pred.cpu().numpy()))
            label_list=np.concatenate((label_list,labels.cpu().numpy()))
        acc=accuracy_score(pred_list,label_list)
        self.pbar.set_description("Accuracy:%s"%acc)
        #self.pbar.set_description(accuracy_score(pred_list,label_list))
    def run(self):
        for i in self.pbar:
            for images,labels in self.train_loader:
                if self.cfg.cuda:
                    images=images.cuda()
                    labels=labels.cuda()
                output=self.model(images)
                loss=self.criterion(output,labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.evaluate()
if __name__=='__main__':
    trainer=Trainer()
    trainer.run()