
from tqdm import trange,tqdm 
from dataset import Dataset
from sklearn.metrics import accuracy_score
from algorithms.function import cal_ber
from models import get_model
import torch 
import numpy as np 
import logging 
import torch.nn as nn 
import argparse

logging.basicConfig(level = logging.INFO,format = '%(asctime)s-%(name)s -%(levelname)s:%(message)s')
logger=logging.getLogger(__name__)

'''
    1 Train the model equiped with dropout
    2 Evaluate the uncertainty of the sample 
    3 Evaluate the 
'''
class Trainer:
    def __init__(self):
        self.cfg=self.config()
        self.pbar=trange(self.cfg.epoch)
        print(self.cfg.model)
        self.model=get_model(self.cfg.model)(self.cfg)
        self.criterion=nn.CrossEntropyLoss()
        self.train_loader,self.test_loader=Dataset(self.cfg).load_dataloader()
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=self.cfg.lr)
        if self.cfg.cuda:
            self.model=self.model.cuda()

    def config(self):
        parser = argparse.ArgumentParser(description='uncertainty')
        parser.add_argument('--dataset_path',default='/home/zyx/datasets')
        parser.add_argument('--model',default='cnn', choices=['cnn', 'mcdropout', 'scissors'])
        parser.add_argument('--data_set',default='cifar10')
        parser.add_argument('--train_batch_size',default=128)
        parser.add_argument('--test_batch_size',default=512)
        parser.add_argument('--in_channels',default=3)
        parser.add_argument('--epoch',default=50)
        parser.add_argument('--lr',default=1e-2)
        parser.add_argument('--cuda',default=True)
        return parser.parse_args()
    def uncertainty(self):
        self.model.eval()
        self.model.enable_dropout()
        for images,labels in self.train_loader:
            if self.cfg.cuda:
                images=images.cuda()
                labels=labels.cuda()
            output=self.model(images)
            predictions=output.cpu().detach().numpy()
            # calculate the uncertainty  
            #logger.info(predictions)
    def evaluate(self):
        self.model.eval()
        pred_list=np.array([])
        label_list=np.array([])
        for images,labels in self.test_loader:
            if self.cfg.cuda:
                images=images.cuda()
                labels=labels.cuda()
            output,_=self.model(images)
            loss=self.criterion(output,labels)
            _,pred=torch.max(output,1)
            #logger.info(_)
            #pred_list.append(pred.cpu().numpy().resize(-1))
            pred_list=np.concatenate((pred_list,pred.cpu().numpy()))
            label_list=np.concatenate((label_list,labels.cpu().numpy()))
        acc=accuracy_score(pred_list,label_list)
        self.pbar.set_description("Accuracy:%s"%acc)
        #self.pbar.set_description(accuracy_score(pred_list,label_list))
    def fit(self):
        self.model=self.model.cuda()
        for i in self.pbar:
            input_ = []
            label_ = []
            pred_list=np.array([])
            label_list=np.array([])
            for images,labels in self.train_loader:
                if self.cfg.cuda:
                    images=images.cuda()
                    labels=labels.cuda()
                output,_=self.model(images)
                input_.extend(_.cpu().detach().numpy().reshape(len(labels), -1))
                label_.extend(labels.cpu().detach().numpy())
                loss=self.criterion(output,labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _,pred=torch.max(output,1)
                pred_list=np.concatenate((pred_list,pred.cpu().numpy()))
                label_list=np.concatenate((label_list,labels.cpu().numpy()))
            acc=accuracy_score(pred_list,label_list)
            logger.info("train acc:%s"%acc)
            inputs_ = np.array(input_)
            label_   = np.array(label_)
            self.evaluate()
            logger.info("qiang_index%s"%cal_ber(inputs_,label_,10))
            #self.uncertainty()
if __name__=='__main__':
    trainer=Trainer()
    trainer.fit()