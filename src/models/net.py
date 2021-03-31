import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from sklearn.metrics import normalized_mutual_info_score as NMI 
from torch.nn import functional as F
from torch.utils import data 
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

'''
   The basic CNN for classification with MNIST
    1 Without dropout:    with softmax: 
    2 With dropout: [99.133,99.608],with softmax 
'''
class CNN(nn.Module):
    def __init__(self,cfg):
        super(CNN,self).__init__()
        self.cfg=cfg
        self.layer1=nn.Sequential(
            nn.Conv2d(self.cfg.in_channels,16,kernel_size=3),
            #nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True)
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3),
            #nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            #nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True)
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3),
            #nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc1=nn.Sequential(
            nn.Linear(128*4*4,1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Linear(1024,128),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True)
        )
    
        self.fc=nn.Sequential(
            nn.Linear(128,10),
            nn.Softmax(dim=1),
        )
    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        #feature=x.view(x.size(0),-1)
        x=self.layer4(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        feature=x
        x=self.fc(x)
        return x,feature

class MCDropoutModel(CNN):
    def __init__(self):
        super(MCDropoutModel,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(1,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d(1)
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=1)
        )

        self.fc1=nn.Sequential(
            nn.Linear(256*11*11,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

        )

        self.fc2=nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

        )
        self.fc=nn.Sequential(
            nn.Linear(256,10),
            nn.Softmax(dim=1),
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        feature=self.fc2(x)
        x=self.fc(feature)
        return x,feature
if __name__=='__main__':
    model=MCDropoutModel()