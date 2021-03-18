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
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3),
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
        self.fc=nn.Sequential(
            nn.Linear(128*4*4,1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Linear(1024,128),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(128,10),
            nn.Softmax(dim=1),
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x 

class DropoutModel(nn.Module):
    def __init__(self):
        
        pass 
if __name__=='__main__':
    model=DropoutModel()