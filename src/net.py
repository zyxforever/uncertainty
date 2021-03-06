import torch.nn as nn 
import torchvision.transforms as transforms
from main import config
from torch.utils import data 
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.dropout1=nn.Dropout(p=0.5)
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1)
        self.dropout2=nn.Dropout(p=0.5)
        self.max_pool1=nn.MaxPool2d(kernel_size=2)
        self.dropout3=nn.Dropout(p=0.5)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1)
        self.dropout4=nn.Dropout(p=0.5)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.dropout5=nn.Dropout(p=0.5)
        self.fc1=nn.Linear(30,500)
        self.dropout6=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(50,10)
    def forward(self,x):
        x=self.dropout1(x)
        x=self.conv1(x)
        x=self.dropout2(x)
        x=self.max_pool1(x)
        x=self.dropout3(x)
        print(x.shape)
        x=self.conv2(x)
        return x
if __name__=='__main__':
    args=config()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
    train_dataset=MNIST(args.dataset_path,train=True,transform=transform)
    train_loader=DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True)
    model=CNNNet()
    model.train()
    samples=enumerate(train_loader)
    for i in range(args.epoch):
        batch_idx,(train_data,train_target) =next(samples)
        predict=model(train_data)
        print(predict.shape)
    pass
