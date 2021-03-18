import torch.nn as nn 

class MCDropoutModel(nn.Module):
    def __init__(self):
        super(MCDropoutModel,self).__init__()
        self.dropout1=nn.Dropout(p=0.5)
        self.conv1=nn.Conv2d(1,32,5,1,0)
        self.dropout2=nn.Dropout(p=0.5)
        self.maxpool1=nn.MaxPool2d(2)
        self.dropout3=nn.Dropout(p=0.5)
        self.conv2=nn.Conv2d(32,32,5,1)
        self.dropout4=nn.Dropout(p=0.5)
        self.maxpool2=nn.MaxPool2d(2)
        self.dropout5=nn.Dropout(p=0.5)
        self.fc1=nn.Linear(512,500)
        self.dropout6=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(500,10)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.dropout1(x)
        x=self.conv1(x)
        x=self.dropout2(x)
        x=self.maxpool1(x)
        x=self.dropout3(x)
        x=self.conv2(x)
        x=self.dropout4(x)
        x=self.maxpool2(x)
        x=self.dropout5(x)
        x=x.flatten(start_dim=1)
        x=self.fc1(x)
        x=self.dropout6(x)
        x=self.fc2(x)
        x=self.softmax(x)
        return x
