import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from main import config
from sklearn.metrics import normalized_mutual_info_score as NMI 
from torch.nn import functional as F
from torch.utils import data 
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=64,affine=False)

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=64,affine=False)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=64,affine=False)

        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=128,affine=False)

        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=128,affine=False)

        self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=128,affine=False)


        self.conv7 = nn.Conv2d(in_channels=128,out_channels=10,kernel_size=1,padding=0)

        # self.conv7 = nn.Conv2d(in_channels=128,out_channels=10,kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=10,affine=False)

        # self.fc1 = nn.Linear(in_features=10 * 3 * 3, out_features=10)
        # self.fc1 = nn.Linear(in_features=10*4*4, out_features=10)
        self.fc1 = nn.Linear(in_features=10 * 1 * 1, out_features=10)

        self.fc2 = nn.Linear(in_features=10, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.bm1 = nn.BatchNorm2d(64,affine=False)
        self.bm2 = nn.BatchNorm2d(128,affine=False)
        self.bm3 = nn.BatchNorm2d(10,affine=False)

        self.fn1 = nn.BatchNorm1d(10,affine=False)
        self.fn2 = nn.BatchNorm1d(num_classes,affine=False)

        self.fn3 = nn.BatchNorm1d(10,affine=False)


        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')   ##
                nn.init.constant_(m.bias,0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.eye_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.max_pool2d(x, 2,2)
        x = self.bm1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = F.max_pool2d(x, 2)
        x = self.bm2(x)

        x = F.relu(self.bn7(self.conv7(x)))

        x = F.avg_pool2d(x,2,2)
        x = self.bm3(x)

        x = x.view(-1,10*1*1)

        x = F.relu(self.fn1(self.fc1(x)))
        x = F.relu(self.fn2(self.fc2(x)))

        b = torch.exp(x)

        a = self.fn3(x)
        y = self.softmax(x)

        return x, y


if __name__=='__main__':
    args=config()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.13092539],[0.3084483])])
    train_dataset=MNIST(args.dataset_path,train=True,transform=transform)
    train_loader=DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True)
    model=ConvNet(10)
    model.eval()
    samples=enumerate(train_loader)
    for i in range(args.epoch):
        batch_idx,(train_data,train_target) =next(samples)
        for i in range(1):
            x,predict=model(train_data)
            predict=l2_norm(predict)
            sim=torch.mm(predict,predict.t())
            pos_loc = torch.gt(sim,0.95)
            neg_loc = torch.lt(sim,0.455)
            
            train_target_one_hot=F.one_hot(train_target)
            sim_onehot=torch.mm(train_target_one_hot,train_target_one_hot.t())

            print(sim_onehot[pos_loc].shape)
            print(sim_onehot[neg_loc].shape)
    pass
