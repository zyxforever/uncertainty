import torch.nn as nn 

class MCDropoutModel(nn.Module):
    def __init__(self):
        super(MCDropoutModel,self).__init__()
        
    def forward(self,x):
        return x
