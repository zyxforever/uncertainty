import torch.nn as nn 

class McDropoutModel(nn.Module):
    def __init__(self):
        super(McDropoutModel,self).__init__()
        
    def forward(self,x):
        return x
