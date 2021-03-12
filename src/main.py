
from mc_dropout_model import MCDropoutModel
from dataset import Dataset

import argparse

def config():
    parser = argparse.ArgumentParser(description='uncertainty')
    parser.add_argument('--dataset_path',default='/home/zyx/datasets')
    parser.add_argument('--data_set',default='mnist')
    parser.add_argument('--train_batch_size',default=100)
    parser.add_argument('--epoch',default=1)
    return parser.parse_args()
def train():
    pass 
def main():
    cfg=config()
    model=MCDropoutModel()
   
    train_loader,test_loader=Dataset(cfg).load_dataloader()
    for i in range(cfg.epoch):
        for images,labels in train_loader:
            print(images.shape)
            print(labels.shape)
        #print("HelloWorld")
if __name__=='__main__':
    main()