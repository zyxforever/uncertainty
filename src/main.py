import argparse

def config():
    parser = argparse.ArgumentParser(description='uncertainty')
    parser.add_argument('--dataset_path',default='/home/zyx/datasets')
    parser.add_argument('--train_batch_size',default=128)
    parser.add_argument('--epoch',default=20)
    return parser.parse_args()
def main():
    args=config()
if __name__=='__main__':
    main()