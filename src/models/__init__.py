
from .net import CNN,MCDropoutModel
from .vgg import VGG
def get_model(name):
    return {
        "cnn":CNN,
        "mcdropout":MCDropoutModel,
        "vgg":VGG
    }[name]