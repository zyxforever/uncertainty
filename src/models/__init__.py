
from .net import CNN,MCDropoutModel
def get_model(name):
    return {
        "cnn":CNN,
        "mcdropout":MCDropoutModel
    }[name]