import json

import _init_path
import torch

from models.conv import GatedConv

def recognize(epoch_load):
    with open("./data/labels.json") as f:
        vocabulary = json.load(f)
        model = GatedConv(vocabulary)
        # load to CPU, `map_location=lambda storage, loc: storage`
        state_dict = torch.load("pretrained/model_{}.pth".format(epoch_load), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        model.eval()

    text = model.predict("test.wav")

    print("")
    print("识别结果:")
    print(text)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('-m', default=0, type=int, help='saved model to load', )
    args = parser.parse_args()
    epoch_load = args.m
    print('using epoch number {}'.format(epoch_load))
    recognize(epoch_load)