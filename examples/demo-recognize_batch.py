import json
import os

import pandas as pd
import torch

from models.conv import GatedConv
from utils import levenshtein

with open("../data/labels.json") as f:
    vocabulary = json.load(f)
    vocabulary = "".join(vocabulary)
    model = GatedConv(vocabulary)
    state_dict = torch.load("../pretrained/model_{}.pth".format(158))
    model.load_state_dict(state_dict)


def asr(f_path):
    text = model.predict(f_path)
    return text
    text = model.predict(f_path)


if __name__ == '__main__':
    # record("record.wav", time=5)  # modify time to how long you want
    # f_path = 'record.wav'
    audio_dir = '/home/ubuntu/honghe/data/asr'
    meta_data_file_path = os.path.join(os.path.dirname(__file__), '../data', 'thchs_test.txt')
    meta_data = pd.read_csv(meta_data_file_path, sep='\t', header=None)
    cers = []
    for i in meta_data.index[:11]:
        filename, pnyn, chars = meta_data.loc[i]
        f_path = os.path.join(audio_dir, filename)
        y_p = asr(f_path)
        chars = ''.join(chars.split())
        print(chars)
        print(y_p)
        cer = levenshtein(y_p, chars) / len(chars)
        cers.append(cer)
        print('cer: {:.3f}'.format(cer))
    print('avg cer: {:.3f}'.format(sum(cers) / len(cers)))
