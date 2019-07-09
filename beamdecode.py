import json

import torch
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder

import feature
from models.conv import GatedConv

alpha = 0.8
beta = 0.3
lm_path = "lm/zh_giga.no_cna_cmn.prune01244.klm"
cutoff_top_n = 40
cutoff_prob = 1.0
beam_width = 32
num_processes = 4
blank_index = 0


class BeamDecode():
    def init(self, epoch_num=0):
        with open("./data/labels.json") as f:
            vocabulary = json.load(f)
            vocabulary = "".join(vocabulary)
            model = GatedConv(vocabulary)
            state_dict = torch.load("pretrained/model_{}.pth".format(epoch_num))
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model

        self.decoder = CTCBeamDecoder(
            model.vocabulary,
            lm_path,
            alpha,
            beta,
            cutoff_top_n,
            cutoff_prob,
            beam_width,
            num_processes,
            blank_index,
        )

    def translate(self, vocab, out, out_len):
        return "".join([vocab[x] for x in out[0:out_len]])

    def predict(self, f):
        wav = feature.load_audio(f)
        spec = feature.spectrogram(wav)
        spec.unsqueeze_(0)
        with torch.no_grad():
            y = self.model.cnn(spec)
            y = F.softmax(y, 1)
        y_len = torch.tensor([y.size(-1)])
        y = y.permute(0, 2, 1)  # B * T * V
        print("decoding")
        out, score, offset, out_len = self.decoder.decode(y, y_len)
        return self.translate(self.model.vocabulary, out[0][0], out_len[0][0])
