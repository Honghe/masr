import torch
from flask import Flask, request
import _init_path
from models.conv import GatedConv
import sys
import json

print("Loading model...")

# import beamdecode

print("Model loaded")

app = Flask(__name__)


# @app.route("/recognize", methods=["POST"])
# def recognize():
#     f = request.files["file"]
#     f.save("test.wav")
#     return beamdecode.predict("test.wav")

with open("./data/labels.json") as f:
    vocabulary = json.load(f)
    vocabulary = "".join(vocabulary)
model = GatedConv(vocabulary)
state_dict = torch.load("pretrained/model_57.pth")
model.load_state_dict(state_dict)
model.eval()

@app.route("/recognize", methods=["POST"])
def recognize_am():
    f = request.files["file"]
    f.save("test.wav")
    text = model.predict("test.wav")
    print('asr text: {}'.format(text))
    return text

app.run("0.0.0.0", debug=True)
