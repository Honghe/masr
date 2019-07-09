from flask import Flask, request

from beamdecode import BeamDecode

beamdecode = BeamDecode()
app = Flask(__name__)


# 声学模型+语言模型
@app.route("/recognize", methods=["POST"])
def recognize():
    f = request.files["file"]
    f.save("test.wav")
    return beamdecode.predict("test.wav")


# 只有声学模型
@app.route("/am", methods=["POST"])
def recognize_am():
    f = request.files["file"]
    f.save("test.wav")
    text = beamdecode.model.predict("test.wav")
    print('asr text: {}'.format(text))
    return text


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('-m', default=0, type=int, help='saved model to load', )
    args = parser.parse_args()
    epoch_load = args.m

    print("Loading model {}...".format(epoch_load))
    beamdecode.init(epoch_load)
    print("Model loaded")

    app.run("0.0.0.0", debug=False)
