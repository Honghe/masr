from flask import Flask, request

print("Loading model...")
import beamdecode

print("Model loaded")

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


app.run("0.0.0.0", debug=True)
