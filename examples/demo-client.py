import requests
import _init_path
import feature
from record import record

server = "http://172.31.23.184:5000/am"
server2 = "http://172.31.23.184:5000/recognize"

def asr(f_path):
    f = open(f_path, "rb")

    files = {"file": f}

    r = requests.post(server, files=files)

    print("")
    print("识别结果am:")
    print(r.text)

    f = open(f_path, "rb")
    files = {"file": f}
    r2 = requests.post(server2, files=files)
    print("识别结果am+lm:")
    print(r2.text)

if __name__ == '__main__':
    # record("record.wav", time=5)  # modify time to how long you want
    f_path = 'record.wav'
    asr(f_path)