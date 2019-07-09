# -*- coding: utf-8 -*-

"""
训练数据格式：
使用自己的数据集来训练模型。你的数据集需要包含至少以下3个文件：

train.index
dev.index
labels.json
train.index和dev.index为索引文件，表示音频文件和标注的对应关系，应具有如下的简单格式：

/path/to/audio/file0.wav,我爱你
/path/to/audio/file1.wav,你爱我吗
...
labels.json是字典文件，应包含数据集标注中出现过的所有字符，表示为一个json数组。
其中第一个字符必须是无效字符（可任意指定，不和其他字符重复就行），预留给CTC作为blank label。
"""
import json
import os
from pypinyin import pinyin, Style

from utils import strQ2B, seg_char

DATA_BASE_PATH = '/home/ubuntu/honghe/data/asr/data_aishell'

OUTPUT_DATA_DIR = 'data'
train_filename = 'train.csv'
dev_filename = 'dev.csv'
labels_filename = 'labels.json'

def cn2pinyin(chars):
    # han先进行中文分字，并过滤空格
    chars = strQ2B(chars)
    chars = seg_char(chars)
    pny = [i[0] for i in pinyin(chars, style=Style.TONE3, heteronym=False)]

    # 拼音与中文转录是否分字后长度一样
    if len(pny) != len(chars):
        raise Exception('not same len: {}\n{}'.format(' '.join(pny), chars))

    # 使用[1-4]表示声调，其中不加数字表示轻声 >> 使用[1-5]，其中5表示轻声。
    # 与中文分字对应的即表示是英文，就不加音标
    pny = [v + '5' if v != chars[i] and v[-1] not in '1234' else v for i, v in enumerate(pny)]
    return pny

def parse_transcript_dict() -> dict:
    d = {}
    transcript_filepath = os.path.join(DATA_BASE_PATH, 'transcript/aishell_transcript_v0.8.txt')
    with open(transcript_filepath) as f:
        line = f.readline().strip()
        while line:
            basename, transcript = line.split(' ', 1)
            transcript = transcript.strip()
            transcript = transcript.replace(' ', '')
            # to pinyin
            transcript = cn2pinyin(transcript)
            d[basename] = transcript
            line = f.readline().strip()
    return d


def parse_vocabulary(ll: list) -> list:
    s = set()
    for line in ll:
        s.update(set(line))

    vocab_list = list(s)
    # 汉字按拼音排序
    vocab_list = sorted(vocab_list, key=lambda c: pinyin(c, style=Style.TONE3)[0])
    # '_', // 第一个字符表示CTC空字符，可以随便设置，但不要和其他字符重复。
    vocab_list.insert(0, '_')
    return vocab_list


def parse_data_tree(sub: str) -> dict:
    d = {}
    sub_wav = 'wav'
    sub_dir = os.path.join(DATA_BASE_PATH, sub_wav, sub)
    for people in sorted(os.listdir(sub_dir)):
        for file in sorted(os.listdir(os.path.join(sub_dir, people))):
            wav_name = os.path.splitext(file)[0]
            d[wav_name] = os.path.join(sub_wav, sub, people, file)
    return d


def save_parsed_dataset(wav_dict, transcript_dict, filename):
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DATA_DIR, filename), 'w') as f:
        for k in wav_dict:
            try:
                f.write(wav_dict[k] + ',' + ' '.join(transcript_dict[k]) + '\n')
            except KeyError as e:
                print(e)


def save_labels_json(vocab_list, filename):
    vocab_json = json.dumps(vocab_list, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DATA_DIR, filename), 'w') as f:
        f.write(vocab_json + '\n')


def preprocess():
    transcript_dict = parse_transcript_dict()
    print(dict(list(transcript_dict.items())[:10]))
    vocabulary = parse_vocabulary(transcript_dict.values())
    print(vocabulary)

    dev_dict = parse_data_tree('dev')
    train_dict = parse_data_tree('train')
    print(dict(list(dev_dict.items())[:10]))

    save_parsed_dataset(dev_dict, transcript_dict, dev_filename)
    save_parsed_dataset(train_dict, transcript_dict, train_filename)
    save_labels_json(vocabulary, labels_filename)


if __name__ == '__main__':
    preprocess()
