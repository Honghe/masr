# -*- coding: utf-8 -*-
import re

import numpy as np


def seg_char(line):
    # 分割中文
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(line)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars


def strQ2B(ustring):
    """
    中文文字永远是全角，只有英文字母、数字键、符号键才有全角半角的概念,一个字母或数字占一个汉字的位置叫全角，占半个汉字的位置叫半角。
    标点符号在中英文状态下、全半角的状态下是不同的。
    转换说明
    全角半角转换说明

    有规律的：

    全角字符unicode编码从65281~65374 （十六进制 0xFF01 ~ 0xFF5E）
    半角字符unicode编码从33~126 （十六进制 0x21~ 0x7E）

    特殊的：

    空格比较特殊，全角为 12288（0x3000），半角为 32（0x20）
    除空格外，全角/半角按unicode编码排序在顺序上是对应的（半角 + 0xfee0= 全角）,所以可以直接通过用+-法来处理非空格数据，对空格单独处理。
    :return:
    """
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]
