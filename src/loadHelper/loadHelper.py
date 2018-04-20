import numpy as np
import re
import pickle
import os

path = "../../res/panlong.txt"


def load(path):
    """
    读取指定目录的文件， 进行简单的空格替换
    :param path:
    :return:
    """
    ret = []
    pattern0 = re.compile(r" ")
    pattern1 = re.compile(r"\*+")
    with open(path, "r") as f:
        for line in f.readlines():
            temp = line.strip()
            temp = pattern0.sub(", ", temp)
            temp = pattern1.sub("", temp)
            if temp != "":
                ret.append(temp)
    return ret


def process():
    if readProcessedFlag():
        return loadPreProcess()
    else:
        return processAndSave()


def loadPreProcess():
    with open("process.txt", "rb") as f:
        return pickle.load(f)


def processAndSave():
    dataSet = load(path)
    charToIntTable, intToCharTable = createLookUpTable(dataSet)
    symbolTable = createSymbolsTable()
    with open("process.txt", "wb") as f:
        pickle.dump((dataSet, charToIntTable, intToCharTable, symbolTable), f)
        f.flush()
    writeProcessedFlag()
    return dataSet, charToIntTable, intToCharTable, symbolTable


def createLookUpTable(inputData):
    """
    创建文字、数字、字符到数字编号的映射
    :param inputData:
    :return:
    """
    characterSet = set(inputData)  # set 不保证元素顺序，相当于对数据进行了打乱
    # for idx, word in enumerate(characterSet):
    #     print(idx, word)
    char_to_int = {word: idx for idx, word in enumerate(characterSet)}  # 文字到数字的映射, 内容作为键， 内容序号作为值
    int_to_char = dict(enumerate(characterSet))  # 数字到文字的映射
    return char_to_int, int_to_char


def createSymbolsTable():
    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n', '.'])
    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R", "L"]
    return dict(zip(symbols, tokens))


def readProcessedFlag():
    if not os.path.exists('config.txt'):
        return False
    with open('config.txt', 'rb') as read_file:
        data = pickle.load(read_file)
        if data.get('isProcessed') == 'yes':
            return True
    return False


def writeProcessedFlag():
    with open('config.txt', 'wb') as f:
        d = {'isProcessed': 'yes'}
        pickle.dump(d, f)
        f.flush()


if __name__ == '__main__':
    dataSet, charToIntTable, intToCharTable, symbolTable = process()
    print(dataSet)
    print("\r\n\r\n")
    print(charToIntTable)
    print("\r\n\r\n")
    print(intToCharTable)
    print("\r\n\r\n")
    print(symbolTable)
