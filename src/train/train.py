# reference ： https://github.com/imdarkie/Chinese-novel-generation
import numpy as np
from src.loadHelper.loadHelper import loadProcess
import tensorflow as tf
from tensorflow.contrib import rnn
from distutils.version import LooseVersion
import warnings

# 训练循环次数
num_epochs = 200

# batch大小
batch_size = 256

# lstm层中包含的unit个数
num_units = 512

# embedding layer的大小
embed_dim = 512

# 训练步长
seq_length = 30

# 学习率
learning_rate = 0.003

# 每多少步打印一次训练信息
show_every_n_batches = 30

# 保存session状态的位置
save_dir = './save'


def checkEnvironment():
    assert LooseVersion(tf.__version__) >= LooseVersion("1.0"), "Ensure your tensorflow version is older than 1.0 ..."
    if not tf.test.gpu_device_name():
        warnings.warn("No gpu device found ...")
    else:
        print('Default GPU Device: {}' + tf.test.gpu_device_name())


def getInputs():
    inputs = tf.placeholder(tf.int32, name='inputs')
    target = tf.placeholder(tf.int32, name='target')
    learningRate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs, target, learningRate


def initCell():
    num_layers = 2  # lstm网络的层数
    keep_prob = 0.8  # drop 时的保留概率
    cell = rnn.BasicLSTMCell(num_units)  # 创建一个包含num_units个隐层的cell
    drop = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)  # 使用dropout机制防止overfitting等
    cell = rnn.MultiRNNCell([drop for _ in range(num_layers)])  # 构建多层lstm网络
    initState = cell.zero_state(batch_size, tf.float32)  # 初始化状态为 0.0
    initState = tf.identity(initState, name='init_state')  # 使用tf.identify给init_state取个名字，后面生成文字的时候，要使用这个名字来找到缓存的state
    return cell, initState


def getEmbed(input_data, vocab_size, embedDim):
    embedding = tf.get_variable("embedding", [vocab_size, embedDim],
                                dtype=tf.float32)  # 随机初始化 vocab_size * embed_dim 纬的向量矩阵
    return tf.nn.embedding_lookup(embedding, input_data)  # word 2 vec ， 训练最合适 向量映射矩阵


def buidRnn(cell, input):
    outputs, finalState = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
    finalState = tf.identity(finalState, name="finalState")  # 同样给finalState一个名字，后面要重新获取缓存
    return outputs, finalState


def build(cell, inputData, vocab_size, embedDim):
    embed = getEmbed(inputData, vocab_size, embedDim)
    outputs, finalState = buidRnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())
    return logits, finalState


def getBatchs(inTxt, batchSize):
    num = len(inTxt)


if __name__ == '__main__':
    # checkEnvironment()
    train_graph = tf.Graph()
    with train_graph.as_default():
        # 获取预处理的文本数据
        dataSet, charToIntVocab, intToVocab, symbolTable = loadProcess()
        # 获取词汇表文字总量
        vocab_size = len(charToIntVocab)
        # 获取模型的输入，目标以及学习率节点，这些都是tf的placeholder
        input_text, targets, learningRate = getInputs()
        # 输入数据的shape
        input_data_shape = tf.shape(input_text)
        # 创建rnn的cell和初始状态节点，rnn的cell已经包含了lstm，dropout
        cell, initState = initCell()
        # 创建计算loss和finalstate的节点
        logits, finalState = build(cell, input_text, vocab_size, embed_dim)

    # with tf.Session(graph=train_graph) as sess:
