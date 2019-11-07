import util
import numpy as np
from pickle import load
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model

from pickle import load
from keras.preprocessing.sequence import pad_sequences
import util

from PIL import Image
import matplotlib.pyplot as plt


def word_for_id(integer, tokenizer):
    """
    将一个整数转换为英文单词
    :param integer: 一个代表英文的整数
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :return: 输入整数对应的英文单词
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo_feature, max_length = 40):
    """
    根据输入的图像特征产生图像的标题
    :param model: 预先训练好的图像标题生成神经网络模型
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :param photo_feature:输入的图像特征, 为VGG16网络修改版产生的特征
    :param max_length: 训练数据中最长的图像标题的长度
    :return: 产生的图像的标题
    """
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        integ = np.argmax(yhat)
        word = word_for_id(integ, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def generate_caption_check(img_path):

    filename = 'Flickr_8k.testImages.txt'
    test = util.load_ids(filename)

    test_features = util.load_photo_features('features3.pkl', test)
    print("Photos: test=%d" % len(test_features))

    # load the model
    model_name = 'model_1.h5'
    model = load_model(model_name)

    tokenizer = load(open('tokenizer.pkl', 'rb'))
    expected = generate_caption(model, tokenizer, test_features[img_path.split("/")[1].split(".")[0]], 40)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.xlabel(expected[9:-6], fontsize=15, color="red")
    #plt.xlabel("WoW, You are so beautiful.", fontsize=15, color="red")
    plt.show()
    return expected


def evaluate_model(model, captions, photo_features, tokenizer, max_length = 40):
    """计算训练好的神经网络产生的标题的质量,根据4个BLEU分数来评估

    Args:
        model:　训练好的产生标题的神经网络
        captions: dict, 测试数据集, key为文件名(不带.jpg后缀), value为图像标题list
        photo_features: dict, key为文件名(不带.jpg后缀), value为图像特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_length：训练集中的标题的最大长度

    Returns:
        tuple:
            第一个元素为权重为(1.0, 0, 0, 0)的ＢＬＥＵ分数
            第二个元素为权重为(0.5, 0.5, 0, 0)的ＢＬＥＵ分数
            第三个元素为权重为(0.3, 0.3, 0.3, 0)的ＢＬＥＵ分数
            第四个元素为权重为(0.25, 0.25, 0.25, 0.25)的ＢＬＥＵ分数

    """
    actual, predicted = list(), list()
    for key, caption_list in captions.items():
        yhat = generate_caption(model, tokenizer, photo_features[key], max_length)
        references = [d.split() for d in caption_list]
        actual.append(references)
        predicted.append(yhat.split())

    blue1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    blue2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    blue3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    blue4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLUE-1: %f' % blue1)
    print('BLUE-2: %f' % blue2)
    print('BLUE-3: %f' % blue3)
    print('BLUE-4: %f' % blue4)

    # return blue1, blue2, blue3, blue4


def evaluate_check():
    filename = 'Flickr_8k.testImages.txt'
    test = util.load_ids(filename)

    test_features = util.load_photo_features('features3.pkl', test)
    print("Photos: test=%d" % len(test_features))

    # load the model
    model_name = 'model_1.h5'
    model = load_model(model_name)

    tokenizer = load(open('tokenizer.pkl', 'rb'))
    captions = util.load_clean_captions('descriptions.txt', test)

    evaluate_model(model, captions, test_features, tokenizer)


if __name__ == '__main__':
    # img_path = '..\Flicker8k_Dataset/9000268201_993b08cb0e.jpg'
    # print(generate_caption_check(img_path))
    evaluate_check()
