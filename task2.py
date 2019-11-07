from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras


def load_vgg16_model():
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """

    json_file = open("vgg16_exported.json")
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("vgg16_exported.h5")

    return model


def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式.
       将图像的每一个图像通道减去其均值

    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.

    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).

    # Returns
        A PIL Image instance.
    """
    img = pil_image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(target_size, pil_image.NEAREST)
    return np.asarray(img, dtype=K.floatx())


def extract_features(directory):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]

    Args:
        directory: 包含jpg文件的文件夹

    Returns:
        None
    """

    model = load_vgg16_model()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    features = dict()
    for index, fn in enumerate(listdir(directory)):

        print(fn)
        fn = directory + '/' + fn
        arr = load_img_as_np_array(fn, target_size=(224, 224))

        # 4D张量很适合用来存诸如JPEG这样的图片文件。之前我们提到过，一张图片有三个参数：高度、宽度和颜色深度。
        # 一张图片是3D张量，一个图片集则是4D，第四维是样本大小
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = preprocess_input(arr)
        feature = model.predict(arr, verbose=0)

        fn_id = fn.split('.')[-2].split('/')[-1]
        features[fn_id] = feature

    # file = open('features.pkl', 'wb')
    # dump(features, file)
    # file.close()
    return features


if __name__ == '__main__':
    # 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    directory = '..\Flicker8k_Dataset'
    features = extract_features(directory)
    print('提取特征的文件个数：%d' % len(features))
    print(keras.backend.image_data_format())
    # 保存特征到文件
    dump(features, open('features3.pkl', 'wb'))
