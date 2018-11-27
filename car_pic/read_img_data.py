# 如何得到train_x和Train_y
import os
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import numpy as np
import codecs
# character classes and matching regex filter
# u/U:表示unicode字符串 
# r/R:非转义的原始字符串 
regex = r'^[a-z ]+$'
# alphabet = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet = u"京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
monogram_file = './car_pic/binaries.txt'

# 将字符映射到唯一整数值
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


# 将数字类反向转换为字符
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

# 计算字符的长度 
def get_output_size():
        return len(alphabet) + 1

# 建立对应的列表
def build_word_list(num_words, max_string_len=None, mono_fraction=0.5,minibatch_size=32):
    tmp_string_list = []
    max_string_len = max_string_len
    Y_data = np.ones([num_words, 7]) * -1
    X_text = []
    Y_len = [0] * num_words

    def _is_length_of_word_valid(word):
        return (max_string_len == -1 or
                max_string_len is None or
                len(word) <= max_string_len)

    # monogram file is sorted by frequency in english speech
    with codecs.open(monogram_file, mode='r', encoding='utf-8') as f:
        for line in f:
            if len(tmp_string_list) == int(num_words * mono_fraction):
                break
            word = line.rstrip() #删除后面空格
            if _is_length_of_word_valid(word):
                tmp_string_list.append(word)
    for i, word in enumerate(tmp_string_list):
        Y_len[i] = len(word)
        Y_data[i, 0:len(word)] = text_to_labels(word)
        X_text.append(word)
    Y_len = np.expand_dims(np.array(Y_len), 1)
    return Y_data

def load_data(imgs):
    print('load data from image')
    #imgs.sort(key=lambda x:int(x[:-4])) #给文件排序
    num = len(imgs)
    print("图片数量：",num)
    data = np.empty((num, 1, 39, 128), dtype="float32")
    label = np.empty((num,), dtype="uint8")

    i = 0
    for imgFile in imgs:
        img = Image.open("./car_pic/3922/" + imgFile).convert('L')
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr
        # label[i] = text_to_labels(imgs[i].split('.')[0])
        i = i + 1
    label = build_word_list(num,7,mono_fraction=1,minibatch_size=32)
    return data, label

if __name__ == "__main__":
    imgs = os.listdir("./car_pic/3922/")
    data, label = load_data(imgs)
    for i in label:
        print("label:", i)