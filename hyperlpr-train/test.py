import os

import cv2
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.models import load_model

import matplotlib.pyplot as plt
import codecs

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z'
         ]



def test_model(y_pred,X_test):
    # get_value 以Numpy array的形式返回张量的值
    # ctc_decode 使用贪婪算法或带约束的字典搜索算法解码softmax的输出
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :7]
    out = ''.join([CHARS[x] for x in out[0]])


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0] #取第一列所有数
    return ''.join([CHARS[x] for x in y])



img_dir = './car_pic/image/val/鲁B88888.jpg'

img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)

a = np.array(img)
a = a.reshape(128,40,3)
a = np.expand_dims(a,axis=0)

base_model = build_model('./model_weight.h5')

y_pred = base_model.predict(a)
y_pred = y_pred[:,2:,:]
table_pred = y_pred.reshape(-1, len(CHARS)+1)
res = table_pred.argmax(axis=1)
test_model(y_pred,'鲁B88888')
