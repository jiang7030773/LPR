import os
import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Activation, Conv2D, Reshape
from keras.layers import BatchNormalization, Lambda, MaxPooling2D, Dropout
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping,Callback
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
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

def build_model(model_path):
    input_tensor = Input(name='the_input', shape=(128, 40, 3), dtype='float32')
    x = input_tensor
    base_conv = 32
    img_size = [128,40]
    rnn_size = 512
    #卷积层1
    x = Conv2D(base_conv * 1, (3,3), padding="same",name='conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #卷积层2
    x = Conv2D(base_conv * 2, (3,3), padding="same",name='conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #卷积层3
    x = Conv2D(base_conv * 4, (3,3), padding="same",name='conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #维度变换
    conv_to_rnn_dims = (img_size[0]//(2**3),(img_size[1]//(2**3))*128)
    x = Reshape(target_shape=conv_to_rnn_dims,name='reshape')(x)
    x =Dense(32,activation='relu',name='dense1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #两层bidirecitonal GRUs
    gru_1 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_1')(x)
    gru_1b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_1b')(x)
    gru1_merged = add([gru_1,gru_1b])
    gru_2 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_2')(gru1_merged)
    gru_2b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_2b')(gru1_merged)

    # transforms RNN output to character activations:  
    x = Dense(len(CHARS)+1,kernel_initializer='he_normal',name='dense2')(concatenate([gru_2,gru_2b]))
    y_pred = Activation('softmax',name='softmax')(x)
    base_model =  Model(inputs=input_tensor, outputs=y_pred)
    base_model.load_weights(model_path)
    return base_model

def test_model(y_pred,X_test):
    # get_value 以Numpy array的形式返回张量的值
    # ctc_decode 使用贪婪算法或带约束的字典搜索算法解码softmax的输出
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :7]
    out = ''.join([CHARS[x] for x in out[0]])
    return out


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0] #取第一列所有数
    return ''.join([CHARS[x] for x in y])



img_dir = './car_pic/image/train/川A02H25.jpg'

img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)
# cv2.imshow('f',img)
a = np.array(img)
a = a.transpose(1,0,2)
a = np.expand_dims(a,axis=0)

base_model = build_model('./model/model_weight.h5')

y_pred = base_model.predict(a)
y_pred = y_pred[:,2:,:]
table_pred = y_pred.reshape(-1, len(CHARS)+1)
res = table_pred.argmax(axis=1)
out = test_model(y_pred,'川A02H25')
print(out)