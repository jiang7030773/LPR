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
    x = Conv2D(base_conv * 1, (3,3), padding="same",name='conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #卷积层2
    x = Conv2D(base_conv * 2, (3,3), padding="same",name='conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(base_conv * 2, (3,3), padding="same",name='conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #卷积层3
    x = Conv2D(base_conv * 4, (3,3), padding="same",name='conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(base_conv * 4, (3,3), padding="same",name='conv6')(x)
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

batch_size = 32
image_size = [128,40]
img_dir = './car_pic/image/val/湘A23456.jpg'
images = np.zeros([batch_size, image_size[1], image_size[0], 3])
img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), 1)
# cv2.imshow('f',img)
images[0, ...] = img
images = np.transpose(images, axes=[0, 2, 1, 3])

base_model = build_model('./model/my_model_weights.h5')

y_pred = base_model.predict(images)
shape = y_pred[:,2:,:].shape
ctc_decode = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0]
out = K.get_value(ctc_decode)[:, :7]
out = ''.join([CHARS[x] for x in out[0]])
print(out)