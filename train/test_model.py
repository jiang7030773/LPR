import os
import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Activation, Conv2D, Reshape
from keras.layers import BatchNormalization, Lambda, MaxPooling2D, Dropout
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping,Callback
from keras.layers.recurrent import GRU,LSTM
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.regularizers import l2  #加入了l2正则化
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

def build_model(path):
    # 超参
    base_conv = 32
    l2_rate = 1e-5
    
    input_tensor = Input(name='the_input', shape=(128,40,3), dtype='float32')
    x = input_tensor
    #卷积层1
    for i, n_cnn in enumerate([3,4]):
        for j in range(n_cnn):
            x = Conv2D(base_conv * 2**i, (3,3), padding="same", kernel_initializer='he_uniform', 
                kernel_regularizer=l2(l2_rate))(x)
            x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)  
          
    for i, n_cnn in enumerate([6]):
        for j in range(n_cnn):
            x = Conv2D(base_conv * 2**2, (3,3), padding="same", kernel_initializer='he_uniform', 
                kernel_regularizer=l2(l2_rate))(x)
            x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
            x = Activation('relu')(x)
    #维度变换
    conv_shape = x.get_shape().as_list()
    rnn_length = conv_shape[1]
    rnn_dimen = conv_shape[2]*conv_shape[3]
    # print(conv_shape,rnn_length,rnn_dimen)

    x = Reshape(target_shape=(rnn_length,rnn_dimen))(x)
    x =Dense(64, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
    x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
    x = Activation('relu')(x)
    
    #两层bidirecitonal GRUs
    gru_1 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal')(x)
    gru_1b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal')(x)
    gru1_merged = add([gru_1,gru_1b])
    gru_2 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal')(gru1_merged)
    gru_2b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal')(gru1_merged)

    # transforms RNN output to character activations:  
    x = concatenate([gru_2,gru_2b])
    x = Dense(len(CHARS)+1,kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
    y_pred = Activation('softmax')(x)

    #打印出模型概况
    base_model = Model(inputs=input_tensor, outputs=y_pred)
    base_model.load_weights(path)
  
    return base_model

# 参数设置
batch_size = 1  #测试一张图，设置为1就行了
image_size = [128,40]
num_test = total_image = 0
rnn_size = 128

base_model = build_model('./model/my_model_weights_12.h5')

#识别车牌集
# with codecs.open('./car_pic/image/test_labels.txt',mode='r', encoding='utf-8') as f:
     
#     for line in f:
#         images = np.zeros([batch_size, image_size[1], image_size[0], 3])  
#         img_dir = './car_pic/image/test/'+ line.strip() +'.jpg'
#         img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), 1)
#         images[0, ...] = img
#         images = np.transpose(images, axes=[0, 2, 1, 3])
#         y_pred = base_model.predict(images)
#         shape = y_pred[:,2:,:].shape
#         ctc_decode = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0]
#         out = K.get_value(ctc_decode)[:, :7]
#         out = ''.join([CHARS[x] for x in out[0]])
#         total_image += 1
#         if out == line[0:7]:
#             num_test += 1
#         else:
#             print(out,line.strip())
#     print('总共准确识别%d张图片'%num_test)
#     percent = num_test/total_image
#     print('完全正确识别率为%3f'%percent)


# 识别单张车牌
img_dir = './car_pic/0.jpg'
images = np.zeros([batch_size, image_size[1], image_size[0], 3])
img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), 1)
# img = cv2.resize(img,(128,40))
# cv2.imshow('f',img)
images[0, ...] = img
images = np.transpose(images, axes=[0, 2, 1, 3])
y_pred = base_model.predict(images)
shape = y_pred[:,2:,:].shape
ctc_decode = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0]
out = K.get_value(ctc_decode)[:, :7]
out = ''.join([CHARS[x] for x in out[0]])
print(out)
