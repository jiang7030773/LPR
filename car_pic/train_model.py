import os
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda,BatchNormalization
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.callbacks import EarlyStopping,Callback

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt

#导入图片数据
import read_img_data
imgs = os.listdir("./car_pic/3922/")
data, label = read_img_data.load_data(imgs)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#GPU--动态申请显存
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

OUTPUT_DIR = 'image_ocr'

np.random.seed(55)

run_name = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M')
start_epoch = 0
stop_epoch  = 200
img_w = 128
img_h = 39
image_per_epoch = 1600
val_split = 0.2
val_image = int(image_per_epoch * (val_split))

# Network parameters
conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 32
rnn_size = 512
input_shape = (img_w, img_h, 3)  #channel 改为3

#实际的损失计算发生在这里，尽管它不是一个内部的Keras损失函数
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


#构建网络
input_data = Input(name="the_input",shape=input_shape,dtype="float32")   #为什么name一定要等于'the_input'才行？
inner = Conv2D(conv_filters,kernel_size,padding="same",kernel_initializer="he_normal",
			   name='conv1')(input_data)
#增加bn层
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)

inner = MaxPooling2D(pool_size=(pool_size,pool_size),name='maxpooling1')(inner)
inner = Conv2D(conv_filters,kernel_size,padding="same",kernel_initializer="he_normal",
			   name='conv2')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(pool_size,pool_size),name='maxpooling2')(inner)

conv_to_rnn_dims = (img_w//(pool_size**2),(img_h//(pool_size**2))*conv_filters) #?
inner = Reshape(target_shape=conv_to_rnn_dims,name='reshape')(inner)

#cut down input size going into RNN:
inner =Dense(time_dense_size,activation='relu',name='dense1')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
# Two layers of bidirecitonal GRUs
# GRU seems to work as well, if not better than LSTM:
gru_1 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_1')(inner)
gru_1b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_1b')(inner)
gru1_merged = add([gru_1,gru_1b])
gru_2 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_2')(gru1_merged)
gru_2b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_2b')(gru1_merged)

# transforms RNN output to character activations:
inner = Dense(read_img_data.get_output_size(),kernel_initializer='he_normal',name='dense2')(concatenate([gru_2,gru_2b]))
y_pred = Activation('softmax',name='softmax')(inner)

Model(inputs=input_data,outputs=y_pred).summary()
# labels = Input(name='the_labels',shape=[img_gen.absolute_max_string_len],dtype='float32')
labels = Input(name='the_labels',shape=[7],dtype='float32')
input_length = Input(name='input_length',shape=[1],dtype='int64')
label_length = Input(name='label_length',shape=[1],dtype='int64')
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer

loss_out = Lambda(ctc_lambda_func,output_shape=(1,),name='ctc')([y_pred,labels,input_length,label_length])
# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02,decay=1e-6,momentum=0.9,nesterov=True,clipnorm=5)

# model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc':lambda y_true,y_pred:y_pred},optimizer=sgd)
if start_epoch>0:
    weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
    model.load_weights(weight_file)

# captures output of softmax so we can decode the output during visualization
test_func = K.function([input_data], [y_pred])

# 反馈函数，即运行固定次数后，执行反馈函数可保存模型，并且可视化当前训练的效果
# viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

# model.fit_generator(generator=img_gen.next_train(), steps_per_epoch=(image_per_epoch - val_image),
#                         epochs=stop_epoch, validation_data=img_gen.next_val(), validation_steps=val_image,
#                         callbacks=[EarlyStopping(patience=10), viz_cb, img_gen], initial_epoch=start_epoch)
model.fit(x=[data,labels,7,7],y=label,epochs=30,verbose=1,validation_split=0.2,)