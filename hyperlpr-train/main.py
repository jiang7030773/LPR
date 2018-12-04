import os
import codecs
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
from keras.utils.vis_utils import plot_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z'
         ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
NUM_CHARS = len(CHARS)
# GPU选用1060，不选会自动调用集显
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#动态申请显存
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


#必要参数
num_channels = 3
ti = './car_pic/image/train/' #训练图片目录
tl = './car_pic/image/train_labels.txt' #训练标签文件
vi = './car_pic/image/val/'  #验证图片目录
vl = './car_pic/image/val_labels.txt' #验证标签文件
img_size = [128,40] #训练图片宽和高
label_len = 7 #标签长度
dir_log = './logs/'
c = './car_pic/image/' #checkpoints format string
num_epochs = 200 #number of epochs
start_of_epoch = 0

#网络参数
conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 32
rnn_size = 512
batch_size = 16

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    #为什么是从2开始？
    y_pred = y_pred[:, 2:, :]  
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

############模型结构############
input_tensor = Input(name='the_input', shape=(img_size[0], img_size[1], num_channels), dtype='float32')
x = input_tensor
base_conv = 32
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
# 参数查看
# conv_shape = x.get_shape().as_list()
# rnn_length = conv_shape[1]
# rnn_dimen = conv_shape[2]*conv_shape[3]
# print(conv_shape, rnn_length, rnn_dimen)
#维度变换
conv_to_rnn_dims = (img_size[0]//(2**3),(img_size[1]//(2**3))*128)
x = Reshape(target_shape=conv_to_rnn_dims,name='reshape')(x)
x =Dense(time_dense_size,activation='relu',name='dense1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# x = Dropout(0.2)(x)
#两层bidirecitonal GRUs
gru_1 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_1')(x)
gru_1b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_1b')(x)
gru1_merged = add([gru_1,gru_1b])
gru_2 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_2')(gru1_merged)
gru_2b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_2b')(gru1_merged)

# transforms RNN output to character activations:  
x = Dense(NUM_CHARS+1,kernel_initializer='he_normal',name='dense2')(concatenate([gru_2,gru_2b]))
x = Activation('softmax',name='softmax')(x)

#打印出模型概况
base_model = Model(inputs=input_tensor, outputs=x)
base_model.summary()
#计算ctc必要参数
pred_length = int(x.shape[1])-2  #为啥会减去2才可以运行？？？
labels = Input(name='the_labels', shape=[label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int32')
label_length = Input(name='label_length', shape=[1], dtype='int32')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])

plot_model(model,to_file=" gru_model.png",show_shapes=True) #show_shapes 带参数显示

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

# 车牌对应的lables
def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label

def parse_line(line):
    parts = line.split('.')
    filename = parts[0]
    label = encode_label(parts[0].strip().upper())
    return filename, label

# 数据生成器
def ImageDataGenerator(img_dir, label_file, batch_size, img_size, input_length, num_channels, label_len):
    num_examples = 0
    next_index = 0
    filenames = []
    labels_all = []
    _input_length = input_length
    with codecs.open(label_file,mode='r', encoding='utf-8') as f:
        for line in f:
            filename, label = parse_line(line)
            filenames.append(filename)
            labels_all.append(label)
            num_examples += 1
    labels_all = np.array(labels_all) #没有必要浮点话？

    while True:
        num_epoches = 0
        # 洗乱数据
        if next_index == 0:
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            filenames = [filenames[i] for i in perm]
            labels_shuttle = labels_all[perm]
        #不洗数据
        # labels_shuttle = labels


        start = next_index
        end = next_index + batch_size
        if end >= num_examples:
            next_index = 0
            num_epoches += 1
            end = num_examples
            batch_size = num_examples - start
        else:
            next_index = end
        images = np.zeros([batch_size, img_size[1], img_size[0], num_channels],dtype=np.uint8)
        # labels = np.zeros([batch_size, label_len],dtype=np.uint8)
        for j, i in enumerate(range(start, end)):
            fname = filenames[i]
            img = cv2.imdecode(np.fromfile(img_dir+fname.strip()+'.jpg', dtype=np.uint8), 1)
            # cv2.imshow('test',img)
            images[j, ...] = img
        # 高与宽转换，便于输入到rnn
        # images1 =  images[0]
        # cv2.imshow('test',images1)
        images = images.transpose(0,2,1,3)
        # images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = labels_shuttle[start:end, ...]
        input_length = np.zeros([batch_size, 1]) #input_length 重新赋值了
        label_length = np.zeros([batch_size, 1])
        input_length[:] = _input_length
        label_length[:] = label_len
        outputs = {'ctc': np.zeros([batch_size])}
        inputs = {'the_input': images,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        yield inputs, outputs

#产生数据
train_gen = ImageDataGenerator(img_dir=ti,
                                 label_file=tl,
                                 batch_size=batch_size,
                                 img_size=img_size,
                                 input_length=pred_length,
                                 num_channels=num_channels,
                                 label_len=label_len)

val_gen = ImageDataGenerator(img_dir=vi,
                                 label_file=vl,
                                 batch_size=batch_size,
                                 img_size=img_size,
                                 input_length=pred_length,
                                 num_channels=num_channels,
                                 label_len=label_len)

#生成器数据测试
# inputs, outputs= next(train_gen)
# title = []
# print(inputs['the_input'].shape,inputs['the_labels'])
# a = inputs['the_labels'].astype(int)
# # title.append(u''.join([CHARS[i] for i in a[0]]))
# for i in range(batch_size):
#     title_one = u''.join([CHARS[j] for j in a[i]])
#     title.append(title_one)
# cv2.imshow('test',inputs['the_input'][0].transpose(1, 0, 2))

# # 模型评估
def evaluate(steps=10):
    batch_acc = 0
    generator = train_gen
    for i in range(steps):
        x_test, y_test = next(generator)
        y_pred = base_model.predict(x_test)
        shape = y_pred[:,2:,:].shape
        ctc_decode = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :label_len]
        if out.shape[1] == label_len:
            batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc / steps

class Evaluator(Callback):
    def __init__(self):
        self.accs = []
    
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(steps=20)*100
        self.accs.append(acc)
        print('')
        print('acc: %f%%' % acc)

evaluator = Evaluator()        
# #该回调函数将在每个epoch后保存模型到路径
# checkpoints_cb = ModelCheckpoint(c, period=1)
# cbs = [checkpoints_cb]

# #tensorboard 
# if dir_log != '':
# tfboard_cb = TensorBoard(log_dir=dir_log, write_images=True)
# cbs.append(tfboard_cb)
import matplotlib.pyplot as plt

h = model.fit_generator(generator=train_gen,
                    steps_per_epoch=100,
                    epochs=20,
                    validation_data=val_gen,
                    validation_steps=20,
                    callbacks=[EarlyStopping(patience=10),evaluator])
                    # callbacks=[EarlyStopping(patience=10)])

# 保存模型  保存权重值
model = Model(inputs=input_tensor, outputs=x)
# model.save(save_name)
model.save_weights('my_model_weight.h5')
print('model saved to {}'.format('my_model_weight.h5'))