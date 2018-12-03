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

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

NUM_CHARS = len(CHARS)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    #为什么是从2开始？
    y_pred = y_pred[:, :, :]  
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(width, num_channels):
    input_tensor = Input(name='the_input', shape=(width, 40, num_channels), dtype='float32')
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
    #维度变换
    conv_to_rnn_dims = (img_size[0]//(2**3),(img_size[1]//(2**3))*128)
    x = Reshape(target_shape=conv_to_rnn_dims,name='reshape')(x)
    x =Dense(time_dense_size,activation='relu',name='dense1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #两层bidirecitonal GRUs
    gru_1 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_1')(x)
    gru_1b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_1b')(x)
    gru1_merged = add([gru_1,gru_1b])
    gru_2 = GRU(rnn_size,return_sequences=True,kernel_initializer='he_normal',name='gru_2')(gru1_merged)
    gru_2b = GRU(rnn_size,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='gru_2b')(gru1_merged)

    x = Dense(NUM_CHARS+1,kernel_initializer='he_normal',name='dense2')(concatenate([gru_2,gru_2b]))
    y_pred = Activation('softmax',name='softmax')(x)

    base_model = Model(inputs=input_tensor, outputs=y_pred)
    #打印出模型概况    
    base_model.summary()

    return input_tensor, y_pred

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

class TextImageGenerator:
    def __init__(self, img_dir, label_file, batch_size, img_size, input_length, num_channels, label_len):
        self.img_dir = img_dir
        self.label_file = label_file
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.label_len = label_len
        self.input_len = input_length
        self.img_w, self.img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = None

        self.init()

    def init(self):
        self.labels = []
        with codecs.open(self.label_file,mode='r', encoding='utf-8') as f:
            for line in f:
                filename, label = parse_line(line)
                self.filenames.append(filename)
                self.labels.append(label)
                self._num_examples += 1
        self.labels = np.float32(self.labels)

    def next_batch(self):
        # 洗乱数据
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]
        #  
        batch_size = self.batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end >= self._num_examples:
            self._next_index = 0
            self._num_epoches += 1
            end = self._num_examples
            batch_size = self._num_examples - start
        else:
            self._next_index = end
        images = np.zeros([batch_size, self.img_h, self.img_w, self.num_channels])
        # labels = np.zeros([batch_size, self.label_len])
        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            # img = cv2.imread(os.path.join(self._img_dir, fname)) 不能读取中文
            img = cv2.imdecode(np.fromfile(self.img_dir+fname.strip()+'.jpg', dtype=np.uint8), -1)
            images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        input_length[:] = self.input_len
        label_length[:] = self.label_len
        outputs = {'ctc': np.zeros([batch_size])}
        inputs = {'the_input': images,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        return inputs, outputs

    def get_data(self):
        while True:
            yield self.next_batch()

def export(save_name):
    """Export the model to a hdf5 file
    """
    input_tensor, y_pred = build_model(None, num_channels)
    model = Model(inputs=input_tensor, outputs=y_pred)
    model.save(save_name)
    print('model saved to {}'.format(save_name))

def test_model():
    model = load_model('model_weight.h5')  #选取自己的.h模型名称
    img = cv2.imdecode(np.fromfile(vi+'藏GAC508.jpg', dtype=np.uint8), -1)
    img = np.array(img).reshape(1,128,40,3)
    y_pred = model.predict(img)
    y_pred = y_pred[:,2:,:]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :7]
    out = ''.join([CHARS[x] for x in out[0]])
    print('pred:' + str(out))
    cv2.imshow("Image1", img)
    cv2.waitKey(0)

def main (train_model=True):

    ckpt_dir = os.path.dirname(c)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if (train_model) and not os.path.isdir(dir_log):
        os.makedirs(dir_log)

    input_tensor, y_pred = build_model(img_size[0], num_channels)
    
    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    pred_length = int(y_pred.shape[1])  #为啥会减去2才可以运行？？？
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    train_gen = TextImageGenerator(img_dir=ti,
                                 label_file=tl,
                                 batch_size=batch_size,
                                 img_size=img_size,
                                 input_length=pred_length,
                                 num_channels=num_channels,
                                 label_len=label_len)

    val_gen = TextImageGenerator(img_dir=vi,
                                 label_file=vl,
                                 batch_size=batch_size,
                                 img_size=img_size,
                                 input_length=pred_length,
                                 num_channels=num_channels,
                                 label_len=label_len)

    # #该回调函数将在每个epoch后保存模型到路径
    # checkpoints_cb = ModelCheckpoint(c, period=1)
    # cbs = [checkpoints_cb]

    # #tensorboard 
    # if dir_log != '':
    #     tfboard_cb = TensorBoard(log_dir=dir_log, write_images=True)
    #     cbs.append(tfboard_cb)
    if train_model:
        model.fit_generator(generator=train_gen.get_data(),
                        steps_per_epoch=(train_gen._num_examples+train_gen.batch_size-1) // train_gen.batch_size,
                        epochs=num_epochs,
                        validation_data=val_gen.get_data(),
                        validation_steps=(val_gen._num_examples+val_gen.batch_size-1) // val_gen.batch_size,
                        # callbacks=cbs,
                        initial_epoch=start_of_epoch,
                        callbacks=[EarlyStopping(patience=10)])
        export(save_name)  #保存模型
    else:
        test_model()



if __name__ == '__main__':

    #by jiang
    num_channels = 3
    ti = './car_pic/image/train/' #训练图片目录
    tl = './car_pic/image/train_labels.txt' #训练标签文件
    vi = './car_pic/image/val/'  #验证图片目录
    vl = './car_pic/image/val_labels.txt' #验证标签文件
    img_size = [128,40] #训练图片宽和高
    label_len = 7 #标签长度
    dir_log = './logs/'
    c = './car_pic/image/' #checkpoints format string
    num_epochs = 200     #number of epochs
    start_of_epoch = 0

    #网络参数
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    batch_size = 32

    train_model = False #是否训练模型  True 是

    #同时保存model和权重的方式
    save_name = 'model_weight.h5'
    main(train_model)