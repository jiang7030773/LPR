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
from keras.regularizers import l2  #加入了l2正则化
from keras.utils.vis_utils import plot_model
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

# The actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    #为什么是从2开始？
    y_pred = y_pred[:, 2:, :]  
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model():
    # 超参
    base_conv = 32
    l2_rate = 1e-5
    
    input_tensor = Input(name='the_input', shape=(img_size[0], img_size[1], num_channels), dtype='float32')
    x = input_tensor
    #卷积层1
    for i, n_cnn in enumerate([3,4]):
        for j in range(n_cnn):
            x = Conv2D(base_conv * 2**i, (3,3), padding="same", kernel_initializer='he_uniform', 
                kernel_regularizer=l2(l2_rate))(x)
            x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    # 去掉了最后一个池化层
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
    print(conv_shape,rnn_length,rnn_dimen)

    x = Reshape(target_shape=(rnn_length,rnn_dimen))(x)
    x =Dense(time_dense_size, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
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
    plot_model(base_model,to_file=" gru_model.png",show_shapes=True) #保存模型图片
    base_model.summary()
  
    return input_tensor, y_pred, base_model

def encode_label(s):
    label = np.zeros([len(s[0:7])])
    for i, c in enumerate(s[0:7]):
        label[i] = CHARS_DICT[c]
    return label

def parse_line(line):
    parts = line.split('.')
    filename = parts[0]
    label = encode_label(parts[0].strip().upper())
    return filename, label
# 数据生成器
class TextImageGenerator:
    def __init__(self, img_dir, label_file, batch_size, img_size, input_length, num_channels, label_len):
        self._img_dir = img_dir
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._input_len = input_length
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = None

        self.init()

    def init(self):
        self.labels = []
        with codecs.open(self._label_file,mode='r', encoding='utf-8') as f:
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
        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end >= self._num_examples:
            self._next_index = 0
            self._num_epoches += 1
            end = self._num_examples
            batch_size = self._num_examples - start
        else:
            self._next_index = end
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])
        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            # img = cv2.imread(os.path.join(self._img_dir, fname))
            img = cv2.imdecode(np.fromfile(self._img_dir+fname.strip()+'.jpg', dtype=np.uint8), 1)
            images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        input_length[:] = self._input_len
        label_length[:] = self._label_len
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

def main ():

    ckpt_dir = os.path.dirname(c)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if dir_log != '' and not os.path.isdir(dir_log):
        os.makedirs(dir_log)

    input_tensor, y_pred, base_model = build_model()
    
    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    pred_length = int(y_pred.shape[1]-2)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

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

    #生成器数据测试
    # inputs, outputs= next(train_gen.get_data())
    # print(inputs['the_input'].shape,inputs['the_labels'],inputs['input_length'],inputs['label_length'])
    # a = inputs['the_labels'].astype(int)
    # title = u''.join([CHARS[i] for i in a[0]])
    # cv2.imshow('test',inputs['the_input'][0].transpose(1, 0, 2))


    #该回调函数将在每个epoch后保存模型到路径
    # checkpoints_cb = ModelCheckpoint(c, period=1)
    # cbs = [checkpoints_cb]

    # if dir_log != '':
    #     tfboard_cb = TensorBoard(log_dir=dir_log, write_images=True)
    #     cbs.append(tfboard_cb)
    
    # 模型评估
    def evaluate(steps):
        batch_acc = 0
        generator = train_gen.get_data()
        for i in range(steps):
            x_test, y_test = next(generator)
            y_pred = base_model.predict(x_test)
            shape = y_pred[:,2:,:].shape
            ctc_decode = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0]
            out = K.get_value(ctc_decode)[:, :label_len]
            # print(x_test['the_labels'],out)
            if out.shape[1] == label_len:
                batch_acc += (x_test['the_labels']==out).all(axis=1).mean()
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

    # h = model.fit_generator(generator=train_gen.get_data(),
    #                     steps_per_epoch=(train_gen._num_examples+train_gen._batch_size-1) // train_gen._batch_size,
    #                     epochs=num_epochs,
    #                     validation_data=val_gen.get_data(),
    #                     validation_steps=(val_gen._num_examples+val_gen._batch_size-1) // val_gen._batch_size,
    #                     callbacks=[EarlyStopping(patience=10),evaluator],
    #                     initial_epoch=start_of_epoch)
    
 
    filepath="weights.{epoch:02d-{val_loss:.2f}}.hdf5"

    checkpoints = ModelCheckpoint( filepath,monitor='val_acc',verbose=1,save_best_only=True,
                                 save_weights_only=True,mode='max',period=1)

    history = model.fit_generator(generator=train_gen.get_data(),
                        steps_per_epoch=(train_gen._num_examples+train_gen._batch_size-1) // train_gen._batch_size,
                        epochs=num_epochs,
                        validation_data=val_gen.get_data(),
                        validation_steps=(val_gen._num_examples+val_gen._batch_size-1) // val_gen._batch_size,
                        callbacks=[checkpoints,evaluator],
                        initial_epoch=start_of_epoch)
    
    print(history.history.keys())



    
    
    #保存模型
    base_model.save_weights(('./model/my_model_weights_12.h5')) 

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
    # save_model_path = './model/my_model_weights_12.h5'

    #网络参数
    batch_size = 64
    num_epochs = 3   #number of epochs
    start_of_epoch = 0
    time_dense_size = 64
    rnn_size = 128

    main()