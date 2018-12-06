from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_io
from keras.layers import Input, Dense, Activation, Conv2D, Reshape
from keras.layers import BatchNormalization, Lambda, MaxPooling2D, Dropout
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping,Callback
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

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
    x = Dense(66,kernel_initializer='he_normal',name='dense2')(concatenate([gru_2,gru_2b]))
    y_pred = Activation('softmax',name='softmax')(x)
    base_model =  Model(inputs=input_tensor, outputs=y_pred)
    base_model.load_weights(model_path)
    return base_model

 
"""----------------------------------配置路径-----------------------------------"""
epochs=20
# h5_model_path='./my_model_ep{}.h5'.format(epochs)
h5_model_path='./model/my_model_weights.h5'
output_path='.'
# pb_model_name='my_model_ep{}.pb'.format(epochs)
pb_model_name='./model/my_model_weights.pb'
 
 
"""----------------------------------导入keras模型------------------------------"""
K.set_learning_phase(0)
net_model = build_model(h5_model_path)
 
print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)
 
"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)
