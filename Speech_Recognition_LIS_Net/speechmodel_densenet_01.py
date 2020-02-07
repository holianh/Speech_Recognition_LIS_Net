#-*- coding:utf-8 -*-
#author:zhangwei

'''
   此脚本是采用Densenet网络构建端到端声学模型，目前最好的识别效果是32.27%;
'''

from general_function.file_wav import *
from general_function.file_wav import *
from general_function.file_dict import *
from general_function.feature_extract import *
from general_function.edit_distance import *

import keras as kr
import numpy as np
import random

from keras.utils import plot_model
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense , Dropout , Input , Reshape , multiply
from keras.layers import Conv2D , MaxPooling2D , Lambda , Activation , regularizers , AveragePooling2D , concatenate
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD , Adadelta , Adam
from keras.layers.advanced_activations import ELU, LeakyReLU

from readdata_densenet_01 import DataSpeech

#-------------------
def conv2d(size):    return Conv2D(size, (3,3), use_bias=True, activation='relu',        padding='same', kernel_initializer='he_normal')
def norm(x):    return BatchNormalization(axis=-1)(x)
def maxpool(x):    return MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(x)
def dense(units, activation="relu"):    return Dense(units, activation=activation, use_bias=True,        kernel_initializer='he_normal')

# A combination of cnn + cnn + maxpool
# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell(size, x, pool=True):
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)
    return x
#------------------------    
class ModelSpeech():
    def __init__(self , datapath):
        MS_OUTPUT_SIZE = 1422
        k1, k2, k3 = 8, 16, 32
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 360
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.datapath = datapath
        

        self.slash = '/'
        if self.datapath[-1] != self.slash:
            self.datapath = self.datapath + self.slash
         
        self.data = DataSpeech(datapath , 'train')
        self.symbol_len=self.data.get_symbol_num()
        self._model , self.base_model = self.creat_model()
        # self.droprate = 0.3
        
    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)

    def creat_model(self):
        self.inputs = Input(shape=[self.AUDIO_LENGTH , self.AUDIO_FEATURE_LENGTH , 1] , name='Input')
        # self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        # 200 / 8 * 128 = 3200
        self.h6 = Reshape((-1, 3200))(self.h4)
        self.h7 = dense(256)(self.h6)
        self.outputs = dense(self.symbol_len, activation='softmax')(self.h7)
        model_data = Model(inputs=self.inputs, outputs=self.outputs)
        # model_data.summary()
        # plot_model(model_data, '/home/zhangwei/01.png' , show_shapes=True)
        labels = Input(shape=[self.label_max_string_length], name='labels', dtype='float32')
        input_length = Input(shape=[1], name='input_length', dtype='int64')
        label_length = Input(shape=[1], name='label_length', dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=[1, ], name='ctc')([self.outputs , labels, input_length, label_length])
        model = Model(inputs=[self.inputs, labels, input_length, label_length], outputs=loss_out)

        sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        ada_d = Adadelta(lr=0.0005 , rho=0.95, epsilon=1e-6)
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam , loss={'ctc': lambda y_true, y_pred: y_pred})

        print('========================== Model created successfully =================================')
        return model, model_data

    # def dense_block(self , input_tensor , channels):
    #     # bn1 = BatchNormalization()(input_tensor)
    #     # relu = Activation(activation='relu')(bn1)
    #     # conv1 = Conv2D(filters= channels, kernel_size=[1, 1], padding='same' , use_bias=True , kernel_initializer='he_normal')(relu)
    #     bn2 = BatchNormalization()(input_tensor)
    #     relu2 = Activation(activation='relu')(bn2)
    #     conv2 = Conv2D(filters=channels, kernel_size=[3, 3], padding='same' , use_bias=True , kernel_initializer='he_normal')(relu2)
    #     return conv2

    # def dense_block_B(self , input_tensor , channels):
    #     bn1 = BatchNormalization()(input_tensor)
    #     relu = Activation(activation='relu')(bn1)
    #     conv1 = Conv2D(filters=4 * channels, kernel_size=[1, 1], padding='same' , use_bias=True , kernel_initializer='he_normal')(relu)
    #     bn2 = BatchNormalization()(conv1)
    #     relu2 = Activation(activation='relu')(bn2)
    #     conv2 = Conv2D(filters=channels, kernel_size=[3, 3], padding='same' , use_bias=True , kernel_initializer='he_normal')(relu2)
    #     return conv2

    # def transition_layer(self , input_tensor , channels):
    #     bn1 = BatchNormalization()(input_tensor)
    #     relu1 = Activation(activation='relu')(bn1)
    #     conv = Conv2D(filters=channels , kernel_size=[1, 1], padding='same' , use_bias=True , kernel_initializer='he_normal')(relu1)
    #     pool = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv)
    #     pool = Dropout(rate=0.3)(pool)
    #     return pool

    # def ctc_lambda_func(self , args):
    #     y_pred , labels , input_length , label_length = args
    #     y_pred = y_pred[: , : , :]
    #     return K.ctc_batch_cost(y_true=labels , y_pred=y_pred , input_length=input_length , label_length=label_length)

    def train_model(self , datapath , epoch=4 , save_step=2000 , batch_size=1):
        # data = DataSpeech(datapath , 'train')
        num_data = self.data.get_datanum()
        symbolNum=self.data.get_symbol_num()
        yielddatas = self.data.data_generator(batch_size , self.AUDIO_LENGTH)
        for epoch in range(epoch):
            print('[*running] train epoch %d .' % epoch)
            n_step = 0
            while True:
                try:
                    print('[*message] epoch %d , Having training data %d+' % (epoch , n_step * save_step))
                    self._model.fit_generator(yielddatas , save_step)
                    n_step += 1
                except StopIteration:
                    print('======================Error StopIteration==============================')
                    break
                self.save_model(comments='_e_' + str(epoch) + '_step_' + str(n_step * save_step))
                self.test_model(datapath=self.datapath , str_dataset='train' , data_count=4)
                self.test_model(datapath=self.datapath , str_dataset='dev' , data_count=16)

    def load_model(self, filename='model_speech_e_0_step_16000.model'):
        self._model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')

    def test_model(self , datapath='' , str_dataset='dev' , data_count=1):
        data = DataSpeech(self.datapath , str_dataset)
        num_data = data.get_datanum()
        # print num_data
        if data_count <=0 and data_count > num_data:
            data_count = num_data
        try:
            ran_num = random.randint(0 , num_data - 1)
            words_num = 0.
            word_error_num = 0.
            for i in range(data_count):
                data_input , data_labels = data.get_data((ran_num + i) % num_data)
                # print data_input
                num_bias = 0
                while data_input.shape[0] > self.AUDIO_LENGTH:
                    print('[*Error] data input is too long %d' % ((ran_num + i) % num_data))
                    num_bias += 1
                    data_input , data_labels = data.get_data((ran_num + i + num_bias) % num_data)

                pre = self.predict(data_input=data_input , input_len=data_input.shape[0] // 8)                   #1
                words_n = data_labels.shape[0]
                words_num += words_n
                edit_distance = get_edit_distance(data_labels , pre)
                if edit_distance <= words_n:
                    word_error_num += edit_distance
                else:
                    word_error_num += words_n
            # print type(words_num)
            print('[*Test Result] Speech Recognition ' + str_dataset + ' set word error ratio : ' + str(word_error_num / words_num * 100) , '%')
        except StopIteration:
            print('=======================Error StopIteration 01======================')

    def save_model(self , filename='/home/zhangwei/speech_model/speech_model' , comments=''):
        self._model.save_weights(filename + comments + '.model')
        self.base_model.save_weights(filename + comments + '.model.base')
        f = open('steps24.txt' , 'w')
        f.write(filename + comments)
        f.close()

    def predict(self , data_input , input_len):
        batch_size = 1
        in_len = np.zeros((batch_size) , dtype=np.int32)
        in_len[0] = input_len
        x_in = np.zeros(shape=[batch_size , self.AUDIO_LENGTH , self.AUDIO_FEATURE_LENGTH , 1] , dtype=np.float)
        for i in range(batch_size):
            x_in[i , 0 : len(data_input)] = data_input
        base_pred = self.base_model.predict(x=x_in)
        base_pred = base_pred[: , : , :]
        r = K.ctc_decode(base_pred , in_len , greedy=True , beam_width=100 , top_paths=1)
        r1 = K.get_value(r[0][0])
        r1 = r1[0]
        return r1

    def recognize_speech(self , wavsignal , fs):
        data_input = get_frequency_feature(wavsignal , fs)
        input_length = len(data_input)
        input_length = input_length // 8                  #2
        data_input = np.array(data_input , dtype=np.float)
        data_input = data_input.reshape(data_input.shape[0] , data_input.shape[1] , 1)
        r1 = self.predict(data_input , input_length)
        # print r1
        list_symbol_dic = get_list_symbol(self.datapath)
        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])
        return r_str

    def recognize_speech_fromfile(self , filename):
        wavsignal , fs = read_wav_data(filename)
        r = self.recognize_speech(wavsignal , fs)
        return r

    def recognize_speech_pinzhen(self , wavsignal , fs):
        data_input = get_frequency_feature(wavsignal , fs)
        input_length = len(data_input)
        input_length = input_length // 8                                                                 #2
        data_input = np.array(data_input , dtype=np.float)
        data_input = data_input.reshape(data_input.shape[0] , data_input.shape[1] , 1)
        r1 = self.predict(data_input , input_length)
        # print r1
        list_symbol_dic = get_list_symbol(self.datapath)
        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])
        return r_str


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.90
    set_session(tf.Session(config=config))
    
    datapath = 'D:/tact/Code/data_thchs30'
    speech = ModelSpeech(datapath=datapath)
    # speech.creat_model()
    speech.train_model(datapath=datapath)