# %tensorflow_version 1.x
import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Reshape, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model

from keras.models import Model, load_model
from keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax
from keras.layers import Add, Dropout, BatchNormalization, Conv2D, Reshape, MaxPooling2D, Dense, CuDNNLSTM, Bidirectional
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import optimizers
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
from tensorflow import reshape, squeeze
from keras.initializers import random_normal
from keras.layers import TimeDistributed, LSTM
from keras.activations import relu


# -** Add CTC loss function, introduced by backend **
# ** Note: CTC_batch_cost input is: **
# -** labels ** labels: [batch_size, l]
# -** y_pred ** Output from cnn network: [batch_size, t, vocab_size]
# -** input_length ** The length of the network output: [batch_size]
# -** label_length ** The length of the label: [batch_size]
def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def clipped_relu(x):
    return relu(x, max_value=20)

thisModel_Name="ASR05_DeepSpeech1"
class ASR05_DeepSpeech1():
    """docstring for Amodel."""
    def __init__(self, vocab_size):
        super(ASR05_DeepSpeech1, self).__init__()
        self.vocab_size = vocab_size
        self.ds1()
        self._ctc_init()
        self.opt_init()
        self.thisModel_Name="ASR05_DeepSpeech1"
        self.loss=[]
        self.minloss=100000

    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)

    def opt_init(self):
        opt = Adam(lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
        #self.ctc_model=multi_gpu_model(self.ctc_model,gpus=2)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)
    def model_name(self):
        print("ASR05_DeepSpeech1")
        
    def ds1(self,input_dim=200, fc_size=1024, rnn_size=1024): 
        # self.inputs = Input(name='the_inputs', shape=(None, input_dim, 1))
        # self.h1 = cnn_cell(32, self.inputs)
        # self.h2 = cnn_cell(64, self.h1)
        # self.h3 = cnn_cell(128, self.h2)
        # self.h4 = cnn_cell(128, self.h3, pool=False)
        # # 200 / 8 * 128 = 3200
        # self.h6 = Reshape((-1, 3200))(self.h4)
        # self.h7 = dense(256)(self.h6)
        # self.outputs = dense(self.vocab_size, activation='softmax')(self.h7)
        # self.model = Model(inputs=self.inputs, outputs=self.outputs)
        
        # -------
        
        """ DeepSpeech 1 Implementation without dropout

        Architecture:
            Input MFCC TIMEx26
            3 Fully Connected using Clipped Relu activation function
            1 BiDirectional LSTM
            1 Fully connected Softmax

        Details:
            - Removed Dropout on this implementation
            - Uses MFCC's rather paper's 80 linear spaced log filterbanks
            - Uses LSTM's rather than SimpleRNN
            - No translation of raw audio by 5ms
            - No stride the RNN

        References:
            https://arxiv.org/abs/1412.5567
        """
        # hack to get clipped_relu to work on bidir layer
        from keras.utils.generic_utils import get_custom_objects
        get_custom_objects().update({"clipped_relu": clipped_relu})

        self.inputs = Input(name='the_inputs', shape=(None, input_dim, 1))
        # input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, 778, 26)
        x = Lambda(lambda x: squeeze(x, axis=-1))(self.inputs)

        init = random_normal(stddev=0.046875)

        # First 3 FC layers
        x = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)
        x = TimeDistributed(Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)
        x = TimeDistributed(Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)  # >>(?, 778, 2048)


        # # Layer 4 BiDirectional RNN - note coreml only supports LSTM BIDIR
        x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation=clipped_relu,
                                    kernel_initializer='glorot_uniform', name='birnn'), merge_mode='sum')(x)  #

        # Layer 5+6 Time Dist Layer & Softmax

        # x = TimeDistributed(Dense(fc_size, activation=clipped_relu))(x)
        self.outputs = TimeDistributed(Dense(self.vocab_size, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"), name="out")(x)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        
# from IPython.display import clear_output
# # am = ASR05_DeepSpeech1(len_vocab) 
# am = ASR05_DeepSpeech1(2884) 
# clear_output()
# am.ctc_model.summary()
