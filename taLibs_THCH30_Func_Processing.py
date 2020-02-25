# @ title Functions  

Mandarin_label=True
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.fftpack import fft
def startCopy_models(source,dest):
    if not exists(dest): os.makedirs(dest)
    files = glob.glob(os.path.join(dest,"*"))
    for f in files: os.remove(f)            
    for file in glob.glob(os.path.join(source,"*.*")):
        shutil.copy2(file,dest)
#-------------------------------------------------------------------------------
 
def compute_fbank(file):
    x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25 # 单位ms
    window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
    wav_arr = np.array(wavsignal)
    wav_length = len(wavsignal)
    range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype = np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]	
        data_line = data_line * w # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    #data_input = data_input[::]
    return data_input

#-------------------------------------------------------------------------------
from os.path import exists, join
def source_get(source_dataset,dat_type='train'): #'data_thchs30', 'train'
    """return list of .wav, list of trn"""
    wave_files = join(source_dataset,dat_type)
    label_lst = []
    wav_lst = []
    for root, dirs, files in os.walk(wave_files):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                wav_file = os.sep.join([root, file])
                label_file = wav_file + '.trn'
                wav_lst.append(wav_file)
                label_lst.append(label_file)            
    return label_lst, wav_lst 
    
source_dataset = 'data_thchs30'
label_lst_dev_, wav_lst_dev_ = source_get(source_dataset,'dev')
label_lst_test, wav_lst_test = source_get(source_dataset,'test')
label_lst, wav_lst = source_get(source_dataset,'train')
NtrainSamples=len(wav_lst)
print('label_lst=',label_lst[:10])
print('wav_lst=',wav_lst[:10])
print("source_get(), done ----------------------")
#-------------------------------------------------------------------------------
def checkwav_label():
  for i in range(NtrainSamples):
    wavname = (wav_lst[i].split('/')[-1]).split('.')[0]
    labelname = (label_lst[i].split('/')[-1]).split('.')[0]
    if wavname != labelname:
        print('error')
checkwav_label()
print('checkwav_label(), Done! -----------------')
#-------------------------------------------------------------------------------
lineth=1
if Mandarin_label: lineth=0
def read_label(label_file):
    with open(label_file, 'r', encoding='utf8') as f:
        data = f.readlines()
        Vl=data[lineth]
        if Mandarin_label: Vl=Vl.replace(" ",'')
        return Vl
# print('read_label(label_lst[0])=',read_label(label_lst[0]))

def gen_label_data(label_lst):
    label_data = []
    for label_file in label_lst:
        pny = read_label(label_file)
        label_data.append(pny.strip('\n'))
    return label_data

label_data_dev_ = gen_label_data(label_lst_dev_)
label_data_test = gen_label_data(label_lst_test)
label_data = gen_label_data(label_lst)
print('len(label_data)=',len(label_data))
print("gen_label_data(), done! ---------------------")
#-------------------------------------------------------------------------------
def mk_vocab(label_data=None):
    vocab = []
    for line in label_data:
        if not Mandarin_label: line = line.split(' ')
        for pny in line:
            if pny not in vocab:
                vocab.append(pny)
    return vocab

vocab = mk_vocab(label_data=label_data+label_data_test+label_data_dev_)
vocab.append('_')
len_vocab=len(vocab)
print('len(vocab)=',len_vocab)
print('mk_vocab(), done! ---------------------------')
#-------------------------------------------------------------------------------
def word2id(line, vocab):
    if Mandarin_label:
      return [vocab.index(pny) for pny in line]
    else: 
      return [vocab.index(pny) for pny in line.split(' ')]

label_id = word2id(label_data[15], vocab)
print('label_data[15]=',label_data[15])
print('label_id=',label_id)
print('word2id(), OK')
#-------------------------------------------------------------------------------
from random import shuffle
shuffle_list = [i for i in range(NtrainSamples)]
shuffle(shuffle_list)
shuffle_list_dev_ = [i for i in range(len(wav_lst_dev_))]
shuffle_list_test = [i for i in range(len(wav_lst_test))]
 
#-------------------------------------------------------------------------------
def get_batch(batch_size, shuffle_list, wav_lst, label_data, vocab, dataset_size=NtrainSamples):
    for i in range(dataset_size//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            fbank = fbank[:fbank.shape[0] // 8 * 8, :]
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(fbank)
            label_data_lst.append(label)
        yield wav_data_lst, label_data_lst

# batch = get_batch(4, shuffle_list, wav_lst, label_data, vocab)
# #-------------------------------------------------------------------------------
# wav_data_lst, label_data_lst = next(batch)
# # for wav_data in wav_data_lst:    print(wav_data.shape)
# # for label_data in label_data_lst:    print(label_data)
# #-------------------------------------------------------------------------------
# lens = [len(wav) for wav in wav_data_lst]
# print('max(lens)=',max(lens))
# print('lens=',lens)
#---------------------------------------------------------------222222222----------------
def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens

# pad_wav_data_lst, wav_lens = wav_padding(wav_data_lst)
# print('pad_wav_data_lst.shape=',pad_wav_data_lst.shape)
# print('wav_lens=',wav_lens)
#-------------------------------------------------------------------------------
def label_padding(label_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens

# pad_label_data_lst, label_lens = label_padding(label_data_lst)
# print('pad_label_data_lst.shape=',pad_label_data_lst.shape)
# print('label_lens=',label_lens)
#-------------------------------------------------------------------------------
def data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab):
    # for i in range(len(wav_lst)//batch_size):
    i=0
    while(True):
        if i==0: shuffle(shuffle_list)
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)
        pad_wav_data, input_length = wav_padding(wav_data_lst)
        pad_label_data, label_length = label_padding(label_data_lst)
        inputs = {'the_inputs': pad_wav_data,
                  'the_labels': pad_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                 }
        outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)} 
        i= (i+1+batch_size) % batch_size       
        yield inputs, outputs
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------        
total_nums = NtrainSamples
batch_size = 16
batch_num = total_nums // batch_size
epochs = 50
val_batch=4
batch     = data_generator(batch_size,shuffle_list,      wav_lst,      label_data,      vocab)
batch_dev_= data_generator(val_batch, shuffle_list_dev_, wav_lst_dev_, label_data_dev_, vocab)
batch_test= data_generator(val_batch, shuffle_list_test, wav_lst_test, label_data_test, vocab)
print("All OK")
