import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras import backend as K
import keras.metrics
import math
import numpy as np
import sys
import keras.utils.np_utils as kutils
import random
import matplotlib.pyplot as plt

sys.setrecursionlimit(1500)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def mcc(y_true, y_pred):
    y_pred_label = K.round(K.clip(y_pred, 0, 1))
    y_true_label = K.round(K.clip(y_true, 0, 1))

    tp = K.sum(y_true_label * y_pred_label)
    fn = K.sum(y_true_label * (1.0 - y_pred_label))
    fp = K.sum((1.0 - y_true_label) * y_pred_label)
    tn = K.sum((1.0 - y_true_label) * (1.0 - y_pred_label))

    numerator = tp * tn - fp * fn
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / denominator


def mcc_loss(y_true, y_pred):
    return K.mean(1 / K.exp(mcc(y_true, y_pred)))


def f1_score(y_true, y_pred):
    y_pred_label = K.round(K.clip(y_pred, 0, 1))
    y_true_label = K.round(K.clip(y_true, 0, 1))

    tp = K.sum(y_true_label * y_pred_label)
    fn = K.sum(y_true_label * (1.0 - y_pred_label))
    fp = K.sum((1.0 - y_true_label) * y_pred_label)

    numerator = 2 * tp
    denominator = 2 * tp + fp + fn

    return numerator / denominator


def f1_loss(y_true, y_pred):
    return K.mean((-1) * K.log(f1_score(y_true, y_pred)))


########## MultiCNN ##########
def MultiCNN(input, Y, batch_size=2048,
             nb_epoch=1000,
             pre_train_seq_path=None,
             pre_train_physical_path=None,
             pre_train_pssm_path=None,
             earlystop=None, transferlayer=1, weights=None, forkinas=False, compiletimes=0,
             class_weights={0: 0.5, 1: 1}, train_time=0,
             compilemodels=None, predict=False):
    ########## Set Oneofkey Network Size and Data ##########

    trainY = kutils.to_categorical(Y)
    # print("trainY:", trainY)
    input_row = input.shape[2]
    input_col = input.shape[3]
    trainX_t = input;

    ########## Set Early_stopping ##########

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    nb_epoch = 500;  # set to a very big value since earlystop used

    ########## TrainX_t For Shape ##########
    trainX_t.shape = (trainX_t.shape[0], input_row, input_col)
    input = Input(shape=(input_row, input_col))

    if compiletimes == 0:

        ########## Total Set Classes ##########
        nb_classes = 2

        ########## Total Set Batch_size ##########
        batch_size = 2048

        ########## Total Set Optimizer ##########
        # optimizer = SGD(lr=0.0001, momentum=0.9, nesterov= True)
        optimization = 'Nadam';

        ########## Begin Oneofkey Network ##########
        x = conv.Convolution1D(101, 1, init='glorot_normal', W_regularizer=l1(0.02), border_mode="same", name='0')(
            input)
        x = Dropout(0.4)(x)
        x = Activation('softsign')(x)

        # x = conv.Convolution1D(308, 3, init='glorot_normal', W_regularizer=l2(0), border_mode="same", name='1')(x)
        # x = Dropout(0.4)(x)
        # x = Activation('softsign')(x)
        #
        # x = conv.Convolution1D(308, 5, init='glorot_normal', W_regularizer=l2(0), border_mode="same", name='2')(x)
        # x = Dropout(0.4)(x)
        # x = Activation('softsign')(x)
        #
        # x = conv.Convolution1D(268, 7, init='glorot_normal', W_regularizer=l2(0), border_mode="same", name='3')(x)
        # x = Activation('softsign')(x)
        # x = Dropout(0.4)(x)

        output_x = core.Flatten()(x)
        output = BatchNormalization()(output_x)
        output = Dropout(0)(output)

        output = Dense(256, init='glorot_normal', activation='relu', name='4')(output)
        output = Dropout(0.298224)(output)
        output = Dense(128, init='glorot_normal', activation="relu", name='5')(output)
        output = Dropout(0)(output)
        output = Dense(128, activation="relu", init='glorot_normal', name='6')(output)
        ########## End Oneofkey Network ##########

        ########## Total Network After Merge ##########
        '''
        output = Dense(512,init='glorot_normal',activation="relu",W_regularizer= l2(0.001),name = '16')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.5)(output)
        output = Dense(256,init='glorot_normal',activation="relu",W_regularizer= l2(0.001),name = '17')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.3)(output)  
        output = Dense(49,init='glorot_normal',activation="relu",W_regularizer= l2(0.001),name = '18')(output)
        output = BatchNormalization()(output)
        output = Dropout(0)(output)
        '''
        out = Dense(nb_classes, init='glorot_normal', activation='softmax', W_regularizer=l2(0.001), name='19')(output)
        ########## Total Network End ##########

        ########## Set Cnn ##########
        cnn = Model([input], out)
        cnn.compile(loss='binary_crossentropy', optimizer=optimization, metrics=[keras.metrics.binary_accuracy])

        ########## Load Models ##########

        if (pre_train_seq_path is not None):
            seq_model = models.load_model(pre_train_seq_path)
            for l in range(0, 6):  # the last layers is not included
                cnn.get_layer(name=str(l)).set_weights(seq_model.get_layer(name=str(l)).get_weights())
                cnn.get_layer(name=str(l)).trainable = False

    else:
        cnn = compilemodels

    ########## Set Class_weight ##########
    # oneofkclass_weights={0 : 0.8 , 1 : 1}
    # physicalclass_weights={0 : 0.3 , 1 : 1}
    # pssmclass_weights={0 : 0.8 , 1 : 1}
    # totalclass_weights={0 : 0.4 , 1 : 1}

    checkpointer = ModelCheckpoint(
        filepath=str(train_time) + '-' + str(class_weights[0]) + '-' + 'merge.h5', verbose=1,
        save_best_only=True, monitor='val_loss', mode='min')
    # weight_checkpointer = ModelCheckpoint(
    #     filepath=str(train_time) + '-' + str(class_weights[0]) + '-' + 'mergeweight.h5', verbose=1,
    #     save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
    fitHistory = cnn.fit(trainX_t, trainY, batch_size=100,
                         nb_epoch=nb_epoch, shuffle=True, validation_split=0.4,
                         callbacks=[early_stopping, checkpointer],
                         class_weight=class_weights)

    plt.plot(fitHistory.history['loss'])
    plt.plot(fitHistory.history['val_loss'])
    plt.title('model loss 20')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    return cnn


def convertSampleToProbMatr(sampleSeq3DArr):  # changed add one column for '1'
    """
    Convertd the raw data to probability matrix

    PARAMETER
    ---------
    sampleSeq3DArr: 3D numpy array
       X denoted the unknow amino acid.


    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """

    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["-"] = 20  ##add -
    AACategoryLen = 21  ##add -

    probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))

    sampleNo = 0
    for sequence in sampleSeq3DArr:

        AANo = 0
        for AA in sequence:

            if not AA in letterDict:
                probMatr[sampleNo][0][AANo] = np.full((1, AACategoryLen), 1.0 / AACategoryLen)

            else:
                index = letterDict[AA]
                probMatr[sampleNo][0][AANo][index] = 1

            AANo += 1
        sampleNo += 1

    return probMatr


seq_list = []


def Cut(Seq, Fix_l, Overlap):
    """
    # -------------------Cut --> Load_file_toNumpy -------------------- #
    :param Seq:
    :param Fix_l:
    :param Overlap:
    :return:
    """
    Sep_str = Seq.strip().replace('\n', '')
    l = len(Sep_str)
    if l > Fix_l:
        seq_list.append(Sep_str[:Fix_l])
        Seq_dg = Sep_str[Fix_l - Overlap:]
        return Cut(Seq_dg, Fix_l, Overlap)
    else:
        seq_list.append(Sep_str + (Fix_l - l) * '-')
        return seq_list


def calculate_length(r_filepath):
    """
    # ----------------1.calculate suitable length to cut sequence------------------ #
    remember!!! file end with \n
    :param r_filepath: read_raw_sequence file
    :return: suitable length
    """
    readpath = open(r_filepath, 'r')
    lines = readpath.readlines()
    m = []
    n = []
    a = []
    for line in lines:
        ll = line.rfind('>')
        a.append(len(line[ll + 1:]))
    count = len(a)
    myset = set(a)
    i = 0
    for item in myset:
        m.append(item)
        n.append(100 * a.count(item) / count)
        i += 1
    local = n.index(max(n))
    # filter values = 0 default
    if m[local] == 0:
        n[local] = 0
    local = n.index((max(n)))
    return m[local]


def Load_file_toNumpy(length, r_filepath):
    """
    #-------------------------2.according to the suitable length ,cut sequence-------------------- #
    # sa is a list of all cut_seq
    # label_list os a list of all label
    :param length:
    :param r_filepath:
    :param w_filepath:
    :return: a list of cut_seq
    """
    sa = []
    readpath = open(r_filepath, 'r')
    lines = readpath.readlines()
    i = 0
    for line in lines:
        """
        # eg: full label = "^4.6.1.16>"  --> label = 4 (type = int)
        """
        ll = line.rfind('>')
        ##################################label = int(line.split('.')[0][1:])##############################
        """
        # length is the cut sequence and 1 is the overlap size
        # cut sequence
        """
        sa = Cut(line[ll + 1:], length, 0)
        h = i
    return sa


if __name__ == "__main__":

    positive_data_a_filepath = '.\data\\new_data_label_sequence.txt'
    negative_data_a_filepath = '.\data\\non_enzyme_new_data_sequence.txt'
    """
    # length is the suitable cut length
    """
    length = calculate_length(positive_data_a_filepath)
    print(length)

    """
    # read raw sequence and cut it
    """
    l_cut_seq = Load_file_toNumpy(length, positive_data_a_filepath)
    h = len(l_cut_seq)

    all_cut_seq = Load_file_toNumpy(length, negative_data_a_filepath)
    label = ["0" for i in range(len(all_cut_seq))]
    for i in range(h):
        label[i] = "1"

    """
    # code sequence 
    """

    index = [i for i in range(len(all_cut_seq))]
    random.shuffle(index)
    shuffled_sample = []
    shuffled_non_sample = []
    for i in index:
        shuffled_sample.append(all_cut_seq[i])
        shuffled_non_sample.append(label[i])
    del all_cut_seq, label

    X = convertSampleToProbMatr(shuffled_sample)
    Y = shuffled_non_sample

    MultiCNN(X, Y)
