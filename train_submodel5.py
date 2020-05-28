import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,SimpleRNN
from keras.layers import Dense,Reshape,Dropout
from keras import optimizers
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import Imputer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# config = tf.ConfigProto()#tf1
config=tf.compat.v1.ConfigProto()#tf2
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = False
sess = tf.compat.v1.Session(config = config)#tf2
# sess = tf.Session(config = config)#tf1

def train_submodel5_DS1():
    predict_size=30
    look_back=40
    dataX=[]
    dataY=[]
    conn = pymysql.connect(host='10.11.112.202',
                           port=3306,
                           user='user',
                           password='123456',
                           db='SHUINI')
    # sql = "select ID,TIMESTAMP,L0069,L0033,L0034,L0097,L0095,L0096,L0162,L0158 ,l0167,L0024,L0022,L0023 from DB_JIANGYIN where L0069 between 80 and 120  order by ID desc limit 12960"  # 2592000
    sql = "select ID,TIMESTAMP,L0069,L0002,L0005,L0008,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167 from DB_JIANGYIN where id timestamp between '1575129651000' and '1577548791000'"  # 12.1-12.28
    slj = pd.read_sql(sql, conn)
    slj = slj.sort_values(by='ID', axis=0, ascending=True)
    slj_arr = np.array(slj)
    data1_arr = np.c_[slj_arr, (slj_arr[:, 2] + slj_arr[:, 3] + slj_arr[:, 4])]
    data1_values = data1_arr[:, 5:].astype('float32')
    for x in range(len(data1_values)-predict_size-look_back-1):
        a=data1_values[x:x+look_back]
        b=data1_values[x+look_back:x+look_back+predict_size,0]
        l0069_newest=a[look_back-1,0]
        if((120>np.min(b)>80) and (120>np.min(a[:,0])>80)):
            dataX.append(a)
            b2=b-l0069_newest
            dataY.append(b)
    dataX=np.array(dataX)
    dataY=np.array(dataY)
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainX_normal = scaler.fit_transform(
        np.reshape(dataX, (dataX.shape[0] * dataX.shape[1], dataX.shape[2])))
    my_imputer = Imputer()
    trainX_normal = my_imputer.fit_transform(trainX_normal)
    trainY2=np.reshape(dataY,(-1,1))
    trainY_normal=[]
    for y in trainY2:
        trainY_normal.append((1.0/(1+np.exp(-float(y-100)))))
    trainY_normal=np.reshape(trainY_normal,(-1,dataY.shape[1]))
    trainX = np.reshape(trainX_normal, (dataX.shape[0], look_back, dataX.shape[2]))
    def baseline_model():
        # create model
        model = Sequential()
        model.add(LSTM(units=64, input_shape=(look_back, trainX.shape[2]), return_sequences=True))
        model.add(LSTM(32, return_sequences=False))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=predict_size, activation='relu'))
        adam = optimizers.adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        return model

    model = baseline_model()
    history = model.fit(trainX, trainY_normal, batch_size=128, epochs=1000, verbose=2,
                        validation_split=0.03)  # batch_size=32
    model.save( 'sub_model5_DS1.h5')
    # Iterative training error diagram
    plt.plot(history.history['loss'])
    plt.title('model train  loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()

def train_submodel5_DS2():
    predict_size=30
    look_back=40
    dataX=[]
    dataY=[]
    conn = pymysql.connect(host='10.11.112.202',
                           port=3306,
                           user='user',
                           password='123456',
                           db='SHUINI')

    sql = "select ID,TIMESTAMP,L0069,L0002,L0005,L0008,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167 from DB_JIANGYIN where id timestamp between 1575129651000 and 1577548791000"  #
    slj = pd.read_sql(sql, conn)
    slj = slj.sort_values(by='ID', axis=0, ascending=True)
    slj_arr = np.array(slj)
    data1_arr = np.c_[slj_arr, (slj_arr[:, 2] + slj_arr[:, 3] + slj_arr[:, 4])]
    data1_values = data1_arr[:, 5:].astype('float32')
    for x in range(len(data1_values)-predict_size-look_back-1):
        a=data1_values[x:x+look_back]
        b=data1_values[x+look_back:x+look_back+predict_size,0]
        l0069_newest=a[look_back-1,0]
        if((120>np.min(b)>80) and (120>np.min(a[:,0])>80)):
            dataX.append(a)
            b2=b-l0069_newest
            dataY.append(b)
    dataX=np.array(dataX)
    dataY=np.array(dataY)
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainX_normal = scaler.fit_transform(
        np.reshape(dataX, (dataX.shape[0] * dataX.shape[1], dataX.shape[2])))
    my_imputer = Imputer()
    trainX_normal = my_imputer.fit_transform(trainX_normal)
    trainY2=np.reshape(dataY,(-1,1))
    trainY_normal=[]
    for y in trainY2:
        trainY_normal.append((1.0/(1+np.exp(-float(y-100)))))
    trainY_normal=np.reshape(trainY_normal,(-1,dataY.shape[1]))
    trainX = np.reshape(trainX_normal, (dataX.shape[0], look_back, dataX.shape[2]))
    def baseline_model():
        # create model
        model = Sequential()
        model.add(LSTM(units=64, input_shape=(look_back, trainX.shape[2]), return_sequences=True))
        model.add(LSTM(32, return_sequences=False))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=predict_size, activation='relu'))
        adam = optimizers.adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        return model

    model = baseline_model()
    history = model.fit(trainX, trainY_normal, batch_size=128, epochs=1000, verbose=2,
                        validation_split=0.03)  # batch_size=32
    model.save( 'sub_model5_DS2.h5')
    # Iterative training error diagram
    plt.plot(history.history['loss'])
    plt.title('model train  loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    train_submodel5_DS1()
    train_submodel5_DS2()