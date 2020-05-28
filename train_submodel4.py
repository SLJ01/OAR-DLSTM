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

# trian_submodel4 on the DS1
def data_Preparation_submodel3_DS1():
    look_back = 40
    predict_size = 30
    move_num = 5
    idsum = []
    dataX = []
    dataY = []
    conn = pymysql.connect(host='10.11.112.202',
                           port=3306,
                           user='user',
                           password='123456',
                           db='SHUINI')
    sql = "select ID,TIMESTAMP,L0069,L0002,L0005,L0008,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167 from DB_JIANGYIN where id timestamp between '1575129651000' and '1577548791000'"  # 12.1-12.28
    slj = pd.read_sql(sql, conn)
    slj = slj.sort_values(by='ID', axis=0, ascending=True)
    slj_arr = np.array(slj)
    temp_167 = slj_arr[0:1, 10:11][0][0]
    temp_24 = slj_arr[0:1, 11:12][0][0]
    id_init = []
    for i in range(0, len(slj['ID'])):
        l0167 = slj_arr[i:i + 1, 10:11][0][0]
        l0024 = slj_arr[i:i + 1, 11:12][0][0]
        id = slj_arr[i:i + 1, 0:1][0][0]
        if ((l0167 != temp_167) or (l0024 != temp_24)):
            temp_167 = l0167
            temp_24 = l0024
            id_init.append(id)
    id_init2 = []
    for x in range(len(id_init)):
        sql2 = 'select ID,TIMESTAMP,L0024,L0167 from DB_JIANGYIN  where id between %s and %s' % (
            id_init[x] - 6, id_init[x])  # 在6个点以内又发生了第二次调控，则认为关联调控
        data0 = pd.read_sql(sql2, conn)
        data0_arr = np.array(data0)
        init_24 = data0_arr[0, 2]
        init_167 = data0_arr[0, 3]
        last_24 = data0_arr[6, 2]
        last_167 = data0_arr[6, 3]
        last2_24 = data0_arr[5, 2]
        last2_167 = data0_arr[5, 3]
        if ((last_167 != last2_167) and (last_24 == last2_24)):
            if (last_24 != init_24):
                id_init2.append(id_init[x])
        if ((last_167 == last2_167) and (last_24 != last2_24)):
            if (last_167 != init_167):
                id_init2.append(id_init[x])
        if ((last_167 != last2_167) and (last_24 != last2_24)):
            id_init2.append(id_init[x])

    id_last = []
    for n in range(len(id_init2) - 1):
        sql3 = 'select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN  where id between %s and %s' % (
            id_init2[n] - 39, id_init2[n] + 55)  # 空5个，标签30个，移动50个，共55；
        data1 = pd.read_sql(sql3, conn)
        data1_arr = np.array(data1)
        # print('id_start',id_init[n]-20,'id_stop',id_init[n]+70)
        data1_arr = np.c_[data1_arr, (data1_arr[:, 2] + data1_arr[:, 3] + data1_arr[:, 4])]
        temp_33 = data1_arr[look_back - 1:look_back, 12:13][0][0]
        temp_167 = data1_arr[look_back - 1:look_back, 27:28][0][0]
        temp_95 = data1_arr[look_back - 1:look_back, 19:20][0][0]
        temp_24 = data1_arr[look_back - 1:look_back, 8:9][0][0]
        temp_22 = data1_arr[look_back - 1:look_back, 6:7][0][0]

        last_33 = data1_arr[94:95, 12:13][0][0]
        last_167 = data1_arr[94:95, 27:28][0][0]
        last_95 = data1_arr[94:95, 19:20][0][0]
        last_24 = data1_arr[94:95, 8:9][0][0]
        last_22 = data1_arr[94:95, 6:7][0][0]
        if ((temp_33 == last_33) and (temp_167 == last_167) and (temp_95 == last_95) and (temp_24 == last_24) and (
                temp_22 == last_22)):
            id_last.append(id_init2[n])
            data1_values = data1_arr[:, 5:].astype('float32')
            for x in range(move_num):
                a = data1_values[x:x + look_back]
                b = data1_values[x + look_back + 5:x + look_back + predict_size + 5, 0]  # 中间空10个点
                l0069_newest = a[look_back - 1, 0]
                if (120 > np.min(b) > 80):
                    if (len(b) == predict_size):
                        dataX.append(a)
                        b2 = b - l0069_newest
                        dataY.append(b2)

    return (dataX, dataY)
def data_test_DS1():
    conn = pymysql.connect(host='10.11.112.202',
                           port=3306,
                           user='user',
                           password='123456',
                           db='SHUINI')
    sql = 'select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN  where timestamp between 1575129651000 and 1577548791000'
    cxy = pd.read_sql(sql, conn)
    cxy1 = cxy.sort_values(by='ID', axis=0, ascending=True)
    cxy1_array = np.array(cxy1)
    cxy1_array = np.c_[cxy1_array, (cxy1_array[:, 2] + cxy1_array[:, 3] + cxy1_array[:, 4])]
    cxy1_array=cxy1[:len(cxy1_array)*0.2]
    cxy1 = pd.DataFrame(cxy1_array)
    dataset2 = cxy1.values
    dataset4 = dataset2[:, 5:].astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset5 = scaler.fit_transform(dataset4)
    dataX, dataY = [], []
    look_back = 40
    predict_size = 30
    for i in range(0, len(dataset5) - look_back - predict_size, look_back):
        a = dataset5[i:(i + look_back)]
        l0069_newset=a[look_back-1,0]
        b = dataset5[i + look_back:i + look_back + predict_size, 0]
        b2=b-l0069_newset
        dataX.append(a)
        dataY.append(b2)
    y_min=np.min(dataY)
    y_max=np.max(dataY)
    d=y_max-y_min
    return np.array(dataX), np.array(dataY),d,y_min
def train_sub_model4_DS1():
    look_back = 40
    predict_size = 30
    trainX, trainY = data_Preparation_submodel3_DS1()
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    y_max = np.max(trainY)
    y_min = np.min(trainY)
    d = y_max - y_min
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainX_normal = scaler.fit_transform(
        np.reshape(trainX, (trainX.shape[0] * trainX.shape[1], trainX.shape[2])))
    my_imputer=Imputer()
    trainX_normal=my_imputer.fit_transform(trainX_normal)
    trainY_normal = scaler.fit_transform(trainY)
    trainX = np.reshape(trainX_normal, (trainX.shape[0], look_back, trainX.shape[2]))

    testX, testY, d_test, y_min_test = data_test_DS1()
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))

    def baseline_model():
        # create model
        model = Sequential()
        model.add(LSTM(units=32, input_shape=(look_back, trainX.shape[2]), return_sequences=True))
        model.add(LSTM(16, return_sequences=False))
        model.add(LSTM(4, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=predict_size, activation='relu'))
        adam = optimizers.adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        return model

    model = baseline_model()
    history = model.fit(trainX, trainY, batch_size=80,epochs=550, verbose=2, validation_split=0.03)#batch_size=32
    model.save('model4_DS1.h5')

    # Iterative training error diagram
    plt.plot(history.history['loss'])
    plt.title('model train  loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()

    model = load_model('model4_DS1.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainY1 = trainY_normal.reshape(-1, 1)
    testY1 = testY.reshape(-1, 1)
    trainPredict1 = trainPredict.reshape(-1, 1)
    testPredict1 = testPredict.reshape(-1, 1)

    trainPredict = trainPredict1 * d + y_min
    testPredict = testPredict1 * d_test + y_min_test
    print('testpredict', testPredict)
    testY2 = testY1 * d_test + y_min_test
    print('testY2', testY2)
    trainY2 = trainY1 * d + y_min
    error_abslisttrain = []
    for i in range(len(trainPredict[:, 0])):
        error_abs = np.abs(trainY2[i, 0] - trainPredict[i, 0])
        error_abslisttrain.append(error_abs)
    error_abslisttest = []
    for i in range(len(testPredict[:, 0])):
        error_abs_test = np.abs(testY2[i, 0] - testPredict[i, 0])
        error_abslisttest.append(error_abs_test)

    # Ploting the training error and test error
    plt.plot(error_abslisttrain)
    plt.plot(error_abslisttest)
    plt.show()

    # ploting the real time value of the training prediction and the real time value of the test prediction
    plt.plot(trainY2)
    plt.plot(trainPredict)
    plt.show()
    plt.plot(testY2)
    plt.plot(testPredict)
    plt.show()

# trian_submodel4 on the DS2
def data_Preparation_submodel3_DS2():
    look_back = 40
    predict_size = 30
    move_num = 5
    idsum = []
    dataX = []
    dataY = []
    conn = pymysql.connect(host='10.11.112.202',
                           port=3306,
                           user='user',
                           password='123456',
                           db='SHUINI')
    sql = "select ID,TIMESTAMP,L0069,L0002,L0005,L0008,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167 from DB_JIANGYIN where id timestamp between '1573315251000' and '1574524791000'"  #
    slj = pd.read_sql(sql, conn)
    slj = slj.sort_values(by='ID', axis=0, ascending=True)
    slj_arr = np.array(slj)
    temp_167 = slj_arr[0:1, 10:11][0][0]
    temp_24 = slj_arr[0:1, 11:12][0][0]
    id_init = []
    for i in range(0, len(slj['ID'])):
        l0167 = slj_arr[i:i + 1, 10:11][0][0]
        l0024 = slj_arr[i:i + 1, 11:12][0][0]
        id = slj_arr[i:i + 1, 0:1][0][0]
        if ((l0167 != temp_167) or (l0024 != temp_24)):
            temp_167 = l0167
            temp_24 = l0024
            id_init.append(id)
    id_init2 = []
    for x in range(len(id_init)):
        sql2 = 'select ID,TIMESTAMP,L0024,L0167 from DB_JIANGYIN  where id between %s and %s' % (
            id_init[x] - 6, id_init[x])  # 在6个点以内又发生了第二次调控，则认为关联调控
        data0 = pd.read_sql(sql2, conn)
        data0_arr = np.array(data0)
        init_24 = data0_arr[0, 2]
        init_167 = data0_arr[0, 3]
        last_24 = data0_arr[6, 2]
        last_167 = data0_arr[6, 3]
        last2_24 = data0_arr[5, 2]
        last2_167 = data0_arr[5, 3]
        if ((last_167 != last2_167) and (last_24 == last2_24)):
            if (last_24 != init_24):
                id_init2.append(id_init[x])
        if ((last_167 == last2_167) and (last_24 != last2_24)):
            if (last_167 != init_167):
                id_init2.append(id_init[x])
        if ((last_167 != last2_167) and (last_24 != last2_24)):
            id_init2.append(id_init[x])

    id_last = []
    for n in range(len(id_init2) - 1):
        sql3 = 'select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN  where id between %s and %s' % (
            id_init2[n] - 39, id_init2[n] + 55)  # 空5个，标签30个，移动50个，共55；
        data1 = pd.read_sql(sql3, conn)
        data1_arr = np.array(data1)
        # print('id_start',id_init[n]-20,'id_stop',id_init[n]+70)
        data1_arr = np.c_[data1_arr, (data1_arr[:, 2] + data1_arr[:, 3] + data1_arr[:, 4])]
        temp_33 = data1_arr[look_back - 1:look_back, 12:13][0][0]
        temp_167 = data1_arr[look_back - 1:look_back, 27:28][0][0]
        temp_95 = data1_arr[look_back - 1:look_back, 19:20][0][0]
        temp_24 = data1_arr[look_back - 1:look_back, 8:9][0][0]
        temp_22 = data1_arr[look_back - 1:look_back, 6:7][0][0]

        last_33 = data1_arr[94:95, 12:13][0][0]
        last_167 = data1_arr[94:95, 27:28][0][0]
        last_95 = data1_arr[94:95, 19:20][0][0]
        last_24 = data1_arr[94:95, 8:9][0][0]
        last_22 = data1_arr[94:95, 6:7][0][0]
        if ((temp_33 == last_33) and (temp_167 == last_167) and (temp_95 == last_95) and (temp_24 == last_24) and (
                temp_22 == last_22)):
            id_last.append(id_init2[n])
            data1_values = data1_arr[:, 5:].astype('float32')
            for x in range(move_num):
                a = data1_values[x:x + look_back]
                b = data1_values[x + look_back + 5:x + look_back + predict_size + 5, 0]  # 中间空10个点
                l0069_newest = a[look_back - 1, 0]
                if (120 > np.min(b) > 80):
                    if (len(b) == predict_size):
                        dataX.append(a)
                        b2 = b - l0069_newest
                        dataY.append(b2)

    return (dataX, dataY)

def data_test_DS2():
    conn = pymysql.connect(host='10.11.112.202',
                           port=3306,
                           user='user',
                           password='123456',
                           db='SHUINI')
    sql = 'select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN  where timestamp between 1575129651000 and 1577548791000'
    cxy = pd.read_sql(sql, conn)
    cxy1 = cxy.sort_values(by='ID', axis=0, ascending=True)
    cxy1_array = np.array(cxy1)
    cxy1_array = np.c_[cxy1_array, (cxy1_array[:, 2] + cxy1_array[:, 3] + cxy1_array[:, 4])]
    cxy1_array=cxy1[:len(cxy1_array)*0.2]
    cxy1 = pd.DataFrame(cxy1_array)
    dataset2 = cxy1.values
    dataset4 = dataset2[:, 5:].astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset5 = scaler.fit_transform(dataset4)
    dataX, dataY = [], []
    look_back = 40
    predict_size = 30
    for i in range(0, len(dataset5) - look_back - predict_size, look_back):
        a = dataset5[i:(i + look_back)]
        l0069_newset=a[look_back-1,0]
        b = dataset5[i + look_back:i + look_back + predict_size, 0]
        b2=b-l0069_newset
        dataX.append(a)
        dataY.append(b2)
    y_min=np.min(dataY)
    y_max=np.max(dataY)
    d=y_max-y_min
    return np.array(dataX), np.array(dataY),d,y_min
def train_sub_model4_DS2():
    look_back = 40
    predict_size = 30
    trainX, trainY = data_Preparation_submodel3_DS1()
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    y_max = np.max(trainY)
    y_min = np.min(trainY)
    d = y_max - y_min
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainX_normal = scaler.fit_transform(
        np.reshape(trainX, (trainX.shape[0] * trainX.shape[1], trainX.shape[2])))
    my_imputer=Imputer()
    trainX_normal=my_imputer.fit_transform(trainX_normal)
    trainY_normal = scaler.fit_transform(trainY)
    trainX = np.reshape(trainX_normal, (trainX.shape[0], look_back, trainX.shape[2]))

    testX, testY, d_test, y_min_test = data_test_DS1()
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))

    def baseline_model():
        # create model
        model = Sequential()
        model.add(LSTM(units=32, input_shape=(look_back, trainX.shape[2]), return_sequences=True))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=predict_size, activation='relu'))
        adam = optimizers.adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        return model

    model = baseline_model()
    history = model.fit(trainX, trainY, batch_size=80,epochs=1050, verbose=2, validation_split=0.03)
    model.save('model4_DS2.h5')

    # Iterative training error diagram
    plt.plot(history.history['loss'])
    plt.title('model train  loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()

    model = load_model('model4_DS2.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainY1 = trainY_normal.reshape(-1, 1)
    testY1 = testY.reshape(-1, 1)
    trainPredict1 = trainPredict.reshape(-1, 1)
    testPredict1 = testPredict.reshape(-1, 1)

    trainPredict = trainPredict1 * d + y_min
    testPredict = testPredict1 * d_test + y_min_test
    print('testpredict', testPredict)
    testY2 = testY1 * d_test + y_min_test
    print('testY2', testY2)
    trainY2 = trainY1 * d + y_min
    error_abslisttrain = []
    for i in range(len(trainPredict[:, 0])):
        error_abs = np.abs(trainY2[i, 0] - trainPredict[i, 0])
        error_abslisttrain.append(error_abs)
    error_abslisttest = []
    for i in range(len(testPredict[:, 0])):
        error_abs_test = np.abs(testY2[i, 0] - testPredict[i, 0])
        error_abslisttest.append(error_abs_test)

    # Ploting the training error and test error
    plt.plot(error_abslisttrain)
    plt.plot(error_abslisttest)
    plt.show()

    # ploting the real time value of the training prediction and the real time value of the test prediction
    plt.plot(trainY2)
    plt.plot(trainPredict)
    plt.show()
    plt.plot(testY2)
    plt.plot(testPredict)
    plt.show()


if __name__ == '__main__':
    train_sub_model4_DS1()
    train_sub_model4_DS2()