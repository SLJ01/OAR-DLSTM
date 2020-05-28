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


def predict_DS1():
    db2 = pymysql.connect(host='10.11.112.202', port=3306, user='user', password='123456', db='SHUINI', charset='utf8')
    cursor = db2.cursor()
    model1 = load_model('model1_DS1')
    model2 = load_model('model2_DS1')
    model3 = load_model('model3_DS1')
    model4 = load_model('model4_DS1')
    model5 = load_model('model5_DS1')
    while (True):
        conn = pymysql.connect(host='10.11.112.202',
                               port=3306,
                               user='user',
                               password='123456',
                               db='SHUINI')

        sql2 = "select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013,l0014,L0070 from DB_JIANGYIN order by id desc limit 40"
        slj2 = pd.read_sql(sql2, conn)
        slj1 = slj2.sort_values(by='ID', axis=0, ascending=True)
        data2_arr = np.array(slj1)

        timestamp_newset = slj2.loc[0, 'TIMESTAMP']
        id_newset = slj2.loc[0, 'ID']
        l0069_newset = slj2.loc[0, 'L0069']
        last2_33 = data2_arr[38:39, 12:13][0][0]
        last2_167 = data2_arr[38:39, 27:28][0][0]
        last2_95 = data2_arr[38:39, 19:20][0][0]
        last2_24 = data2_arr[38:39, 8:9][0][0]
        last2_22 = data2_arr[38:39, 6:7][0][0]

        last_33 = data2_arr[39:40, 12:13][0][0]
        last_167 = data2_arr[39:40, 27:28][0][0]
        last_95 = data2_arr[39:40, 19:20][0][0]
        last_24 = data2_arr[39:40, 8:9][0][0]
        last_22 = data2_arr[39:40, 6:7][0][0]
        if ((last2_33 != last_33) or (last2_167 != last_167) or (last2_95 != last_95) or ((last2_24 != last_24))):
            n = 0
            id_init = 0
            print('startmodel1')
            while (n < 200):
                conn = pymysql.connect(host='10.11.112.202',
                                       port=3306,
                                       user='user',
                                       password='123456',
                                       db='SHUINI')
                sql2 = "select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN order by id desc limit 40"
                slj2 = pd.read_sql(sql2, conn)
                slj1 = slj2.sort_values(by='ID', axis=0, ascending=True)
                data2_arr = np.array(slj1)
                data_l0033 = data2_arr
                data2_l0033 = np.c_[
                    data_l0033, (data_l0033[:, 2] + data_l0033[:, 3] + data_l0033[:, 4])]
                last2_33 = data_l0033[38:39, 12:13][0][0]
                last2_167 = data_l0033[38:39, 27:28][0][0]
                last2_95 = data_l0033[38:39, 19:20][0][0]
                last2_24 = data_l0033[38:39, 8:9][0][0]
                last2_22 = data_l0033[38:39, 6:7][0][0]

                last_33 = data_l0033[39:40, 12:13][0][0]
                last_167 = data_l0033[39:40, 27:28][0][0]
                last_95 = data_l0033[39:40, 19:20][0][0]
                last_24 = data_l0033[39:40, 8:9][0][0]
                last_22 = data_l0033[39:40, 6:7][0][0]

                if (1):

                    data2_l0033 = data2_l0033[:, 5:]
                    timestamp_newset = slj2.loc[0, 'TIMESTAMP']
                    id_newset = slj2.loc[0, 'ID']
                    l0069_newset = slj2.loc[0, 'L0069']
                    if (id_init != id_newset):
                        id_init = id_newset
                        n = n + 1

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data3_l0033 = scaler.fit_transform(data2_l0033)
                    X_data = data3_l0033.reshape(1, data3_l0033.shape[0], data3_l0033.shape[1])
                    predictions_1 = model1.predict(X_data)
                    predictions1 = predictions_1 + l0069_newset
                    l0069_arr = np.array(data2_arr[:, 5], dtype='float')
                    l0069_arr2 = np.array(data2_arr[20:, 5], dtype='float')

                    x_axis = np.arange(1, 21, 1)
                    x_axis_pre = np.arange(21, 52, 1)
                    z1 = np.polyfit(x_axis, l0069_arr2, 2)
                    p1 = np.poly1d(z1)
                    y_pre = p1(x_axis_pre)
                    row = ['P015', 'P016', 'P017', 'P018', 'P019', 'P020', 'P021', 'P022', 'P023', 'P024', 'P025',
                           'P026', 'P027', 'P028', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036'
                        , 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047',
                           'P048', 'P049', 'P050']

                    # #注入predict_L0069里面：
                    # for m in range(16):
                    #     T2 = int(timestamp_newset) + 2000 * (m + 1+14)
                    #     T2_arr = np.array(T2)  # sun
                    #     ID2 = int(id_newset) + (m + 1+14)
                    #     ID_arr = np.array(ID2)  # sun
                    #     y_pre3=0.8*y_pre[14+m]+0.2*predictions1[:, 9+m]
                    #     predict_str = str(y_pre3).strip('[' + ']')
                    #     # print('predict_str',predict_str)
                    #     predict_str = np.array(predict_str)
                    #     conn = pymysql.connect(host='10.11.112.202',port=3306,user='user',password='123456',db='SHUINI')
                    #     sql1 = "insert into predict_L0069(ID,%s) values(%s,%s) ON DUPLICATE KEY UPDATE %s=%s" % (row[m],
                    #         ID_arr, predict_str, row[m],predict_str)
                    #     cursor.execute(sql1)
                    #     db2.commit()
                    #     # break
                    #     sql3 = "select ID,L0069 from DB_JIANGYIN order by id desc limit 1"
                    #     actual = pd.read_sql(sql3, conn)
                    #     id_actual = actual.loc[0, 'ID']
                    #     # print('actual',actual)
                    #     l0069_actual = actual.loc[0,'L0069']
                    #     sql4="update predict_L0069 set L0069_actual=%s where id=%s"%(l0069_actual.astype('float'),id_actual)
                    #     cursor.execute(sql4)
                    #     db2.commit()

                    # 注入DB_JIANGYIN_PREDICT_LAB213里面：
                    # for m in range(16):
                    T2 = int(timestamp_newset) + 2000 * (0 + 1 + 14)
                    T2_arr = np.array(T2)  # sun
                    ID2 = int(id_newset) + (0 + 1 + 14)
                    ID_arr = np.array(ID2)  # sun
                    y_pre3 = 0.8 * y_pre[14 + 0] + 0.2 * predictions1[:, 9 + 0]
                    predict_str = str(y_pre3).strip('[' + ']')
                    # print('predict_str',predict_str)
                    predict_str = np.array(predict_str)
                    conn = pymysql.connect(host='10.11.112.202', port=3306, user='user', password='123456', db='SHUINI')
                    sql1 = "insert into DB_JIANGYIN_PREDICT_LAB213(ID ,M_015,timestamp ) values(%s,%s,%s) ON DUPLICATE KEY UPDATE M_015=%s" % (
                    ID_arr, predict_str, T2_arr, predict_str)
                    cursor.execute(sql1)
                    db2.commit()

                else:
                    break
        if((last2_167!=last_167)):
            judgedata_24=data2_arr[33:34, 8:9][0][0]
            if(judgedata_24!=last_24):
                n=0
                id_init=0
                while(n<100):
                    conn = pymysql.connect(host='10.11.112.202',
                               port=3306,
                               user='user',
                               password='123456',
                               db='SHUINI')

                    sql4 = 'select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN  order by id desc limit 40'
                    data2 = pd.read_sql(sql4, conn)
                    data3 = data2.sort_values(by='ID', axis=0, ascending=True)
                    timestamp_newset = data2.loc[39, 'TIMESTAMP']
                    id_newset = data2.loc[39, 'ID']
                    l0069_newset = data2.loc[39, 'L0069']
                    data2_arr = np.array(data3)
                    data2_arr = np.c_[data2_arr, (data2_arr[:, 2] + data2_arr[:, 3] + data2_arr[:, 4])]
                    data2_arr = data2_arr[:, 5:]
                    if(id_init!=id_newset):
                        id_init=id_newset
                        n=n+1
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data4 = scaler.fit_transform(data2_arr)
                    X_data = data4.reshape(1, data4.shape[0], data4.shape[1])
                    predictions_1 = model2.predict(X_data)
                    predictions1 = predictions_1 + l0069_newset
                    l0069_arr=np.array(data2_arr[:,0],dtype='float')
                    l0069_arr2=np.array(data2_arr[20:,0],dtype='float')
                    # print('l0069_arr',l0069_arr)
                    x_axis=np.arange(1,21,1)
                    x_axis_pre=np.arange(21,51,1)#s输出30个
                    z1 = np.polyfit(x_axis, l0069_arr2, 2)
                    p1 = np.poly1d(z1)
                    y_pre=p1(x_axis_pre)
                    row=['P015','P016','P017','P018','P019','P020','P021','P022','P023','P024','P025','P026','P027','P028','P029','P030','P031','P032','P033','P034','P035','P036'
                        ,'P037','P038','P039','P040','P041','P042','P043','P044','P045','P046','P047','P048','P049','P050']
                    for m in range(16):
                        T2 = int(timestamp_newset) + 2000 * (m + 1+14)
                        T2_arr = np.array(T2)  # sun
                        ID2 = int(id_newset) + (m + 1+14)
                        ID_arr = np.array(ID2)  # sun
                        y_pre3=0.7*y_pre[14+m]+0.3*predictions1[:, 9+m]
                        predict_str = str(y_pre3).strip('[' + ']')
                        # print('predict_str',predict_str)
                        predict_str = np.array(predict_str)
                        conn = pymysql.connect(host='10.11.112.202',port=3306,user='user',password='123456',db='SHUINI')
                        sql1 = "insert into predict_L0069(ID,%s) values(%s,%s) ON DUPLICATE KEY UPDATE %s=%s" % (row[m],
                            ID_arr, predict_str, row[m],predict_str)
                        cursor.execute(sql1)
                        db2.commit()
                        # break
                        sql3 = "select ID,L0069 from DB_JIANGYIN order by id desc limit 1"
                        actual = pd.read_sql(sql3, conn)
                        id_actual = actual.loc[0, 'ID']
                        # print('actual',actual)
                        l0069_actual = actual.loc[0,'L0069']
                        sql4="update predict_L0069 set L0069_actual=%s where id=%s"%(l0069_actual.astype('float'),id_actual)
                        cursor.execute(sql4)
                        db2.commit()

        else:
            n=0
            id_init=0
            while(n<100):
                conn = pymysql.connect(host='10.11.112.202',
                           port=3306,
                           user='user',
                           password='123456',
                           db='SHUINI')
                sql4 = 'select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN  order by id desc limit 40'
                data2 = pd.read_sql(sql4, conn)
                data3 = data2.sort_values(by='ID', axis=0, ascending=True)
                timestamp_newset = data2.loc[39, 'TIMESTAMP']
                id_newset = data2.loc[39, 'ID']
                l0069_newset = data2.loc[39, 'L0069']
                data2_arr = np.array(data3)
                data2_arr = np.c_[data2_arr, (data2_arr[:, 2] + data2_arr[:, 3] + data2_arr[:, 4])]
                data2_arr = data2_arr[:, 5:]
                if(id_init!=id_newset):
                    id_init=id_newset
                    n=n+1
                scaler=MinMaxScaler(feature_range=(0,1))
                data3=scaler.fit_transform(data2_arr)
                X_data = data3.reshape(1, data3.shape[0], data3.shape[1])
                predictions_1 = model3.predict(X_data)
                predictions1=predictions_1+l0069_newset
                l0069_arr=np.array(data2_arr[:,0],dtype='float')
                l0069_arr2=np.array(data2_arr[20:,0],dtype='float')
                x_axis=np.arange(1,21,1)
                x_axis_pre=np.arange(21,51,1)
                z1 = np.polyfit(x_axis, l0069_arr2, 2)
                p1 = np.poly1d(z1)
                y_pre=p1(x_axis_pre)
                row=['P015','P016','P017','P018','P019','P020','P021','P022','P023','P024','P025','P026','P027','P028','P029','P030','P031','P032','P033','P034','P035','P036'
                    ,'P037','P038','P039','P040','P041','P042','P043','P044','P045','P046','P047','P048','P049','P050']
                for m in range(16):
                    T2 = int(timestamp_newset) + 2000 * (m + 1+14)
                    T2_arr = np.array(T2)  # sun
                    ID2 = int(id_newset) + (m + 1+14)
                    ID_arr = np.array(ID2)  # sun
                    y_pre3=0.7*y_pre[14+m]+0.3*predictions1[:, 9+m]
                    predict_str = str(y_pre3).strip('[' + ']')
                    # print('predict_str',predict_str)
                    predict_str = np.array(predict_str)
                    conn = pymysql.connect(host='10.11.112.202',port=3306,user='user',password='123456',db='SHUINI')
                    sql1 = "insert into predict_L0069(ID,%s) values(%s,%s) ON DUPLICATE KEY UPDATE %s=%s" % (row[m],
                        ID_arr, predict_str, row[m],predict_str)
                    cursor.execute(sql1)
                    db2.commit()
                    sql3 = "select ID,L0069 from DB_JIANGYIN order by id desc limit 1"
                    actual = pd.read_sql(sql3, conn)
                    id_actual = actual.loc[0, 'ID']
                    l0069_actual = actual.loc[0,'L0069']
                    sql4="update predict_L0069 set L0069_actual=%s where id=%s"%(l0069_actual.astype('float'),id_actual)
                    cursor.execute(sql4)
                    db2.commit()
        if (last2_24 != last_24):
            judgedata_167 = data2_arr[33:44, 27:28][0][0]
            if (judgedata_167 != last_167):
                print('24后跳的24——167模型开始')
                n = 0
                id_init = 0
                while (n < 40):
                    conn = pymysql.connect(host='10.11.112.202',
                                           port=3306,
                                           user='user',
                                           password='123456',
                                           db='SHUINI')

                    sql4 = 'select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013 from DB_JIANGYIN  order by id desc limit 40'
                    data2 = pd.read_sql(sql4, conn)
                    data3 = data2.sort_values(by='ID', axis=0, ascending=True)
                    timestamp_newset = data2.loc[39, 'TIMESTAMP']
                    id_newset = data2.loc[39, 'ID']
                    l0069_newset = data2.loc[39, 'L0069']
                    data2_arr = np.array(data3)
                    data2_arr = np.c_[data2_arr, (data2_arr[:, 2] + data2_arr[:, 3] + data2_arr[:, 4])]
                    data2_arr = data2_arr[:, 5:]

                    if (id_init != id_newset):
                        id_init = id_newset
                        n = n + 1
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data4 = scaler.fit_transform(data2_arr)
                    X_data = data4.reshape(1, data4.shape[0], data4.shape[1])
                    predictions_1 = model2.predict(X_data)
                    predictions1 = predictions_1 + l0069_newset
                    l0069_arr = np.array(data2_arr[:, 0], dtype='float')
                    l0069_arr2 = np.array(data2_arr[20:, 0], dtype='float')
                    x_axis = np.arange(1, 21, 1)
                    x_axis_pre = np.arange(21, 51, 1)
                    z1 = np.polyfit(x_axis, l0069_arr2, 2)
                    p1 = np.poly1d(z1)
                    y_pre = p1(x_axis_pre)
                    row = ['P015', 'P016', 'P017', 'P018', 'P019', 'P020', 'P021', 'P022', 'P023', 'P024', 'P025',
                           'P026', 'P027', 'P028', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036'
                        , 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047',
                           'P048', 'P049', 'P050']

                    # #注入到predict_L0069里面
                    # for m in range(16):
                    #     T2 = int(timestamp_newset) + 2000 * (m + 1+14)
                    #     T2_arr = np.array(T2)  # sun
                    #     ID2 = int(id_newset) + (m + 1+14)
                    #     ID_arr = np.array(ID2)  # sun
                    #     y_pre3=0.7*y_pre[14+m]+0.3*predictions1[:, 9+m]
                    #     predict_str = str(y_pre3).strip('[' + ']')
                    #     # print('predict_str',predict_str)
                    #     predict_str = np.array(predict_str)
                    #     conn = pymysql.connect(host='10.11.112.202',port=3306,user='user',password='123456',db='SHUINI')
                    #     sql1 = "insert into predict_L0069(ID,%s) values(%s,%s) ON DUPLICATE KEY UPDATE %s=%s" % (row[m],
                    #         ID_arr, predict_str, row[m],predict_str)
                    #     cursor.execute(sql1)
                    #     db2.commit()
                    #     # break
                    #     sql3 = "select ID,L0069 from DB_JIANGYIN order by id desc limit 1"
                    #     actual = pd.read_sql(sql3, conn)
                    #     id_actual = actual.loc[0, 'ID']
                    #     # print('actual',actual)
                    #     l0069_actual = actual.loc[0,'L0069']
                    #     sql4="update predict_L0069 set L0069_actual=%s where id=%s"%(l0069_actual.astype('float'),id_actual)
                    #     cursor.execute(sql4)
                    #     db2.commit()

                    # 注入DB_JIANGYIN_PREDICT_LAB213里面：
                    # for m in range(16):
                    T2 = int(timestamp_newset) + 2000 * (0 + 1 + 14)
                    T2_arr = np.array(T2)  # sun
                    ID2 = int(id_newset) + (0 + 1 + 14)
                    ID_arr = np.array(ID2)  # sun
                    y_pre3 = 0.8 * y_pre[14 + 0] + 0.2 * predictions1[:, 9 + 0]
                    predict_str = str(y_pre3).strip('[' + ']')
                    # print('predict_str',predict_str)
                    predict_str = np.array(predict_str)
                    conn = pymysql.connect(host='10.11.112.202', port=3306, user='user', password='123456', db='SHUINI')
                    sql1 = "insert into DB_JIANGYIN_PREDICT_LAB213(ID ,M_015,timestamp ) values(%s,%s,%s) ON DUPLICATE KEY UPDATE M_015=%s" % (
                    ID_arr, predict_str, T2_arr, predict_str)
                    cursor.execute(sql1)
                    db2.commit()

        if ((last2_24 == last_24) and (last2_167 == last_167) and (last2_33 == last_33)):
            conn = pymysql.connect(host='10.11.112.202',
                                   port=3306,
                                   user='user',
                                   password='123456',
                                   db='SHUINI')

            sql2 = "select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013,l0014,L0070 from DB_JIANGYIN order by id desc limit 40"
            slj2 = pd.read_sql(sql2, conn)
            slj1 = slj2.sort_values(by='ID', axis=0, ascending=True)
            data2_arr = np.array(slj1)

            timestamp_newset = slj2.loc[0, 'TIMESTAMP']
            id_newset = slj2.loc[0, 'ID']
            l0069_newset = slj2.loc[0, 'L0069']
            data2_arr = np.c_[data2_arr, (data2_arr[:, 2] + data2_arr[:, 3] + data2_arr[:, 4])]  # 将总喂料量添加到最后一列
            data2_arr = data2_arr[:, 5:]
            scaler = MinMaxScaler(feature_range=(0, 1))
            data3 = scaler.fit_transform(data2_arr)
            X_data = data3.reshape(1, data3.shape[0], data3.shape[1])
            predictions_1 = model5.predict(X_data)
            y_min = np.min(slj1['L0069'])
            y_max = np.max(slj1['L0069'])
            d = y_max - y_min
            predictions_1 = predictions_1 * d + y_min
            predictions1 = predictions_1
            l0069_arr = np.array(data2_arr[:, 0], dtype='float')
            l0069_arr2 = np.array(data2_arr[20:, 0], dtype='float')
            x_axis = np.arange(1, 21, 1)
            x_axis_pre = np.arange(21, 51, 1)
            z1 = np.polyfit(x_axis, l0069_arr2, 2)
            p1 = np.poly1d(z1)
            y_pre = p1(x_axis_pre)
            row = ['P015', 'P016', 'P017', 'P018', 'P019', 'P020', 'P021', 'P022', 'P023', 'P024', 'P025', 'P026',
                   'P027', 'P028', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036'
                , 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047', 'P048',
                   'P049', 'P050']

            # 注入到predict——L0069里面
            # for m in range(16):
            #     T2 = int(timestamp_newset) + 2000 * (m + 1+14)
            #     T2_arr = np.array(T2)  # sun
            #     ID2 = int(id_newset) + (m + 1+14)
            #     ID_arr = np.array(ID2)  # sun
            #     y_pre3=0.7*y_pre[9+m]+0.3*predictions1[:, 9+m]
            #     predict_str = str(y_pre3).strip('[' + ']')
            #     # print('predict_str',predict_str)
            #     predict_str = np.array(predict_str)
            #     conn = pymysql.connect(host='10.11.112.202',port=3306,user='user',password='123456',db='SHUINI')
            #     sql1 = "insert into predict_L0069(ID,%s) values(%s,%s) ON DUPLICATE KEY UPDATE %s=%s" % (row[m],
            #         ID_arr, predict_str, row[m],predict_str)
            #     cursor.execute(sql1)
            #     db2.commit()
            #     # break
            #     sql3 = "select ID,L0069 from DB_JIANGYIN order by id desc limit 1"
            #     actual = pd.read_sql(sql3, conn)
            #     id_actual = actual.loc[0, 'ID']
            #     # print('actual',actual)
            #     l0069_actual = actual.loc[0,'L0069']
            #     sql4="update predict_L0069 set L0069_actual=%s where id=%s"%(l0069_actual.astype('float'),id_actual)
            #     cursor.execute(sql4)
            #     db2.commit()

            # 注入DB_JIANGYIN_PREDICT_LAB213里面：
            # for m in range(16):
            T2 = int(timestamp_newset) + 2000 * (0 + 1 + 14)
            T2_arr = np.array(T2)  # sun
            ID2 = int(id_newset) + (0 + 1 + 14)
            ID_arr = np.array(ID2)  # sun
            y_pre3 = 0.7 * y_pre[9 + 0] + 0.3 * predictions1[:, 9 + 0]
            predict_str = str(y_pre3).strip('[' + ']')
            # print('predict_str',predict_str)
            predict_str = np.array(predict_str)
            conn = pymysql.connect(host='10.11.112.202', port=3306, user='user', password='123456', db='SHUINI')
            sql1 = "insert into DB_JIANGYIN_PREDICT_LAB213(ID ,M_015,timestamp ) values(%s,%s,%s) ON DUPLICATE KEY UPDATE M_015=%s" % (
            ID_arr, predict_str, T2_arr, predict_str)
            cursor.execute(sql1)
            db2.commit()
        else:
            conn = pymysql.connect(host='10.11.112.202',
                                   port=3306,
                                   user='user',
                                   password='123456',
                                   db='SHUINI')

            sql2 = "select ID,TIMESTAMP,L0002,L0005,L0008,L0069,L0022,L0023,L0024,L0026,L0027,L0028,L0033,L0034,L0040,L0042,L0070,L0093,L0094,L0095,L0096,L0097,L0157,L0158,L0159,L0160,L0162,L0167,L0013,l0014,L0070 from DB_JIANGYIN order by id desc limit 40"
            slj2 = pd.read_sql(sql2, conn)
            slj1 = slj2.sort_values(by='ID', axis=0, ascending=True)
            data2_arr = np.array(slj1)

            timestamp_newset = slj2.loc[0, 'TIMESTAMP']
            id_newset = slj2.loc[0, 'ID']
            l0069_newset = slj2.loc[0, 'L0069']
            data2_arr = np.c_[data2_arr, (data2_arr[:, 2] + data2_arr[:, 3] + data2_arr[:, 4])]  # 将总喂料量添加到最后一列
            data2_arr = data2_arr[:, 5:]
            scaler = MinMaxScaler(feature_range=(0, 1))
            data3 = scaler.fit_transform(data2_arr)
            X_data = data3.reshape(1, data3.shape[0], data3.shape[1])
            predictions_1 = model4.predict(X_data)
            y_min = np.min(slj1['L0069'])
            y_max = np.max(slj1['L0069'])
            d = y_max - y_min
            predictions_1 = predictions_1 * d + y_min
            predictions1 = predictions_1
            l0069_arr = np.array(data2_arr[:, 0], dtype='float')
            l0069_arr2 = np.array(data2_arr[20:, 0], dtype='float')
            x_axis = np.arange(1, 21, 1)
            x_axis_pre = np.arange(21, 51, 1)
            z1 = np.polyfit(x_axis, l0069_arr2, 2)
            p1 = np.poly1d(z1)
            y_pre = p1(x_axis_pre)
            row = ['P015', 'P016', 'P017', 'P018', 'P019', 'P020', 'P021', 'P022', 'P023', 'P024', 'P025', 'P026',
                   'P027', 'P028', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036'
                , 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047', 'P048',
                   'P049', 'P050']
            T2 = int(timestamp_newset) + 2000 * (0 + 1 + 14)
            T2_arr = np.array(T2)  # sun
            ID2 = int(id_newset) + (0 + 1 + 14)
            ID_arr = np.array(ID2)  # sun
            y_pre3 = 0.7 * y_pre[9 + 0] + 0.3 * predictions1[:, 9 + 0]
            predict_str = str(y_pre3).strip('[' + ']')
            # print('predict_str',predict_str)
            predict_str = np.array(predict_str)
            conn = pymysql.connect(host='10.11.112.202', port=3306, user='user', password='123456', db='SHUINI')
            sql1 = "insert into DB_JIANGYIN_PREDICT_LAB213(ID ,M_015,timestamp ) values(%s,%s,%s) ON DUPLICATE KEY UPDATE M_015=%s" % (
                ID_arr, predict_str, T2_arr, predict_str)
            cursor.execute(sql1)
            db2.commit()
if __name__ == '__main__':

    predict_DS1()
