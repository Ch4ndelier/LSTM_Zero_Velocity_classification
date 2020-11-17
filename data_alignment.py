from __future__ import division
import numpy as np
import os

IMU_PRESS_PATH = './Dataset/ori_paths.txt'

def Process_data_alignment_precise(imu_path, press_path):
    # print("upsample")
    imu = np.loadtxt(imu_path, usecols=(2, 3, 4, 5, 6, 7))
    imu_date = np.loadtxt(imu_path, dtype=bytes, usecols=(0, 1)).astype(str)
    press_label = np.loadtxt(press_path)
    X_row = np.size(imu, 0)
    Y_row = np.size(press_label, 0)
    print("Before sample", X_row, Y_row)
    max_row = max(X_row, Y_row)
    if Y_row == max_row:
        dis = max_row - X_row
        inter = X_row / dis
        print(inter)
        _iter = X_row
        cnt = 0
        p = 1
        # print("interval", inter)
        while _iter:
            cnt += 1
            target = inter * p // 1
            if cnt == target:
                to_insert_former = imu[_iter - 1]
                to_insert_later = imu[_iter]
                date_to_insert = imu_date[_iter]
                imu_date = np.insert(imu_date, _iter, date_to_insert, axis=0)
                to_insert = [(to_insert_former[i] + to_insert_later[i]) / 2 for i in range(len(to_insert_former))]
                to_insert = np.array(to_insert)
                imu = np.insert(imu, _iter, to_insert, axis=0)
                p += 1
                # print("imu size: ", np.size(imu, 0))
                # print("imu_date size:", np.size(imu_date, 0))
            _iter -= 1
        # print(np.size(imu, 0), np.size(press_label))
    elif X_row == max_row:
        print("OPPs!!")
        exit()

    imu_date_list = imu_date.tolist()
    imu_list = imu.tolist()
    templist = [imu_date_list[i] + imu_list[i] for i in range(len(imu_date))]
    print("after sample", len(templist), len(press_label))
    return templist
# data alignment 


def Process_data_alignment(imu_path, press_path):
    # print("upsample")
    imu = np.loadtxt(imu_path, usecols=(2, 3, 4, 5, 6, 7))
    imu_date = np.loadtxt(imu_path, dtype=bytes, usecols=(0, 1)).astype(str)
    press_label = np.loadtxt(press_path)
    X_row = np.size(imu, 0)
    Y_row = np.size(press_label, 0)
    print("Before sample", X_row, Y_row)
    max_row = max(X_row, Y_row)
    if Y_row == max_row:
        dis = max_row - X_row
        inter = X_row / dis
        print(inter)
        _iter = X_row
        cnt = 0
        # print("interval", inter)
        while _iter:
            cnt += 1
            if cnt == inter // 1 and dis != 0:
                if dis > 0:
                    dis -= 1
                to_insert_former = imu[_iter - 1]
                to_insert_later = imu[_iter]
                date_to_insert = imu_date[_iter]
                imu_date = np.insert(imu_date, _iter, date_to_insert, axis=0)
                to_insert = [(to_insert_former[i] + to_insert_later[i]) / 2 for i in range(len(to_insert_former))]
                to_insert = np.array(to_insert)
                imu = np.insert(imu, _iter, to_insert, axis=0)
                # print("imu size: ", np.size(imu, 0))
                # print("imu_date size:", np.size(imu_date, 0))
                cnt = 0
            _iter -= 1
        # print(np.size(imu, 0), np.size(press_label))
    elif X_row == max_row:
        print("OPPs!!")
        exit()

    imu_date_list = imu_date.tolist()
    imu_list = imu.tolist()
    templist = [imu_date_list[i] + imu_list[i] for i in range(len(imu_date))]
    print("after sample", len(templist), len(press_label))
    return templist

data_path_list = []
with open(IMU_PRESS_PATH, 'r+', encoding='utf-8') as f:
    for line in f.readlines():
        # print(line[:-1].split('$'))
        data_path_list.append(line[:-1].split('$'))

for data_path in data_path_list:
    templist = Process_data_alignment_precise(data_path[0], data_path[1])
    output = open(data_path[0], 'w', encoding='gbk') 
    for row in templist:
        rowtxt = '{} {} {} {} {} {} {} {}'.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        output.write(rowtxt)
        output.write('\n')
    output.close()
