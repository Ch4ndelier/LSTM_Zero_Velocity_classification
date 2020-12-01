from __future__ import division
import numpy as np
import os

IMU_PRESS_PATH = './bupt_data/paths.txt'
LENGTH = 64
INTERVAL = 4


def Process_data_get_numpy(imu_path, press_path):
    imu = np.loadtxt(imu_path, usecols=(2, 3, 4, 5, 6, 7))
    press_label = np.loadtxt(press_path)
    X_row = np.size(imu, 0)
    Y_row = np.size(press_label, 0)
    # print(X_row, Y_row)
    max_row = max(X_row, Y_row)
    if Y_row == max_row:
        dis = max_row - X_row
        inter = Y_row // dis
        ignore_label = slice(0, Y_row, inter)
        press_label = np.delete(press_label, ignore_label, axis=0)
        # print(np.size(press_label))
    elif X_row == max_row:
        dis = max_row - Y_row
        inter = X_row // dis
        ignore_label = slice(0, X_row, inter)
        imu = np.delete(imu, ignore_label, axis=0)
        # print(np.size(imu))

    imu_list = imu.tolist()
    press_label_list = press_label.tolist()
    for i in range(len(press_label_list)):
        if press_label_list[i] == 0:
            press_label_list[i] = [0, 1]
        else:
            press_label_list[i] = [1, 0]

    train_x = []
    train_y = []
    for i in range(LENGTH, min(len(imu_list), len(press_label_list)), INTERVAL):
        a_seq = []
        for j in range(i - LENGTH, i):
            a_seq.append(imu_list[j])
        train_x.append(a_seq)
        train_y.append(press_label_list[i])

    return [np.array(train_x), np.array(train_y)]


# up sampling
def Process_data_get_numpy_upsample(imu_path, press_path):
    # print("upsample")
    imu = np.loadtxt(imu_path, usecols=(2, 3, 4, 5, 6, 7))
    press_label = np.loadtxt(press_path)
    X_row = np.size(imu, 0)
    Y_row = np.size(press_label, 0)
    max_row = max(X_row, Y_row)
    # print(X_row, Y_row)
    if X_row == Y_row:
        pass
    elif Y_row == max_row:
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
                to_insert = [(to_insert_former[i] + to_insert_later[i]) / 2 for i in range(len(to_insert_former))]
                to_insert = np.array(to_insert)
                imu = np.insert(imu, _iter, to_insert, axis=0)
                # print("size: ", np.size(imu, 0))
                p += 1
            _iter -= 1
        # print(np.size(imu, 0), np.size(press_label))
    elif X_row == max_row:
        dis = max_row - Y_row
        inter = Y_row // dis
        _iter = Y_row
        cnt = 0
        while _iter:
            cnt += 1
            if cnt == inter:
                to_insert_former = press_label[_iter - 1]
                to_insert_later = press_label[_iter]
                to_insert = [(to_insert_former[i] + to_insert_later[i]) / 2 for i in range(len(to_insert_former))]
                press_label = np.insert(press_label, _iter, to_insert, axis=0)
                cnt = 0
            _iter -= 1

    imu_list = imu.tolist()
    press_label_list = press_label.tolist()
    for i in range(len(press_label_list)):
        if press_label_list[i] == 0:
            press_label_list[i] = [0, 1]
        else:
            press_label_list[i] = [1, 0]

    train_x = []
    train_y = []
    for i in range(LENGTH, min(len(imu_list), len(press_label_list)), INTERVAL):
        a_seq = []
        for j in range(i - LENGTH, i):
            a_seq.append(imu_list[j])
        train_x.append(a_seq)
        train_y.append(press_label_list[i])
    print(np.array(train_x).shape, np.array(train_y).shape)
    return [np.array(train_x), np.array(train_y)]

data_path_list = []
with open(IMU_PRESS_PATH, 'r+', encoding='utf-8') as f:
    for line in f.readlines():
        # print(line[:-1].split('$'))
        data_path_list.append(line[:-1].split('$'))

ALL_X, ALL_Y, TRAIN_X, TRAIN_Y, DEV_X, DEV_Y = [], [], [], [], [], []
N = len(data_path_list)
i = 0
for data_path in data_path_list:
    # print(data_path)
    np_list = Process_data_get_numpy_upsample(data_path[0], data_path[1])
    ALL_X.extend(np_list[0].tolist())
    ALL_Y.extend(np_list[1].tolist())
    if i < N * 0.9:
        TRAIN_X.extend(np_list[0].tolist())
        TRAIN_Y.extend(np_list[1].tolist())
    else:
        DEV_X.extend(np_list[0].tolist())
        DEV_Y.extend(np_list[1].tolist())
    i += 1
print("All data shape:")
print(np.array(ALL_X).shape)
print(np.array(ALL_Y).shape)
print("Train data shape:")
print(np.array(TRAIN_X).shape)
print(np.array(TRAIN_Y).shape)
print("Val data shape:")
print(np.array(DEV_X).shape)
print(np.array(DEV_Y).shape)

dir_name = "./data_process/int_4_len_64_91_bupt/"
if os.path.exists(dir_name):
    print("path already exists!!")
    exit()
else:
    print("saving ...")
    os.mkdir(dir_name)
    np.save(dir_name + "TRAIN_X_1.npy", np.array(TRAIN_X))
    np.save(dir_name + "TRAIN_Y_1.npy", np.array(TRAIN_Y))
    np.save(dir_name + "DEV_X_1.npy", np.array(DEV_X))
    np.save(dir_name + "DEV_Y_1.npy", np.array(DEV_Y))
print("complete!!")