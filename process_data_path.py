import numpy as np
import os

IMU_PRESS_PATH = './Dataset/paths.txt'


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
    for i in range(256, min(X_row, Y_row) - 10, 50):
        a_seq = []
        for j in range(i - 256, i):
            a_seq.append(imu_list[j])
        train_x.append(a_seq)
        train_y.append(press_label_list[i])

    return [np.array(train_x), np.array(train_y)]


data_path_list = []
with open(IMU_PRESS_PATH, 'r+', encoding='utf-8') as f:
    for line in f.readlines():
        # print(line[:-1].split('$'))
        data_path_list.append(line[:-1].split('$'))

ALL_X = []
ALL_Y = []
TRAIN_X = []
TRAIN_Y = []
DEV_X = []
DEV_Y = []
N = len(data_path_list)
i = 0
for data_path in data_path_list:
    np_list = Process_data_get_numpy(data_path[0], data_path[1])
    ALL_X.extend(np_list[0].tolist())
    ALL_Y.extend(np_list[1].tolist())
    if i < N * 0.7:
        TRAIN_X.extend(np_list[0].tolist())
        TRAIN_Y.extend(np_list[1].tolist())
    else:
        DEV_X.extend(np_list[0].tolist())
        DEV_Y.extend(np_list[1].tolist())
    i += 1
print(np.array(ALL_X).shape)
print(np.array(ALL_Y).shape)
print(np.array(TRAIN_X).shape)
print(np.array(TRAIN_Y).shape)
print(np.array(DEV_X).shape)
print(np.array(DEV_Y).shape)

np.save("./data_process/processed/TRAIN_X.npy", np.array(TRAIN_X))
np.save("./data_process/processed/TRAIN_Y.npy", np.array(TRAIN_Y))
np.save("./data_process/processed/DEV_X.npy", np.array(DEV_X))
np.save("./data_process/processed/DEV_Y.npy", np.array(DEV_Y))
