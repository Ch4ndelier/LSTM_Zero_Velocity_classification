import numpy as np
import os

imu = np.loadtxt("./data_process/raw_data/IMU.txt", usecols=(2, 3, 4, 5, 6, 7))
print("shape of imu", imu.shape)
# print(imu)
press_label = np.loadtxt("./data_process/raw_data/press_label.txt")
# print(press_label)
print("shape of press_label", press_label.shape)


X_row = np.size(imu, 0)
Y_row = np.size(press_label, 0)
print(X_row, Y_row)
max_row = max(X_row, Y_row)
if Y_row == max_row:
    dis = max_row - X_row
    inter = Y_row // dis
    ignore_label = slice(0, Y_row, inter)
    press_label = np.delete(press_label, ignore_label, axis=0)
    print(np.size(press_label))
elif X_row == max_row:
    dis = max_row - Y_row
    inter = X_row // dis
    ignore_label = slice(0, X_row, inter)
    imu = np.delete(imu, ignore_label, axis=0)
    print(np.size(imu))

print('行数缩放至一致', np.size(imu, 0), np.size(press_label, 0))

# TODO:numpy
imu_list = imu.tolist()
press_label_list = press_label.tolist()
for i in range(len(press_label_list)):
    if press_label_list[i] == 0:
        press_label_list[i] = [0, 1]
    else:
        press_label_list[i] = [1, 0]

train_x = []
train_y = []
for i in range(256, np.size(imu, 0) - 1, 50):
    a_batch = []
    for j in range(i - 256, i):
        a_batch.append(imu_list[j])
    train_x.append(a_batch)
    train_y.append(press_label_list[i])

print(np.array(train_x).shape)
print(np.array(train_y).shape)

np.save("./data_process/processed/dev_X.npy", np.array(train_x))
np.save("./data_process/processed/dev_Y.npy", np.array(train_y))
