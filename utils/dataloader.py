import torch
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(DATA_DIR):
    train_X_preprocessed_data = os.path.join(DATA_DIR, "TRAIN_X_1.npy")
    # train_Y_preprocessed_data = "./data/data_train_target.npy"
    train_Y_preprocessed_data = os.path.join(DATA_DIR, "TRAIN_Y_1.npy")
    dev_X_preprocessed_data = os.path.join(DATA_DIR, "DEV_X_1.npy")
    dev_Y_preprocessed_data = os.path.join(DATA_DIR, "DEV_Y_1.npy")
    # test_X_preprocessed_data = "./data/data_test_input.npy"
    # test_Y_preprocessed_data = "./data/data_test_target.npy"

    if (
        os.path.isfile(train_X_preprocessed_data)
        and os.path.isfile(train_Y_preprocessed_data)
        and os.path.isfile(dev_X_preprocessed_data)
        and os.path.isfile(dev_Y_preprocessed_data)
        # and os.path.isfile(test_X_preprocessed_data)
        # and os.path.isfile(test_Y_preprocessed_data)
    ):
        print("Preprocessed files exist, deserializing npy files")
        train_X = torch.from_numpy(np.load(train_X_preprocessed_data)).type(torch.Tensor).to(device)
        train_Y = torch.from_numpy(np.load(train_Y_preprocessed_data)).type(torch.Tensor).to(device)
        print(train_X.type())
        print(np.load(train_X_preprocessed_data).shape)
        print(np.load(train_Y_preprocessed_data).shape)
        dev_X = torch.from_numpy(np.load(dev_X_preprocessed_data)).type(torch.Tensor).to(device)
        dev_Y = torch.from_numpy(np.load(dev_Y_preprocessed_data)).type(torch.Tensor).to(device)

        # test_X = torch.from_numpy(np.load(test_X_preprocessed_data)).type(torch.Tensor)
        # test_Y = torch.from_numpy(np.load(test_Y_preprocessed_data)).type(torch.Tensor)
    else:
        print("Loss of Data! Press make sure the path is right!")
        exit()
    return train_X, train_Y, dev_X, dev_Y
