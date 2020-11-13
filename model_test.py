import torch
import os
import numpy as np
from LSTM import LSTM


DATA_DIR = "./data_process/int_1_len_24_91_up"
MODEL_PATH = './model/test.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev_X_preprocessed_data = os.path.join(DATA_DIR, "DEV_X_1.npy")
dev_Y_preprocessed_data = os.path.join(DATA_DIR, "DEV_Y_1.npy")

dev_X = torch.from_numpy(np.load(dev_X_preprocessed_data)).type(torch.Tensor).to(device)
dev_Y = torch.from_numpy(np.load(dev_Y_preprocessed_data)).type(torch.Tensor).to(device)

model = torch.load(MODEL_PATH)
model.eval()


def test_get_accuracy(logits, target):
    out_index = torch.max(logits.cpu().data, 1)[1].numpy()
    prob = torch.max(logits.cpu().data, 1)[0].numpy()
    # out_index[np.where(prob < 0.85)] = 0
    cor = torch.from_numpy(out_index)
    target = target.cpu()
    corrects = (
        cor.view(target.size()).data == target.data
    ).sum()
    print(corrects)
    accuracy = 100.0 * corrects / len(target)
    return accuracy.item()

#TODO:Xseq?
dev_X = dev_X.permute(1, 0, 2)
dev_Y = torch.max(dev_Y, 1)[1]

y_pred = model(dev_X)
acc = model.get_accuracy(y_pred, dev_Y) * model.batch_size / len(dev_Y)
print(acc)
