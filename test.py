import torch
ref_model = torch.load('/Users/liujunyuan/Zero-velocity/MyLSTM/model/zv_lstm_model.tar')
print(ref_model)

'''
t = torch.tensor([[[1,2,4],[4,5,6]],[[1,2,4],[4,5,6]]])
print(t)
t1 = t.view(3,-1)
print(t1)
t2 = t.view((3,-1))
print(t2)
print(t.size())
print(t.size(0))
print(t.size(1))
print(t.size(2))
'''