import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from models import BasicBlock
import models
import time
from datetime import datetime

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


device = torch.device("cpu")
torch.cuda.manual_seed_all(777)
device = 'cpu'


FILE_I_END = 7
WIDTH = 480
HEIGHT = 270

model = models.ResNet(block = BasicBlock, layers = [2, 2, 2, 2]).to(device)  # (*args, **kwargs)
#model = resnet.ResNet(block=resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=9).to(device)

model.load_state_dict(torch.load("C:/Users/esaw2/work/Project/pygta5 data/balanced data/saved model/saved_model39-epoch_model.pt", map_location=device)



    # 데이터 순서 정하고 셔플
data_order = [i for i in range(1, FILE_I_END + 1)]

for count, i in enumerate(data_order):
    # try:
    # load data
    file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/test data/balanced training_data-{}.npy'.format(i)
    # # full file info
    test = np.load(file_name)

    test_X = torch.FloatTensor(np.array([i[0] for i in test]).reshape(-1, 3, WIDTH, HEIGHT))
    test_Y = torch.LongTensor([i[1] for i in test])

    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    # print(train_X)
    # print(train_X.shape)
    # print(train_Y)
    # print(train_Y.shape)

    test = TensorDataset(test_X, test_Y)

    data_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              )

    with torch.no_grad():
        model.eval()  # set the model to evaluation mode (dropout=False)

        X_test = test_X.to(device)
        Y_test = test_Y.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())






