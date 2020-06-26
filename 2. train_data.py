import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import models
import numpy as np
from models import BasicBlock
from datetime import datetime
import visdom


#We are gonna use visdom
vis = visdom.Visdom()
vis.close(env="main")


def show_validation(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )


# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# torch.manual_seed(777)
# if device == 'cuda':
#     torch.cuda.manual_seed_all(777)


device = torch.device("cuda")
torch.cuda.manual_seed_all(777)
device = 'cuda'

#target FILE number = 600 but for balancing, Goal File number is maybe about 1800
#last 2 files left for checking Accuracy. 1781 ~ 1787
#new balanced data 2160 2020-06-07
FILE_I_END = 2160

learning_rate = 0.001
training_epochs = 40
#epoch = 21

MODEL_NAME = 'MY AWSOEM AI MODEL'
PREV_MODEL = ''

LOAD_MODEL = True

WIDTH = 480
HEIGHT = 270
batch_size = 100  # 여기서 정해준 배치사이즈는 data_loader 에서 돌아감
#batch_size = 30




### 키 설정
w = 0
s = 1
a = 2
d = 3
wa = 4
wd = 5
sa = 6
sd = 7
nk = 8

# instantiate ResNet model


#model = models.CNN_simple().to(device)



# # load data
# file_name = 'C:/Users/esaw2/work/Project/pygta5/Self_driving-GTA5-with-Pytorch/data/training_data-{}.npy'.format(i)
# # # full file info
# train = np.load(file_name)
#
# train_X = torch.FloatTensor(np.array([i[0] for i in train]).reshape(-1, 3, WIDTH, HEIGHT))
# train_Y = torch.LongTensor([i[1] for i in train])
#
# train_X = train_X.to(device)
# train_Y = train_Y.to(device)
#
# #print(train_X)
# #print(train_X.shape)
# #print(train_Y)
# #print(train_Y.shape)
#
# train = TensorDataset(train_X, train_Y)
#
# data_loader = torch.utils.data.DataLoader(dataset=train,
#                                           batch_size=batch_size,
#                                           shuffle=True,
#                                           drop_last=True)


#make plot
training_loss_iteration = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Traning loss(per iteration)', legend=['loss'], showlegend=True))
training_loss_epoch = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Traning loss(per epoch)', legend=['loss'], showlegend=True))
Test_loss__iteration = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Testing loss(per iteration)', legend=['loss'], showlegend=True))
Test_loss_epoch = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Testing loss(per epoch)', legend=['loss'], showlegend=True))
Accruacy_iteration = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Accuracy (per iteration)', legend=['Accuracy'], showlegend=True))
Accuracy_epoch = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Accuracy (per epoch)', legend=['Accuracy'], showlegend=True))

#Resnet model
model = models.ResNet(block = BasicBlock, layers=[2, 2, 2, 2], num_classes=9).to(device)
# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
#criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train my model
# model.train()  # set the model to train mode (dropout=True)



check_train_start_time = datetime.now()

print("Train Start Time!:", check_train_start_time)
print('\b')

Count_calculation = 0

for epoch in range(training_epochs):


    avg_loss = 0.0
    avg_test_loss = 0.0
    sum_accuracy = 0.0
    Count_iteration = 0

    epoch_start_time = datetime.now()

    # 데이터 순서 정하고 셔플
    data_order = [i for i in range(1, FILE_I_END + 1)]
    #shuffle(data_order)

    # total_batch = len(data_loader)


    for count, i in enumerate(data_order):
        # iteration update
        # iteration += 1

        # try:
        # load data
        train_file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/train data/balanced training_data-{}.npy'.format(i)
        test_file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/test data/balanced testing data-{}.npy'.format(i)
        # # full file info
        train = np.load(train_file_name)
        test = np.load(test_file_name)

        train_X = torch.FloatTensor(np.array([i[0] for i in train]).reshape(-1, 3, WIDTH, HEIGHT))
        train_Y = torch.LongTensor([i[1] for i in train])

        test_X = torch.FloatTensor(np.array([i[0] for i in test]).reshape(-1, 3, WIDTH, HEIGHT))
        test_Y = torch.LongTensor([i[1] for i in test])

        train_X = train_X.to(device)
        train_Y = train_Y.to(device)


        #print(train_X)
        #print(train_X.shape)
        #print(train_Y)
        #print(train_Y.shape)

        train = TensorDataset(train_X, train_Y)

        train_set = torch.utils.data.DataLoader(dataset=train,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True)


        total_batch = len(train_set) * FILE_I_END
        total_test_length = len(test) * FILE_I_END




        for X, Y in train_set:
            # train my model. I think I'm trying to train with test so must to change train mode each epoch
            model.train()  # set the model to train mode (dropout=True)

            X, Y = Variable(X), Variable(Y)

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            loss = criterion(hypothesis, Y)
            #netloss += loss.item()
            loss.backward()
            optimizer.step()
            #del loss
            #del hypothesis


            avg_loss += loss / total_batch

            Count_calculation += 1
            Count_iteration += 1

            # print('Count_iteration:', Count_iteration)

        show_validation(training_loss_iteration, torch.Tensor([loss]), torch.Tensor([Count_calculation]))







        if (count + 1) % 10 == 0:
            torch.save(model.state_dict(),
                           "C:/Users/esaw2/work/Project/pygta5 data/balanced data/saved model/saved_{}-epoch_model.pt".format(
                               epoch + 1))
            print("{}-epoch model saved!".format(epoch + 1))
            print('\b')


        print('[Real Time Training loss: {:>.6}]'.format(loss))


        with torch.no_grad():

            #for validation we must to change eval mode at this script
            model.eval()

            test_X = test_X.to(device)
            test_Y = test_Y.to(device)

            prediction = model(test_X).to(device)
            correct_prediction = torch.argmax(prediction, 1) == test_Y
            # print("correct prediction:", correct_prediction)
            test_loss = torch.nn.CrossEntropyLoss()(prediction, test_Y).to(device)
            accuracy = correct_prediction.float().mean()
            # print('accuracy:', accuracy)
            # print('accuracy.item():', accuracy.item())

            avg_test_loss += test_loss / total_test_length

            sum_accuracy += accuracy.item()
            # print('sum_accuracy:', sum_accuracy)
            print('[Real Time Testing loss:{:>.6}]'.format(test_loss.item()))
            print('[Real Time Accuracy:{:>.8} %]'.format(accuracy.item() * 100))
            print('\b')

            #test loss 문제 있음 6/6

            show_validation(Test_loss__iteration, torch.Tensor([test_loss.item()]), torch.Tensor([Count_calculation]))
            show_validation(Accruacy_iteration, torch.Tensor([accuracy.item() * 100]), torch.Tensor([Count_calculation]))
            # print('val iteration:', iteration)

        # vis.text('[file iteration loss: {:>.6}]'.format(loss), env="main")
        # show_validation(training_loss_iteration, torch.Tensor([loss]), torch.Tensor([iteration]))
        # print('loss iteration:', iteration)

        # except Exception as e:
        #     print(str(e))

    print('\b')
    print('[Epoch: {:>4}] avg loss = {:>.6} '.format(epoch + 1, avg_loss))
    print('[Epoch: {:>4}] acurracy: {:>.8} %'.format(epoch + 1, (100 * sum_accuracy / Count_iteration)))
    show_validation(training_loss_epoch, torch.Tensor([avg_loss]), torch.Tensor([epoch + 1]))

    # Test loss와 Accuracy_epoch 그래프 문제가 있음. 로직 다시 볼 것 6/9
    show_validation(Test_loss_epoch, torch.Tensor([avg_test_loss]), torch.Tensor([epoch+1]))
    show_validation(Accuracy_epoch, torch.Tensor([100 * sum_accuracy / Count_iteration]), torch.Tensor([epoch + 1]))

    epoch_time = datetime.now() - epoch_start_time
    print("[{} Epoch took time] :".format(epoch + 1), epoch_time)
    print("[Time now]:", datetime.now())
    print('\b')
    print('\b')

    epoch_start_time = datetime.now()

finished_time = datetime.now() - check_train_start_time
print("[Whole Training took time] :", finished_time)
print('[Learning Finished!]')

