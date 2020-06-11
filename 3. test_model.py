from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check
from collections import deque
import random
from statistics import mean
from motion import motion_detection
import torch
import models
import numpy as np
from models import BasicBlock

# 5/15 can't start with gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

device = 'cuda'

# GAME_WIDTH = 1920
# GAME_HEIGHT = 1080



GAME_WIDTH = 800
GAME_HEIGHT = 625

how_far_remove = 800
rs = (20, 15)
log_len = 25

motion_req = 800
motion_log = deque(maxlen=log_len)

WIDTH = 480
HEIGHT = 270
# LR = 1e-3
LR = 0.00023
EPOCHS = 10

choices = deque([], maxlen=5)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)

# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]
# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]
# nk = [0,0,0,0,0,0,0,0,1]


# 4.5, 0.1, 0.1, 0.1, 2, 2, 0.5, 0.5, 0.2
w = 0
s = 1
a = 2
d = 3
wa = 4
wd = 5
sa = 6
sd = 7
nk = 8

t_time = 0.25


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    # ReleaseKey(S)

# def left():
#     ReleaseKey(S)
#     PressKey(A)
#     ReleaseKey(W)
#     ReleaseKey(D)

def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

# def right():
#     ReleaseKey(S)
#     ReleaseKey(A)
#     ReleaseKey(W)
#     PressKey(D)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)

# def no_keys():
#     ReleaseKey(S)
#     ReleaseKey(D)
#     ReleaseKey(W)
#     ReleaseKey(A)

# model = models.ResNet(block = models.Bottleneck, layers = [2, 2, 2, 2]).to(device)  # (*args, **kwargs)
model = models.ResNet(block = BasicBlock, layers = [2, 2, 2, 2]).to(device)  # (*args, **kwargs)
#model = resnet.ResNet(block=resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=9).to(device)

model.load_state_dict(torch.load("C:/Users/esaw2/work/Project/pygta5 data/balanced data/saved model/saved_39-epoch_model.pt", map_location=device))
model.to(device)

model.eval()

# model = googlenet(WIDTH, HEIGHT, 3, LR, output=9)
# MODEL_NAME = 'MY AWSOEM AI MODEL'
# model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')


def main():
    last_time = time.time()
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0, 40, GAME_WIDTH, GAME_HEIGHT))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (WIDTH, HEIGHT))

    t_minus = prev
    t_now = prev
    t_plus = prev

    while (True):

        if not paused:
            screen = grab_screen(region=(0, 40, GAME_WIDTH, GAME_HEIGHT))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            # img = screen
            # cv2.imshow('test', img)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break

            delta_count = motion_detection(t_minus, t_now, t_plus)

            t_minus = t_now
            t_now = t_plus
            t_plus = screen
            t_plus = cv2.blur(t_plus, (4, 4))

            screen = torch.FloatTensor(screen)
            #print(screen)

            input_image = screen.view(-1, 3, WIDTH, HEIGHT)

            #모델에 데이터를 제공하는 input Tensor를 cuda최적화
            input_image = input_image.to(device)


            # prediction = model.predict([screen.reshape(WIDTH,HEIGHT,3)])[0]
            with torch.no_grad():
                prediction = model(input_image)
                mode_choice = torch.argmax(prediction, 1)
                # prediction = prediction * torch.Tensor([0.8, 0.9, 1, 1, 1, 1, 1, 1, 1])
                # prediction_argmax = torch.argmax(prediction, -1)

                # prediction_argmax = prediction_argmax * torch.Tensor([0.8, 0.9, 1, 1, 1, 1, 1, 1, 1])

                # prediction_argmax = prediction_argmax.numpy()
                #prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1, 2, 2, 0.5, 0.5, -0.00001])
                # print(prediction.shape)
                # print(prediction)


                # prediction = np.array(prediction)


                # prediction = np.array(prediction) * np.array([2, 0.1, 0.1, 0.1, 1.5, 1.8, 0.5, 0.5, 0])
                # 단순 확률로 계산시. ㄴprediction = np.array(prediction) * np.array([0.406, -0.384, -0.056, -0.048, -0.04, - 0.01, -0.038, -0.02, -0.008])
                # WEIGHTS =[0.030903154382632643, 1000.0, 0.020275559590445278, 0.013302794647291147, 0.0225283995449392, 0.025031555049932444, 1000.0, 1000.0, 0.016423203268260675]
                # 0 strait, 1 reverse, 2 left, 3 right, 4 forward+left, 5, forward+right, 6 reverse+left, 7 reverse+right, 8 nokey
                #prediction = np.array(prediction)
                # print(prediction)
                # print(prediction.shape)
                # print("Model's state_dict:")
                # for param_tensor in model.state_dict():
                #     print(param_tensor, "\t", model.state_dict()[param_tensor])
            # -0.1609, -0.0034, -0.0372

            #     mode_choice = np.argmax(prediction)


            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            motion_log.append(delta_count)
            motion_avg = round(mean(motion_log), 3)
            print('loop took {} seconds. Motion: {}. Choice: {}'.format(round(time.time() - last_time, 3), motion_avg,
                                                                        choice_picked))

            if motion_avg < motion_req and len(motion_log) >= log_len:
                print('WERE PROBABLY STUCK FFS, initiating some evasive maneuvers.')

                # 0 = reverse straight, turn left out
                # 1 = reverse straight, turn right out
                # 2 = reverse left, turn right out
                # 3 = reverse right, turn left out

                quick_choice = random.randrange(0, 4)

                if quick_choice == 0:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 1:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 2:
                    reverse_left()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 3:
                    reverse_right()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))

                for i in range(log_len - 2):
                    del motion_log[0]

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                no_keys()
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(s)
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()