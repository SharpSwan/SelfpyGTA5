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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

device = 'cuda'

# GAME_WIDTH = 1024
# GAME_HEIGHT = 793

GAME_WIDTH = 1024
GAME_HEIGHT = 793

how_far_remove = 800
rs = (20, 15)
log_len = 25

motion_req = 800
motion_log = deque(maxlen=log_len)

WIDTH = 480
HEIGHT = 270

choices = deque([], maxlen=5)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)


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


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


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


model = models.ResNet(block = BasicBlock, layers = [2, 2, 2, 2]).to(device)


model.load_state_dict(torch.load("C:/Users/esaw2/work/Project/pygta5 data/balanced data/saved model/saved_40-epoch_model.pt", map_location=device))


model.to(device)

model.eval()

print('Loading Model!!!!')


def main():
    last_time = time.time()
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    print('Start Self driving!!!!')

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
            delta_count = motion_detection(t_minus, t_now, t_plus)

            t_minus = t_now
            t_now = t_plus
            t_plus = screen
            t_plus = cv2.blur(t_plus, (4, 4))

            screen = torch.FloatTensor(screen)

            input_image = screen.view(-1, 3, WIDTH, HEIGHT)

            #모델에 데이터를 제공하는 input Tensor를 cuda최적화
            input_image = input_image.to(device)


            with torch.no_grad():
                prediction = model(input_image)
                mode_choice = torch.argmax(prediction, 1)

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
            print('Scene loop took {} seconds. Motion: {}. Choice: {}'.format(round(time.time() - last_time, 3), motion_avg,
                                                                        choice_picked))

            if motion_avg < motion_req and len(motion_log) >= log_len:
                print('Probably stuck something, inisiating evasion.')

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
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(s)
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

if __name__ == "__main__":
    main()
