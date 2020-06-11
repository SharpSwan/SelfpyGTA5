import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
from datetime import datetime



w = 0
s = 1
a = 2
d = 3
wa = 4
wd = 5
sa = 6
sd = 7
nk = 8

starting_number = 1

while True:
    file_name = 'C:/Users/esaw2/work/Project/pygta5 data/data/training_data-{}.npy'.format(starting_number)

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_number)
        starting_number += 1
    else:
        print('File does not exist, starting fresh!', starting_number)

        break


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    # output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def main(file_name, starting_number):
    file_name = file_name
    starting_number = starting_number
    training_data = []

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)



    paused = False
    print('STARTING!!!')
    start_time = datetime.now()
    print('Collection start time:', start_time)



    while (True):

        if not paused:
            screen = grab_screen(region=(0, 40, 800, 625))
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480, 270))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()
            output = keys_to_output(keys)
            last_time = time.time()
            training_data.append([screen, output])


            #Just to compare time with getting screen in Test_model. Not Essential
            # print('loop took {:>.6} seconds.'.format((time.time() - last_time)))

            # time.sleep(0.001)

            # time.sleep(0.05) # 5/11  추가한 부분. 너무 연속적인 이미지는 비슷한 이미지가 많아 학습에 방해될 듯 싶어서.


            if len(training_data) % 100 == 0:
                print(len(training_data))

                if len(training_data) == 500:
                    np.save(file_name, training_data)
                    print('{}th file SAVED'.format(starting_number))
                    saving_time = datetime.now() - start_time
                    print("{} file saving took time :".format(starting_number), saving_time)
                    training_data = []
                    starting_number += 1
                    file_name = 'C:/Users/esaw2/work/Project/pygta5 data/data/training_data-{}.npy'.format(starting_number)


        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Restart!')
                time.sleep(1)

            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
                so_far_time = datetime.now() - start_time
                print("so far took time:", so_far_time)


main(file_name, starting_number)
