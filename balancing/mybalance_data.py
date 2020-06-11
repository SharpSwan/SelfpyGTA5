import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import os
import time

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


### 레이블끼리 따로 저장하는 코드 ###


#Final data number = 477
FILE_I_END = 477



    # 데이터 순서 정하고 셔플
data_order = [i for i in range(1, FILE_I_END + 1)]

straight = 0
reverse = 1
left = 2
right = 3
forward_left = 4
forward_right = 5
reverse_left = 6
reverse_right = 7
nokeys = 8

# select = nokeys

straight_data = []
reverse_data = []
left_data = []
right_data = []
forward_left_data = []
forward_right_data = []
reverse_left_data = []
reverse_right_data = []
nokeys_data = []

Goal_length = len(straight_data)


for count, i in enumerate(data_order):
    try:
        # load data
        file_name = 'C:/Users/esaw2/work/Project/pygta5 data/data/training_data-{}.npy'.format(i)
        # # full file info
        train = np.load(file_name)

        # train_X = np.array([i[0] for i in train]) X 데이터만 모으는것
        train_Y = ([i[1] for i in train]) # 전체 Y 레이블을 모아놓았음.
        # print("finish collect Y label")

        for idx, keys in enumerate(train_Y):

            if keys == 0:
                straight_data.append([train[idx][0], train[idx][1]])

            elif keys == 1:
                reverse_data.append([train[idx][0], train[idx][1]])

            elif keys == 2:
                left_data.append([train[idx][0], train[idx][1]])

            elif keys == 3:
                right_data.append([train[idx][0], train[idx][1]])

            elif keys == 4:
                forward_left_data.append([train[idx][0], train[idx][1]])

            elif keys == 5:
                forward_right_data.append([train[idx][0], train[idx][1]])

            elif keys == 6:
                reverse_left_data.append([train[idx][0], train[idx][1]])

            elif keys == 7:
                reverse_right_data.append([train[idx][0], train[idx][1]])

            elif keys == 8:
                nokeys_data.append([train[idx][0], train[idx][1]])

    except Exception as e:
        print(str(e))


reverse_data = reverse_data * 41
left_data = left_data * 22
right_data = right_data * 21
forward_left_data = forward_left_data * 11
forward_right_data = forward_right_data * 14
reverse_left_data = reverse_left_data * 728
reverse_right_data = reverse_right_data * 607
nokeys_data = nokeys_data * 2




Slicing_number = 120000

straight_data = straight_data[:Slicing_number]
reverse_data = reverse_data[:Slicing_number]
left_data = left_data[:Slicing_number]
right_data = right_data[:Slicing_number]
forward_left_data = forward_left_data[:Slicing_number]
forward_right_data = forward_right_data[:Slicing_number]
reverse_left_data = reverse_left_data[:Slicing_number]
reverse_right_data = reverse_right_data[:Slicing_number]
nokeys_data = nokeys_data[:Slicing_number]




final_data = straight_data + reverse_data + left_data + right_data + forward_left_data + forward_right_data + reverse_left_data + reverse_right_data + nokeys_data



shuffle(final_data)






starting_number = 1








while True:
    file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/balanced training_data-{}.npy'.format(starting_number)

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_number)
        starting_number += 1
    else:
        print('File does not exist, starting fresh!', starting_number)

        break

def main(file_name, starting_number):
    file_name = file_name
    starting_number = starting_number
    training_data = []


    for data in final_data:

        training_data.append(data)

        if len(training_data) % 100 == 0:
            print(len(training_data))

            if len(training_data) == 500:
                np.save(file_name, training_data)
                print('{}th file SAVED'.format(starting_number))
                training_data = []
                starting_number += 1
                file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/balanced training_data-{}.npy'.format(
                    starting_number)


print("Balancing Finished!")

main(file_name, starting_number)


# while True:
#     try:
#
#         if len(reverse_data[:]) < Goal_length:
#             reverse_data.append()
#
#
# 　　
# 　 　if len(reverse_data) == len(left_data) == len(right_data) == len(forward_left) == len(forward_right) == len(
#         reverse_left) == len(reverse_right) == len(nokeys_data) == Goal_length:
#         break


    #
    # except Exception as e:
    #     print(str(e))
    #     #
# df=pd.DataFrame(left_data)
#     # print(straight_data)
# print(df.head())
# print(Counter(df[1].apply(str)))
#     # print(straight_data[0][0])

    # 그냥 데이터를 프린트 했을때 모양
    # [[array([[[128, 91, 41],
    #           [24, 27, 29]]], dtype=uint8)
    #   0]]

    # 레이블 모으기 코드를 적용한 후 프린트 했을때 모양-> 생각을 잘못했음. 레이블은 따로 해줘서 모아야함. 해결!
    # array([array([[[129, 89, 41],
    #                [15, 16, 16]]], dtype=uint8),
    #        0], dtype=object)]



#기준이 될 length는 straight_data. 그 길이에 맞춰서 복사하고 append 하기.
# if len(file) == len(straight_data): append 멈추기










