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
FILE_I_END = 216



    # 데이터 순서 정하고 셔플
data_order = [i for i in range(1, FILE_I_END + 1)]

final_test_data = []

for count, i in enumerate(data_order):
    try:
        # load data
        file_name = 'E:/project pygta5 data backup/data/training_data-{}.npy'.format(i)
        # # full file info
        test = np.load(file_name)

        for idx, test_set in enumerate(test):

            final_test_data.append(test_set)

    except Exception as e:
        print(str(e))



shuffle(final_test_data)

print(len(final_test_data))


starting_number = 1



while True:
    file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/test data/balanced testing data-{}.npy'.format(starting_number)

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_number)
        starting_number += 1
    else:
        print('File does not exist, starting fresh!', starting_number)

        break

def main(file_name, starting_number):
    file_name = file_name
    starting_number = starting_number
    test_data = []


    for data in final_test_data:

        test_data.append(data)

        if len(test_data) % 10 == 0:
            print(len(test_data))

            if len(test_data) == 50:
                np.save(file_name, test_data)
                print('{}th file SAVED'.format(starting_number))
                test_data = []
                starting_number += 1
                file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/test data/balanced testing data-{}.npy'.format(
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










