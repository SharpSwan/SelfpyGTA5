import numpy as np
import pandas as pd
from collections import Counter

# w = 0
# s = 1
# a = 2
# d = 3
# wa = 4
# wd = 5
# sa = 6
# sd = 7
# nk = 8
#
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
#
# # load data
# file_name = 'C:/Users/esaw2/work/Project/pygta5 data/data/training_data-1.npy'
# # # full file info
# train = np.load(file_name)
#
# label = train[1]
#
# # print([i[1] for i in train])
#
# df = pd.DataFrame(train)
#
# print(df.head())
# print(Counter(df[1].apply(str)))
# print(train.shape)


"""
[밸런싱 작전]

**********  FIRST STEP  *************


w = 0
s = 1
a = 2
d = 3
wa = 4
wd = 5
sa = 6
sd = 7
nk = 8


1. 레이블 전체의 갯수를 센다.

2. 100% 확률로 w = 0 의 갯수가 제일 많다.  나머지 갯수를 세서 w = 0 에서 뺀만큼의 부족한 숫자를 구한다.

3. 그 숫자만큼 전체 데이터를 복사하여 새로운 파일을 만든다. 

4. 그 다음에 돌려주면 됨.

5. 그렇게 될 경우, 이미지 파일의 갯수는 w = 0 의 갯수 X 9 가 된다.

6. 전체 넘파이 파일에서 w = 0 의 갯수만 구하는 코드를 일단 만들어야 함.

6. target 이미지 데이터의 숫자는 최소 300,000이다.

7. 300,000 % 9 를 한 숫자만큼의 w = 0 데이터가 생기면 데이터 수집을 멈춘다. 
   = 33,333 개. 그러나 이미 100,000개가 넘어버림. 즉 여기서 멈춰도 가능할듯.

8. 일단 straight의 갯수만 구하는 코드를 만들어보자.

9.  코드 작성

파일을 계속 load 하는  for loop 코드를 작성 후, 거기서 0의 레이블을 count하여 
한 파일이 끝나면 0 레이블이 나온 숫자를 구하고 그걸 다음 루프를 돌릴 때 더하는 식으로 전체 파일의 레이블 갯수를 구할 수 있다.

이때 부족한 숫자 갯수를 구한 후, 거기서 필요한만큼 자가복제를 한다.

그 후, 넘파이 파일을 불러와서 전체를 다 더해준 후 셔플 한 다음에, 특정 길이만큼 슬라이스하여 다시 저장한다.

"""

###  Count specific label code  ###

FILE_I_END = 831


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

select = straight
# select = reverse
# select = left
# select = right
# select = forward_left
# select = forward_right
# select = reverse_left
# select = reverse_right
# select = nokeys


Count_straight : int = 0
Count_reverse : int = 0
Count_left : int = 0
Count_right : int = 0
Count_forward_left : int = 0
Count_forward_right : int = 0
Count_reverse_left : int = 0
Count_reverse_right : int = 0
Count_nokeys : int = 0

Count_move_order: int = 0
for count, i in enumerate(data_order):
    try:
        # load data
        file_name = 'E:/project pygta5 data backup/data/training_data-{}.npy'.format(i)
        # # full file info
        train = np.load(file_name)

        # train_X = np.array([i[0] for i in train]) X 데이터만 모으는것
        train_Y = ([i[1] for i in train])
        Count_straight += train_Y.count(straight)
        Count_reverse += train_Y.count(reverse)
        Count_left += train_Y.count(left)
        Count_right += train_Y.count(right)
        Count_forward_left += train_Y.count(forward_left)
        Count_forward_right += train_Y.count(forward_right)
        Count_reverse_left += train_Y.count(reverse_left)
        Count_reverse_right += train_Y.count(reverse_right)
        Count_nokeys += train_Y.count(nokeys)
        # print(train_Y)
        # Count_move_order += train_Y.count(select)
        # print('{}-file contain'.format(i), select, ':', train_Y.count(select))
        # print('Count Order so far ', Count_move_order)
    except Exception as e:
        print(str(e))

    print('\b')
    print('{} th file count:'.format(i))
    print('Count_straight so far ', Count_straight)
    print('Count_reverse so far ', Count_reverse)
    print('Count_left so far ', Count_left)
    print('Count_right so far ', Count_right)
    print('Count_forward_left so far ', Count_forward_left)
    print('Count_forward_right so far ', Count_forward_right)
    print('Count_reverse_left so far ', Count_reverse_left)
    print('Count_reverse_right so far ', Count_reverse_right)
    print('Count_nokeys so far ', Count_nokeys)

#6/6 이후 데이터
""" count straight = 0
477-file contain 0 : 319
Count Order so far  123522
"""

""" count reverse = 1
477-file contain 1 : 5
Count Order so far  2959

need to multiply 40 = 118,360
2959 x 41 = 121,319‬
"""

""" count left = 2
477-file contain 2 : 20
Count Order so far  5520

need to multiply 22
5520 x 22 = 121,440
"""

""" count right = 3
477-file contain 3 : 4
Count Order so far  5735

need to multiply 21
5735 x 21 = 120,435
"""

""" count forward_left = 4
477-file contain 4 : 25
Count Order so far  11589

need to multiply 11

11589 x 11 = 127,479
"""

""" count forward_right =5
477-file contain 5 : 15
Count Order so far  9272

need to multiply 14
9272 x 14 = 129,808‬
"""

""" count reverse_left = 6
477-file contain 6 : 0
Count Order so far  165

need to multiply 728
165 x 728 = 120,120‬‬
"""

""" count reverse_right = 7
477-file contain 7 : 0
Count Order so far  198

need to multiply 607
198 x 607 = 120,186‬
"""

""" count nokeys = 8
477-file contain 8 : 112
Count Order so far  79540

need to multiply

79540 x 2 = 159,080‬
"""





#6/5 이전 데이터
""" count straight = 0
354-file how many include 0 is 337
Count straight so far  106517
"""

""" count reverse = 1
354-file how many include 1 is 27
Count reverse so far  5283

need to multiply 19 = 100,377
5283 x 19 = 100,377
"""

""" count left = 2
354-file how many include 2 is 16
Count left so far  3100

need to multiply 32
3100 x 32 = 99,200
"""

""" count right = 3
354-file how many include 3 is 4
Count straight so far  2875

need to multiply 34
2875 x 34 = 97,750
"""

""" count forward_left = 4
354-file how many include 4 is 11
Count Order so far  9948

need to multiply 10

9948 x 10 = 99,480
"""

""" count forward_right =5
354-file how many include 5 is 34
Count Order so far  9293

need to multiply 10
9293 x 10 = 92,930
"""

""" count reverse_left = 6
354-file how many include 6 is 0
Count Order so far  285

need to multiply 350
285 x 350 = 99,750
"""

""" count reverse_right = 7
354-file how many include 7 is 1
Count Order so far  379

need to multiply
379 x 263 = 99,677
"""

""" count nokeys = 8
354-file how many include 8 is 70
Count Order so far  39320

need to multiply

39320 x 2.5 = 98,300
"""

"""
**********  SECOND STEP  *************
필요한만큼 자가복제를 한다.

how to? 일단 for loop 코드를 만들어서 레이블끼리 따로 저장.

그 후 필요한 레이블마다 자가복제.

그 다음에 전체를 다 더해준 후 셔플 한 다음에, 특정 길이만큼 슬라이스하여 다시 넘파이로 저장
"""

"""
7/1 데이터 숫자
831 th file count:
Count_straight so far  230039
Count_reverse so far  8242    13
Count_left so far  8620    12
Count_right so far  8610   12
Count_forward_left so far  21537   5
Count_forward_right so far  18565  6
Count_reverse_left so far  450    223
Count_reverse_right so far  577   174
Count_nokeys so far  118860


"""