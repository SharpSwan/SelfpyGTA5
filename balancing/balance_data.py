# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

foward = []
reverse = []
left = []
right = []
left_foward = []
right_foward = []
left_reverse = []
right_reverse = []
nokey = []

shuffle(train_data)

'''
Convert keys to a ...multi-hot... array
 0  1  2  3  4   5   6   7    8
[W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
'''

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [0]:
        foward.append([img,choice])
    elif choice == [1]:
        reverse.append([img,choice])
    elif choice == [2]:
        left.append([img,choice])
    elif choice == [3]:
        right.append([img,choice])
    elif choice == [4]:
        left_foward.append([img,choice])
    elif choice == [5]:
        right_foward.append([img,choice])
    elif choice == [6]:
        left_reverse.append([img,choice])
    elif choice == [7]:
        right_reverse.append([img,choice])
    elif choice == [8]:
        nokey.append([img,choice])
    else:
        print('no matches')


forward = forward[:len(left)][:len(right)]
left = left[:len(forward)]
right = right[:len(forward)]

final_data = forward + left + right
shuffle(final_data)

np.save('training_data.npy', final_data)




