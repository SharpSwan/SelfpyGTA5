import numpy as np
import pandas as pd
from collections import Counter

w = 0
s = 1
a = 2
d = 3
wa = 4
wd = 5
sa = 6
sd = 7
nk = 8

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


#before balancing
# file_name = 'C:/Users/esaw2/work/Project/pygta5 data/data/training_data-50.npy'
file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/test data/balanced testing data-1.npy'


# balanced load data
# file_name = 'C:/Users/esaw2/work/Project/pygta5 data/balanced data/balanced training_data-1787.npy'


# # full file info
train = np.load(file_name)

label = train[1]

# print([i[1] for i in train])

df = pd.DataFrame(train)

print(df.head())
print(Counter(df[1].apply(str)))
print(len(train))
# print(train[0][1])
#print(train)
# print(train.shape)
# # print(train)

#                                                    0  1
# 0  [[[129, 89, 41], [181, 114, 80], [180, 113, 79...  0
# 1  [[[129, 89, 41], [181, 114, 80], [180, 113, 79...  0
# 2  [[[129, 89, 41], [181, 114, 80], [180, 113, 79...  0
# 3  [[[129, 89, 41], [181, 114, 80], [180, 113, 79...  0
# 4  [[[129, 89, 41], [181, 114, 80], [180, 113, 79...  0