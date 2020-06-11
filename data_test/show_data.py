import numpy as np
import cv2
import time


# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


#Before balancing
# train_data = np.load('C:/Users/esaw2/work/Project/pygta5 data/data/training_data-352.npy')
train_data = np.load('C:/Users/esaw2/work/Project/pygta5 data/balanced data/test data/balanced testing data-1.npy')

#After balancing
# train_data = np.load('C:/Users/esaw2/work/Project/pygta5 data/balanced data/balanced training_data-1787.npy')

#1파일은 time.sleep(0.5), 2파일은 (0.2)로 했음. 3파일은 그대로
for i in range(0, 549):
    img = train_data[i][0]
    cv2.imshow('test', img)
    # time.sleep(0.05)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllwindows()
        break
#216부터는 타임슬립 없이