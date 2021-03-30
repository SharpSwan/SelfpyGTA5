# Self_driving-GTA5-with-Pytorch  

![GTA5](https://github.com/SharpSwan/pygta5web/blob/master/gta5_self_driving.jpg)

## Requirements  

---

*Python 패키지

```
* python == 3.6 이상
* pytorch
* numpy
* win32api
```

*게임 내 설정

```
* script hook v
* LeFixspeedo
* 
```
## Dataset

---
`1.collect_data.py` 파일로 직접 운전을 하며 데이터를 모아야합니다.

데이터는 `[[이미지,명령]]`의 넘파이로 기록되며 `len = 500`일때 npy파일로 저장됩니다.

데이터 셋을 모을 때 주의할 점은 희소 데이터를 유의해서 모으는 것과, 고속도로 주행 데이터와 시내 운전 데이터를 밸런스 있게 모으는 것 입니다.

일반적으로 운전하며 데이터를 모을시 `w(straight)`가 너무 많이 모이는 문제가 발생하므로 `wa(forward-left)`, `wd(forward-right)`를 의식적으로 더 많이 모을 수 있게 주의합니다.

데이터는 이미지 `300,000` 정도면 유의미한 학습이 가능하다고 하나 충분한 성능을 내기 위해서는 이미지 `1,000,000` 정도가 필요하다고 합니다.

npy 파일로 500개 정도이나, 이 정도의 데이터를 밸런스있게 모으는건 상당한 시간이 들어가므로 `straight`를 `100,000` 장 정도 모은 후에는 나머지 데이터를 모으는데 더 집중해야 합니다.



## Learning Model  

---

Resnet

softmax->hyperbolic tangent

Adam optimizer

initial learning rate : 0.001

MSE loss function

training_epochs = 40

batch_size = 100

weather: sunlight

Shuffle = True

Car: FUTO

Using RTX 2080ti : it will takes more than 40 hours.

## References 

---

> [1] H. Kinsley, “Python Plays GTA V,” Python Programming Tutorials  
https://pythonprogramming.net/next-steps-python-plays-gta-v/

> [2] Beyond Grand Theft Auto V for Training, Testing and Enhancing Deep Learning in Self Driving Cars  
https://arxiv.org/pdf/1712.01397.pdf

## Author

---

[Lee sang woo](https://sharpswan.github.io/) 

### Website

https://sharpswan.github.io/pygta5web/index.html
