# Self_driving-GTA5-with-Pytorch  

---

# Requirements  

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
### Dataset

---
1.collect_data.py 파일로 직접 운전을 하며 데이터를 모아야합니다.
데이터 셋을 모을 때 주의할 점은 희소 데이터를 유의해서 모으는 것과, 고속도로 주행 데이터와 시내 운전 데이터를 밸런스 있게 모으는 것 입니다.

일반적으로 운전시 


### Learning Model  

---

direct perception CNN -> Alex net

softmax->hyperbolic tangent

Adam optimizer

initial learning rate : 0.001

MSE loss function

21epochs,  approximately 900,000 images with batch size of 32

weather: sunlight

maybe it will takes about for 40 hours.

car: FUTO

## References 

---

> [1] H. Kinsley, “Python Plays GTA V,” Python Programming Tutorials.
https://pythonprogramming.net/next-steps-python-plays-gta-v/

## Author

---

![Lee sang woo](https://sharpswan.github.io/)
