import sys, os
sys.path.append(os.pardir)
# 부모 디렉토리에서 import할 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
# mnist data load할 수 있는 함수 import
from PIL import Image
import matplotlib.pyplot as plt
import KNN


(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=False)

label_name =  ['0','1','2','3','4','5','6','7','8','9']


#knn클래스 생성 방법
#KNN.KNN(x_train, t_train, label_name, K값)
K = 11
knn = KNN.KNN(x_train, t_train, label_name,K)

#size 는 테스트 할 image 숫자 입니다. 1 - 1000 사이의 값을 입력해 주세요.
size = 10



#knn 함수의 3가지 parameter 설명
#(x_test,      t_test,      x_test 에서 무작위로 뽑을 image갯수)


#custom 함수는 초반 이미지 가공시간이 걸립니다.
#test 하고싶은 아래의 함수의 주석을 제거해 주세요.

knn.weighted_vote(x_test, t_test, size)
#knn.majority_vote(x_test, t_test, size)
#knn.custom_weighted_vote(x_test, t_test, size)    
#knn.custom_majority_vote(x_test, t_test, size)


