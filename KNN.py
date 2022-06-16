import numpy as np 

class KNN:
    #KNN 클래스의 생성자
    def __init__(self, x_train, t_train, label_name ,K):
        #K값을 저장합니다
        self.k = K
        
        #reshpae 함수를사용해서 (x_train.shape[0], 28, 1) 차원으로 변환합니다.
        self.x_train = (x_train).reshape(x_train.shape[0], 28, 28)
        
        self.t_train = t_train
        #label_name 리스트를 정의합니다.
        self.label_name = label_name


    def getDistance(self, vector1, vector2):
        #행렬을 입력으로 받아 유클리드 distance 를 구해주는 함수입니다.
        #행렬 연산과정에서 오버플로우를 막기위해 64비트로 변환해줍니다.
        vector1 = vector1.astype('int64')
        vector2 = vector2.astype('int64')
        return np.linalg.norm(vector1 - vector2)


    #x_train 의 각 image 의 row 에서 0이 아닌 숫자의 갯수를 추출합니다
    #추출한 데이터들을 토대로 custom data set 을 만듭니다.
    def custom_weighted_vote(self, x_test, t_test,size):
        #custom 된 x_train 이 들어갈 변수 temp1 을 선언해줍니다.
        #과제 ppt 에 나온 방식대로 각 row 의 0의 갯수를 이용했습니다.
        temp1 = np.array([])

        #x_train 의 image 갯수만큼 반복해줍니다.
        for i in range(self.x_train.shape[0]):
            #x_train[i] 번째 image 의 각 row 에서 0이 아닌 숫자의 갯수를  temp1 배열에 계속 추가해줍니다.
            temp1 = np.append(temp1,np.count_nonzero(self.x_train[i], axis=1, keepdims=True))
        
        #행렬 연산을 위해 temp1 배열은 (x_train의 이미지 갯수, 28, 1) 차원으로 바꿔줍니다.
        temp1 = temp1.reshape(self.x_train.shape[0], 28, 1)

        #실제 이미지 label 값과  계산된 label 값이 일치하면 num 값을 증가 시킵니다.
        num = 0
        #distance 가 가장작은 label 에 vote 를 주기위해 길이 label_name 길이의 배열을 선언합니다.
        label_vote = np.zeros(shape=(len(self.label_name), ))


        #input data 의 image 들과 각 x_train data 사이의 distance 를 저장할 배열입니다.
        #distance 배열의 길이는 x_train data 의 image 갯수 입니다.
        distance = np.zeros(shape=(self.x_train.shape[0], ))

        x_test = x_test

        
        #x_test 를 (x_test의 image 갯수, 28, 28) 차원으로 변환합니다.
        x_test = x_test.reshape(x_test.shape[0],28,28)

        #무작위 input 값을 뽑기위해 sample 배열을 선언해줍니다.
        #numpy 의 random.ranint method 를 통해 size 길이만큼의 배열이 생깁니다.
        #sample 배열의 element 들은 0 ~t_test의길이 사잇값을 가집니다.
        sample = np.random.randint(0, t_test.shape[0], size)


         #test data 의 image 수는 size 입니다.
        for j in range(size):
            #x_test 데이터에서 sample[j]번째 image 를 무작위로 뽑습니다.
            #x_test 데이터도 가공해줍니다.
            #x_test에서 뽑은 image 에서 각 row 의 0 이 아닌갯수들을 추출하여 새로운 행렬을 만들어 temp2 에 저장해줍니다.
            temp2= np.count_nonzero(x_test[sample[j]], axis=1, keepdims=True)
            
            #customed x_train 데이터들과 customed x_test 데이터의 거리를 계산에 distance 배열에 저장합니다.           
            for i in range(self.x_train.shape[0]):
                distance[i] = self.getDistance(temp1[i], temp2)
            #distance의 각 element 를 오름차순 으로 정렬했을때
            #해당되는 인덱스가 배열로 되어있는 sortDist 를 구합니다.
            sortDist = distance.argsort()

             #이제 k 개의 distacne 가 작은 후보들을 구합니다.
            for k in range(self.k):
                #sortDist 배열에서 0에서 k-1 번째 까지의 element 들을 찾습니다.
                #t_train[sortDist[k]] 를 계산함으로서 후보들의 label 을 구합니다.
                cnadidateLabel = self.t_train[sortDist[k]]
                #weighted vote 임으로 가중치를 주어야합니다.
                #1/(distance + 1) 을 더해주는 방식으로 진행합니다
                label_vote[cnadidateLabel] += 1/(distance[sortDist[k]]+1)
            #정답과 계산된 label 값을 보기위해선 밑의 주석을 제거해 주세요
            print(sample[j],"th data result ", self.label_name[np.where(label_vote == label_vote.max())[0][0]], " label ", t_test[sample[j]])
            #계산된 label 과 정답 label 이 같다면 num에 1을 더해줍니다.
            if(t_test[sample[j]] ==  np.where(label_vote == label_vote.max())[0][0]):
                num += 1
            #다음 연산을 위해 label_vote 를 초기화 해줍니다
            label_vote = np.zeros(shape=(10, ))
        print("accuracy = ", (num/size)*100)
        print("")
        print("10000개 test data 중 ", size, "개 사용")
        return num
    def custom_majority_vote(self, x_test, t_test, size):
        #custom 된 x_train 이 들어갈 변수 temp1 을 선언해줍니다.
        #과제 ppt 에 나온 방식대로 각 row 의 0의 갯수를 이용했습니다.
        temp1 = np.array([])

        #x_train 의 image 갯수만큼 반복해줍니다.
        for i in range(self.x_train.shape[0]):
            #x_train[i] 번째 image 의 각 row 에서 0이 아닌 숫자의 갯수를  temp1 배열에 계속 추가해줍니다.
            temp1 = np.append(temp1,np.count_nonzero(self.x_train[i], axis=1, keepdims=True))
        
        #행렬 연산을 위해 temp1 배열은 (x_train의 이미지 갯수, 28, 1) 차원으로 바꿔줍니다.
        temp1 = temp1.reshape(self.x_train.shape[0], 28, 1)
        #실제 이미지 label 값과  계산된 label 값이 일치하면 num 값을 증가 시킵니다.
        num = 0
        #distance 가 가장작은 label 에 vote 를 주기위해 길이 label_name 길이의 배열을 선언합니다.
        label_vote = np.zeros(shape=(len(self.label_name), ))


        #input data 의 image 들과 각 x_train data 사이의 distance 를 저장할 배열입니다.
        #distance 배열의 길이는 x_train data 의 image 갯수 입니다.
        distance = np.zeros(shape=(self.x_train.shape[0], ))

        x_test = x_test
        
        #x_test 를 (x_test의 image 갯수, 28, 28) 차원으로 변환합니다.
        x_test = x_test.reshape(x_test.shape[0],28,28)

        #무작위 input 값을 뽑기위해 sample 배열을 선언해줍니다.
        #numpy 의 random.ranint method 를 통해 size 길이만큼의 배열이 생깁니다.
        #sample 배열의 element 들은 0 ~t_test의길이 사잇값을 가집니다.
        sample = np.random.randint(0, t_test.shape[0], size)


        #test data 의 image 수는 size 입니다.
        for j in range(size):
            #x_test 데이터에서 sample[j]번째 image 를 무작위로 뽑습니다.
            #x_test 데이터도 가공해줍니다.
            #x_test에서 뽑은 image 에서 각 row 의 0 이 아닌갯수들을 추출하여 새로운 행렬을 만들어 temp2 에 저장해줍니다.
            temp2= np.count_nonzero(x_test[sample[j]], axis=1, keepdims=True)
            
            #customed x_train 데이터들과 customed x_test 데이터의 거리를 계산에 distance 배열에 저장합니다.           
            for i in range(self.x_train.shape[0]):
                distance[i] = self.getDistance(temp1[i], temp2)
            #distance의 각 element 를 오름차순 으로 정렬했을때
            #해당되는 인덱스가 배열로 되어있는 sortDist 를 구합니다.
            sortDist = distance.argsort()

             #이제 k 개의 distacne 가 작은 후보들을 구합니다.
            for k in range(self.k):
                #sortDist 배열에서 0에서 k-1 번째 까지의 element 들을 찾습니다.
                #t_train[sortDist[k]] 를 계산함으로서 후보들의 label 을 구합니다.
                cnadidateLabel = self.t_train[sortDist[k]]
                #  1을  각 label_vote[candidataLabel]에 더합니다.
                #majority vote 는 동일한 가중치를 가짐으로 1을 더합니다.
                label_vote[cnadidateLabel] += 1
            #정답과 계산된 label 값을 보기위해선 밑의 주석을 제거해 주세요
            print(sample[j],"th data result ", self.label_name[np.where(label_vote == label_vote.max())[0][0]], " label ", t_test[sample[j]])
            #계산된 label 과 정답 label 이 같다면 num에 1을 더해줍니다.
            if(t_test[sample[j]] ==  np.where(label_vote == label_vote.max())[0][0]):
                num += 1
            #다음 연산을 위해 label_vote 를 초기화 해줍니다
            label_vote = np.zeros(shape=(10, ))
        print("accuracy = ", (num/size)*100)
        print("")
        print("10000개 test data 중 ", size, "개 사용")
        return num
    


    def weighted_vote(self, x_test, t_test, size):
         #실제 이미지 label 값과  계산된 label 값이 일치하면 num 값을 증가 시킵니다.
        num = 0
        #distance 가 가장작은 label 에 vote 를 주기위해 길이 label_name 길이의 배열을 선언합니다.
        label_vote = np.zeros(shape=(len(self.label_name), ))


        #input data 의 image 들과 각 x_train data 사이의 distance 를 저장할 배열입니다.
        #distance 배열의 길이는 x_train data 의 image 갯수 입니다.
        distance = np.zeros(shape=(self.x_train.shape[0], ))

        x_test = x_test
        
        #x_test 를 (x_test의 image 갯수, 28, 28) 차원으로 변환합니다.
        x_test = x_test.reshape(x_test.shape[0],28,28)

        #무작위 input 값을 뽑기위해 sample 배열을 선언해줍니다.
        #numpy 의 random.ranint method 를 통해 size 길이만큼의 배열이 생깁니다.
        #sample 배열의 element 들은 0 ~t_test의길이 사잇값을 가집니다.
        sample = np.random.randint(0, t_test.shape[0], size)


        #test data 의 image 수는 size 입니다.
        for j in range(size):
            #x_train 의 모든 image 와의 distance 를 구합니다.
            for i in range(self.x_train.shape[0]):
                #x_train 데이터와 x_test[sample[j]] 의 거리를 연산하여 distance 배열에 저장합니다
                distance[i] =self.getDistance(self.x_train[i], x_test[sample[j]])                
            #distance의 각 element 를 오름차순 으로 정렬했을때
            #해당되는 인덱스가 배열로 되어있는 sortDist 를 구합니다.
            sortDist = distance.argsort()

            #이제 k 개의 distacne 가 작은 후보들을 구합니다.
            for k in range(self.k):
                #sortDist 배열에서 0에서 k-1 번째 까지의 element 들을 찾습니다.
                #t_train[sortDist[k]] 를 계산함으로서 후보들의 label 을 구합니다.
                cnadidateLabel = self.t_train[sortDist[k]]
                #weighted vote 임으로 가중치를 주어야합니다.
                #1/(distance + 1) 을 더해주는 방식으로 진행합니다
                label_vote[cnadidateLabel] += 1/(distance[sortDist[k]]+1)
            #정답과 계산된 label 값을 보기위해선 밑의 주석을 제거해 주세요
            print(sample[j],"th data result ", self.label_name[np.where(label_vote == label_vote.max())[0][0]], " label ", t_test[sample[j]])
            #계산된 label 과 정답 label 이 같다면 num에 1을 더해줍니다.
            if(t_test[sample[j]] ==  np.where(label_vote == label_vote.max())[0][0]):
                num += 1
            #다음 연산을 위해 label_vote 를 초기화 해줍니다
            label_vote = np.zeros(shape=(10, ))
        
        print("accuracy = ", (num/size)*100)
        print("")
        print("10000개 test data 중 ", size, "개 사용")
        return num
    def majority_vote(self, x_test, t_test, size):
         #실제 이미지 label 값과  계산된 label 값이 일치하면 num 값을 증가 시킵니다.
        num = 0
        #distance 가 가장작은 label 에 vote 를 주기위해 길이 label_name 길이의 배열을 선언합니다.
        label_vote = np.zeros(shape=(len(self.label_name), ))


        #input data 의 image 들과 각 x_train data 사이의 distance 를 저장할 배열입니다.
        #distance 배열의 길이는 x_train data 의 image 갯수 입니다.
        distance = np.zeros(shape=(self.x_train.shape[0], ))

        x_test = x_test
        
        #x_test 를 (x_test의 image 갯수, 28, 28) 차원으로 변환합니다.
        x_test = x_test.reshape(x_test.shape[0],28,28)

        #무작위 input 값을 뽑기위해 sample 배열을 선언해줍니다.
        #numpy 의 random.ranint method 를 통해 size 길이만큼의 배열이 생깁니다.
        #sample 배열의 element 들은 0 ~t_test의길이 사잇값을 가집니다.
        sample = np.random.randint(0, t_test.shape[0], size)


        #test data 의 image 수는 size 입니다.
        for j in range(size):
            #x_train 의 모든 image 와의 distance 를 구합니다.
            for i in range(self.x_train.shape[0]):
                #x_train 데이터와 x_test[sample[j]] 의 거리를 연산하여 distance 배열에 저장합니다
                distance[i] =self.getDistance(self.x_train[i], x_test[sample[j]])                
            #distance의 각 element 를 오름차순 으로 정렬했을때
            #해당되는 인덱스가 배열로 되어있는 sortDist 를 구합니다.
            sortDist = distance.argsort()

            #이제 k 개의 distacne 가 작은 후보들을 구합니다.
            for k in range(self.k):
                #sortDist 배열에서 0에서 k-1 번째 까지의 element 들을 찾습니다.
                #t_train[sortDist[k]] 를 계산함으로서 후보들의 label 을 구합니다.
                cnadidateLabel = self.t_train[sortDist[k]]
                #  1을  각 label_vote[candidataLabel]에 더합니다.
                #majority vote 는 동일한 가중치를 가짐으로 1을 더합니다.
                label_vote[cnadidateLabel] += 1
            #정답과 계산된 label 값을 보기위해선 밑의 주석을 제거해 주세요
            print(sample[j],"th data result ", self.label_name[np.where(label_vote == label_vote.max())[0][0]], " label ", t_test[sample[j]])
            #계산된 label 과 정답 label 이 같다면 num에 1을 더해줍니다.
            if(t_test[sample[j]] ==  np.where(label_vote == label_vote.max())[0][0]):
                num += 1
            #다음 연산을 위해 label_vote 를 초기화 해줍니다
            label_vote = np.zeros(shape=(10, ))
        print("accuracy = ", (num/size)*100)
        print("")
        print("10000개 test data 중 ", size, "개 사용")
        #성공률을 출력합니다
        
        return num






