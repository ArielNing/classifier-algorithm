import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数

# 读取数据
def load_dataset(file_X,file_Y):
    X=np.loadtxt(file_X)
    Y=np.loadtxt(file_Y)
    Y = Y.astype(int)
    return X,Y

#将训练样本改成增广向量
def add_bias(X):
    bias=np.ones((len(X),1))
    return np.hstack((X,bias))

#处理训练样本
def init_dataset(X,Y):
    X=np.asarray(X)
    Y=np.asarray(Y)
    X=add_bias(X)
    labels=np.unique(Y)
    dim_X=X.shape[1]
    return X,Y,dim_X,len(labels)

#训练感知器
def train_perceptron(X,Y,solver,learning_rate,Max_k):
    #step1
    X,Y,dim,sum_class=init_dataset(X,Y)
    weights=np.zeros((sum_class,dim))
    n = len(X)
    isTrue = set()
    k=0 # 迭代次数
    if solver=='a': #多类情况3
        while len(isTrue)!=n: #step3 所有样本都分类正确，终止迭代
            isTrue.clear()
            for i in range(n): #step2 每次选取一个样本i，该样本属于class_i
                k=k+1
                change=False
                class_i=Y[i]
                d = np.dot(X[i], weights.transpose())
                for j in range(sum_class):
                    if class_i == j:
                        continue
                    if d[class_i] <= d[j]:
                        weights[j] = weights[j] - learning_rate * X[i]
                        change=True
                if change:
                    weights[class_i] = weights[class_i] + learning_rate * X[i]
                else:
                    isTrue.add(i)
                if k==Max_k:
                    return weights,k
        return weights,k
    if solver == 'b': #多类情况1
        while len(isTrue) != n:  # step3 所有样本都分类正确，终止迭代
            isTrue.clear()
            for i in range(n):  # step2 每次选取一个样本i，该样本属于class_i
                k = k + 1
                change = False
                class_i = Y[i]
                d = np.dot(X[i], weights.transpose())
                if d[class_i] <= 0:
                    weights[class_i] = weights[class_i] + learning_rate * X[i]
                    change = True
                for j in range(sum_class):
                    if class_i == j:
                        continue
                    if d[j] >= 0:
                        weights[j] = weights[j] - learning_rate * X[i]
                        change = True
                if not change:
                    isTrue.add(i)
                if k == Max_k:
                    return weights, k
        return weights, k

# 分类
def classify_a(x,weights):
    d=np.dot(x,weights.transpose())
    l=d.tolist()
    class_x=l.index(max(d))
    return class_x

def classify_b(x,weights):
    d=np.dot(x,weights.transpose())
    sum=len(weights) # IR=sum 不确定类
    class_x=sum
    for i in range(sum):
        if d[i]>0:
            for j in range(sum):
                if j==i:
                    continue
                if d[j]>=0:
                    class_x=sum
                    break
                class_x=i
    return class_x

# 对测试集进行测试
def test(X,weights,a):
    X=add_bias(X)
    Y=[]
    if a=='a':
        for x in X:
            class_x=classify_a(x,weights)
            Y.append(class_x)
    if a=='b':
        for x in X:
            class_x=classify_b(x,weights)
            Y.append(class_x)
    return Y

def cal_accuracy(predict_y,class_y):
    if len(predict_y)!=len(class_y):
        print("the number of dimensions does not match")
    else:
        sum=len(predict_y)
        n=0
        for i in range(sum):
            if predict_y[i]==class_y[i]:
                n=n+1
        return n/sum

if __name__ == '__main__':
    train_x='../data/X'
    train_y='../data/Y'
    test_x='../data/test_X'
    test_y='../data/test_Y'
    X,Y=load_dataset(train_x,train_y)
    weights,k=train_perceptron(X,Y,'a',1,1000000)
    print("Total iterations:"+str(k))
    for w in weights:
        print(w)
    y1 = test(X, weights,'a')
    print("Accuracy for the train set:" + str(cal_accuracy(y1, Y)))
    tx,ty=load_dataset(test_x,test_y)
    y2=test(tx,weights,'a')
    print("Accuracy for the test set:" + str(cal_accuracy(y2, ty)))
    if tx.shape[1] == 2:
        plt.title("classify for the test set", fontsize='small')
        plt.scatter(tx[:, 0], tx[:, 1], marker='o', c=y2)
        plt.show()
    else:
        fig1 = plt.figure()  # 创建一个绘图对象
        ax = Axes3D(fig1)
        ax.scatter(tx[:,0], tx[:,1], tx[:,2], marker='o', c=y2)
        plt.show()