from classifier import Perceptron as pct
import numpy as np

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
    if(len(Y.shape)>1):
        labels=np.unique(Y[:,0])
    else:
        labels=np.unique(Y)
    dim_X=X.shape[1]
    return X,Y,dim_X,len(labels)

def cal_add(labels):
    add=[]
    add.append(0) # 首个类不用叠加
    for i in range(1,len(labels)):
        add_i=0
        for j in range(i):
            add_i=add_i+labels[j]
        add.append(add_i)
    return add

def cal_dict(num_subclass):
    assign = dict()  # 类和子类的分配表 子类：对应的父类
    index = dict()  # 类的子类的索引表 父类：对应子类的开始和结束索引
    add=cal_add(num_subclass)
    for c in range(len(num_subclass)):
        start_i = add[c]
        end_i = start_i + num_subclass[c]
        index[c] = [start_i, end_i]
        for i in range(start_i, end_i):
            assign[i] = c
    return assign,index

def create_subclass(Y):
    labels=np.unique(Y[:,0])
    sublabels=[]
    for l in labels:
        s=Y[Y[:,0]==l][:,1]
        sublabels.append(max(s)+1)
    add=cal_add(sublabels)
    assign,index=cal_dict(sublabels)
    for l in labels:
        for y in Y:
            if y[0]==l:
                y[1]=add[l]+y[1]
    return Y,assign,index

def PLdiscriminant(X,Y,learning_rate,Max_k,num_subclass): #num_subclass=[l1,l2,...li]
    if len(Y.shape)>1:
        if Y.shape[1]==2: # class subclass       situation1:已知子类划分
            Y,assign,index=create_subclass(Y)
            new_class=Y[:,1]
            weights,k=pct.train_perceptron(X,new_class,'a',learning_rate,Max_k)
            return weights,k,assign
    elif num_subclass is not None: # situation2：已知子类数目
        X, Y, dim, sum_class = init_dataset(X, Y)
        sum_subclass=np.sum(num_subclass) # 所有子类总数目
        weights = np.zeros((sum_subclass, dim))
        assign,index=cal_dict(num_subclass)
        n=len(X)
        isTrue = set()
        k=0
        while len(isTrue)!=n: #step3 所有样本都分类正确，终止迭代
            isTrue.clear()
            for i in range(n): #step2 每次选取一个样本i，该样本属于class_i
                k=k+1
                change=False
                class_i=Y[i]
                d = np.dot(X[i], weights.transpose())
                start=index[class_i][0]
                end=index[class_i][1]
                di_max=max(d[start:end])
                di_index=np.array(range(start,end))
                change_=set() # 需要进行重新计算的父类集合
                for j in range(sum_subclass):
                    if j in di_index:
                        continue
                    if di_max<=d[j]:
                        change_.add(assign[j])
                        change=True
                if change:
                    l = d.tolist()
                    for s in change_:
                        start_s=index[s][0]
                        end_s=index[s][1]
                        ds_max=max(d[start_s:end_s]) # 取父类中最大子类值
                        ds_max_index=l[start_s:end_s].index(ds_max)+start_s # 相应的索引
                        weights[ds_max_index]=weights[ds_max_index]-learning_rate*X[i]
                    di_max_index=l[start:end].index(di_max)+start # 该父类最大子类值
                    weights[di_max_index] = weights[di_max_index] + learning_rate * X[i]
                else:
                    isTrue.add(i)
                if k==Max_k:
                    return weights,k,assign
        return weights,k,assign

def classify(x,weights,assign):
    d = np.dot(x, weights.transpose())
    l = d.tolist()
    subclass_x = l.index(max(d))
    return assign[subclass_x]

def test(X,weights,assign):
    X=add_bias(X)
    Y = []
    for x in X:
        class_x=classify(x,weights,assign)
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
    train_x='../data/X2'
    train_y='../data/Y3'
    X, Y = load_dataset(train_x, train_y)
    weights,k,assign=PLdiscriminant(X,Y,1,1000,[3,3,3,3])
    print("total iterations:" + str(k))
    for w in weights:
        print(w)
    y1 = test(X, weights, assign)
    print("Accuracy for the train set:" + str(cal_accuracy(y1, Y)))