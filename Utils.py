
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd


# In[4]:

def init_weight_and_bias(M1,M2):
    W = np.random.randn(M1,M2)/np.sqrt(M1+M2)
    b=np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


# In[5]:

w, b =init_weight_and_bias(5,7)


# In[8]:

def relu(x):
    return x * (x>0)


# In[9]:

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[10]:

def softmax(X):
    expX=np.exp(X)
    return expX/expX.sum(axis=1,keepdims=True)


# In[1]:

def sigmoid_cost(T,Y):
    return -(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()


# In[2]:

def cost(T,Y):
    return -(T*np.log(Y)).sum()


# In[3]:

def cost2(T,Y):
    N=len(T)
    return -np.log(Y[np.arange(N),T]).sum()


# In[24]:

def error_rate(targets, predictions):
    return np.mean(targets != predictions)


# In[28]:

def y2indicator(y):
    N=len(y)
    K=len(set(y))
    ind=np.zeros((N,K))
    for i in xrange(N):
        ind[i,y[i]]=1
    return ind


# In[ ]:

def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


# In[29]:

def getImageData():
    X,Y=getData()
    N,D=X.shape
    d=int(np.sqrt(D))
    X=X.reshape(N,1,d,d)
    return X,Y

def getBinaryData():
    Y=[]
    X=[]
    first=True
    for line in open('fer2013.csv'):
        if first:
            first=False
        else:
            row=line.split(',')
            y=int(row[0])
            if y==0 or y==1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X)/255.0 , np.array(Y)
