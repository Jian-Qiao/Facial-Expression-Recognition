import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Utils import getImageData, getBinaryData, y2indicator, error_rate, init_weight_and_bias
from sklearn.utils import shuffle
from sklearn import model_selection
import pickle

def convpool(X,W,b):
    conv_out=tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')
    conv_out=tf.nn.bias_add(conv_out,b)
    pool_out=tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return pool_out

def init_filter(shape,poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)

class ConvPoolLayer(object):
    def __init__(self,mi,mo,fw=5,fh=5,poolsz=(2,2)):
        sz=(fw,fh,mi,mo)
        W0=init_filter(sz,poolsz)
        self.W=tf.Variable(W0)
        b0=np.zeros(mo,dtype=np.float32)
        self.b=tf.Variable(b0)
        self.poolsz=poolsz
        self.params=[self.W,self.b]
    
    def forward(self,X):
        conv_out=tf.nn.conv2d(X,self.W,strides=[1,1,1,1],padding='SAME')
        conv_out=tf.nn.bias_add(conv_out,self.b)
        pool_out=tf.nn.max_pool(conv_out,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        return tf.tanh(pool_out)
    
class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1,M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]
    
    def forward(self,X):
        return tf.nn.relu(tf.matmul(X,self.W)+self.b)    
    
class CNN(object):
    def __init__(self,convpool_layer_sizes,hidden_layer_sizes):
        self.convpool_layer_sizes=convpool_layer_sizes
        self.hidden_layer_sizes=hidden_layer_sizes
        
    def fit(self,X,Y,learning_rate=10e-4,mu=0.99,reg=10e-4,decay=0.9999,eps=10e-3,batch_sz=30, epochs=500, show_fig=True):
        learning_rate=np.float32(learning_rate)
        mu=np.float32(mu)
        reg=np.float32(reg)
        decay=np.float32(decay)
        eps=np.float32(eps)
        K=len(set(Y))
        
        X,Y=shuffle(X,Y)
        X=X.astype(np.float32)
        Y=y2indicator(Y).astype(np.float32)
        
        Xvalid,Yvalid=X[-500:],Y[-500:]
        X,Y=X[:-500],Y[:-500]
        
        Yvalid_flat=np.argmax(Yvalid,axis=1)
        
        N,d,d,c=X.shape
        mi=c
        outw=d
        outh=d
        self.convpool_layers=[]
        
        for mo,fw,fh in self.convpool_layer_sizes:
            layer=ConvPoolLayer(mi,mo,fw,fh)
            self.convpool_layers.append(layer)
            outw=outw//2
            outh=outh//2
            mi=mo
        
        self.hidden_layers=[]
        M1=self.convpool_layer_sizes[-1][0]*outw*outh
        
        count=0
        for M2 in self.hidden_layer_sizes:
            h=HiddenLayer(M1,M2,count)
            self.hidden_layers.append(h)
            M1=M2
            count+=1
            
        
        W,b=init_weight_and_bias(M1,K)
        self.W=tf.Variable(W,'W_logreg')
        self.b=tf.Variable(b,'b_logreg')
        
        self.params=[self.W,self.b]
        for h in self.convpool_layers:
            self.params += h.params
        for h in self.hidden_layers:
            self.params += h.params
            
        tfX=tf.placeholder(tf.float32,shape=(None,d,d,c),name='X')
        tfY=tf.placeholder(tf.float32,shape=(None,K),name='Y')
        act=self.forward(tfX)
        
        rcost=reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act,labels=tfY))+rcost
        prediction=self.predict(tfX)
        
        train_op=tf.train.RMSPropOptimizer(learning_rate,decay=decay,momentum=mu).minimize(cost)
        
        n_batches=N//batch_sz
        costs=[]
        init=tf.global_variables_initializer()
        errors=[]
        
        with tf.Session() as session:
            session.run(init)
            for i in xrange(epochs):
                X,Y=shuffle(X,Y)
                for j in xrange(n_batches):
                    Xbatch= X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch= Y[j*batch_sz:(j*batch_sz+batch_sz)]
                    
                    session.run(train_op,feed_dict={tfX:Xbatch,tfY:Ybatch})
                    
                    if j%20 == 0:
                        c=session.run(cost,feed_dict={tfX:Xvalid,tfY:Yvalid})
                        costs.append(c)
                        
                        p=session.run(prediction,feed_dict={tfX:Xvalid,tfY:Yvalid})
                        e=error_rate(Yvalid_flat,p)
                        errors.append(e)
                        
                        print 'i: %d, j:%d, nb: %d, cost: %.3f, e: %.3f' %(i,j,n_batches,c,e)
                        
            
        if show_fig:
                plt.plot(costs)
                plt.show()
        
    def forward(self,X):
        Z=X
        for c in self.convpool_layers:
            Z=c.forward(Z)
        Z_shape=Z.get_shape().as_list()
        Z=tf.reshape(Z,[-1,np.prod(Z_shape[1:],)])
        for h in self.hidden_layers:
            Z=h.forward(Z)
        return tf.matmul(Z,self.W)+self.b

    def predict(self,X):
        pY=self.forward(X)
        return tf.argmax(pY,1)
    
def main():
    X, Y = getImageData()
    
    X=X.transpose((0,2,3,1))
    model=CNN(convpool_layer_sizes=[(50,5,5),(20,2,2)],
             hidden_layer_sizes=[2500,1500,750,300]
             )
    model.fit(X,Y,show_fig=False)

if __name__=='__main__':
    main()
