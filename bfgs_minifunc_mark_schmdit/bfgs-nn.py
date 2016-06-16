import tensorflow as tf
import numpy as np
import math
import time



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

def getParam():
    param=[]
    for k in weights:
        param.append(weights[k])
    for k in biases:
        param.append(biases[k])
    return param

def assignParam(weights,bias):
    for k in weights:
        assign_op = x.assign


param=getParam()



var_grad = tf.gradients(cost,param)
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

def minimize(cost,epoch, data_x,data_y,x,alpha,batch_size=None, verbose=True, test=False, test_x=None ,test_y=None):
    if batch_size!=None:
        if epoch*batch_size>len(data_x):
            print "Mismatching training datasets"
    rho=0.01
    sig=0.5
    INT=0.1
    ext=3.0
    nmax=20
    ratio=100
    i=0
    red=1
    ls_failed=False
    fx=[]
    f1,df1=getGradient(cost,data_x,data_y,x)
    evaluate=1
    s=-df1
    d1=-np.inner(s,s)
    z1=red/(1-d1)
    counter=0
    lc=0
    ls_limit=10
    while counter<epoch and np.linalg.norm(df1)>0.0000001:
        if (batch_size!=None):
            small_data_x=data_x[counter*batch_size:(counter+1)*batch_size]
            small_data_y=data_y[counter*batch_size:(counter+1)*batch_size]
        else:
            small_data_x=data_x
            small_data_y=data_y
        counter=counter+1
        x0=x
        f0=f1
        df0=df1
        x=x+z1*s
        f2,df2=getGradient(cost,small_data_x,small_data_y,x)
        if verbose:
            if test:
                tc,dtc=getGradient(cost,test_x,test_y,x)
                print counter,"epoch", "training cost,", f2, "testing cost", tc
            else:
                print counter,"epoch", "training cost,", f2
        evaluate=evaluate+1
        d2=np.inner(df2,s)
        f3=f1
        d3=d1
        z3=-z1
        M=nmax
        success=0
        limit=-1
        while (True):
            while ((f2>f1+z1*rho*d1 or (d2 > -sig*d1) and (M > 0)) and lc<ls_limit):
                lc=lc+1
                limit=z1
                if f2>f1:
                    z2=z3-(0.5*d3*z3*z3)/(d3*z3+f2-f3)
                else:
                    A = 6*(f2-f3)/z3+3*(d2+d3)
                    B = 3*(f3-f2)-z3*(d3+2*d2)
                    z2= (math.sqrt(B*B-A*d2*z3*z3)-B)/A
                if np.isinf(z2) or np.isnan(z2):
                    z2=z3/2
                z2=max(min(z2, INT*z3),(1-INT)*z3)
                z1=z1+z2
                x=x+z2*s
                f2,df2=getGradient(cost,small_data_x,small_data_y,x)
                evaluate=evaluate+1
                M=M-1
                d2=np.inner(df2,s)
                z3=z3-z2
            if f2>f1+z1*rho*d1 or d2>-sig*d1:
                break
            elif d2>sig*d1:
                success=1
                break
            elif M==0:
                break
            A = 6*(f2-f3)/z3+3*(d2+d3)
            B = 3*(f3-f2)-z3*(d3+2*d2)
            valid=True
            try:
                z2=-d2*z3*z3/(B+math.sqrt(B*B-A*d2*z3*z3))
            except:
                valid=False
            if (not valid) or (not np.isreal(z2)) or (np.isnan(z2)) or np.isinf(z2):
                if limit<-0.5:
                    z2=z1*(ext-1)
                else:
                    z2=(limit-z1)/2
            elif (limit>-0.5) and (z2+z1>limit):
                z2=z1*(limit-z1)/2
            elif (limit<-0.5) and (z2+z1>z1*ext):
                z2=z1*(ext-1.0)
            elif z2<-z3*INT:
                z2=-z3*INT
            elif (limit>-0.5) and (z2<(limit-z1)*(1.0-INT)):
                z2=(limit-z1)*(1.0-INT)
            f3=f2
            d3=d2
            z3=-z2
            z1=z1+z2
            x=x+z2*s
            f2,df2=getGradient(cost,small_data_x,small_data_y,x)
            evaluate=evaluate+1
            M=M-1
            d2=np.inner(df2,s)
        if success==1:
            f1=f2
            s = (np.inner(df2,df2)-np.inner(df1,df2))/(np.inner(df1,df1))*s - df2
            tmp = df1; df1 = df2; df2 = tmp
            d2=np.inner(df1,s)
            if d2>0:
                s=-df1
                d2=np.inner(-s,s)
            z1=z1 * min(ratio, d1/(d2-0.00000000000000001))
            d1 = d2
            ls_failed = 0
        else:
            x=x0
            f1=f0
            df1=df0
            if ls_failed==1:
                x=x+df1*alpha
            tmp=df1
            df1=df2
            df2=tmp
            s=-df1
            d1=-np.inner(s,s)
            z1=1/(1-d1)
            ls_failed =1
    return x

def getGradient(cost,data_x,data_y,wt):

    #vg=sess.run(var_grad, feed_dict={x: data_x, y_: data_y})
    #c=sess.run(cost, feed_dict={x: data_x, y_: data_y})
    #g1=list(np.array(vg[0]).reshape(1,7840)[0])
    #g2=list(np.array(vg[1]).reshape(1,10)[0])
    #return c,np.array(g1+g2).reshape(1,7850)



getGradient(cost,[1],[2],[1])