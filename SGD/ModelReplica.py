__author__ = 'billywu'

import tensorflow as tf
import numpy as np


class DPSGD:
    def __init__(self,param,data,batch_size,comm,train_step,sess,x,y,cost,rank,server,accuracy,test_feed):
        self.param=param
        self.location=0
        self.batch_size=batch_size
        self.comm=comm
        self.x,self.y=data
        self.data_size=len(self.x)
        self.train_step=train_step
        self.accuracy=accuracy
        self.sess=sess
        self.tx=x
        self.ty=y
        self.cost=cost
        self.var_t=tf.trainable_variables()
        self.test_feed=test_feed
        self.var_v=[]
        self.server=server
        for t in self.var_t:
            self.var_v.append(t.eval(session=sess))
        self.rank=rank

    def next_batch(self):
        old_loc=self.location
        self.location=(self.location+self.batch_size)%self.data_size
        return self.x[old_loc:self.location], self.y[old_loc:self.location]

    def optimize(self):
        data_x,data_y=self.next_batch()
        self.sess.run(self.train_step, feed_dict={self.tx: data_x, self.ty: data_y})
        return self.sess.run(self.cost,feed_dict={self.tx: self.x, self.ty: self.y})

    def publish(self):
        delta_v=[]
        for t,v in zip(self.var_t,self.var_v):
            nv=t.eval(session=self.sess)
            delta_v.append(nv-v)
        delta_v=np.array(delta_v)
        self.comm.send(delta_v,dest=self.server,tag=11)
        waiting=True
        while waiting:
            waiting=not self.comm.Iprobe(source=self.server, tag=11)
            if not waiting:
                break
        self.var_v=self.comm.recv(source=self.server, tag=11)
        l=[]
        for v,t in zip(self.var_v,self.var_t):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))
        return self.sess.run(self.accuracy,feed_dict={self.tx: self.x, self.ty: self.y}), \
               self.sess.run(self.accuracy,feed_dict=self.test_feed)













