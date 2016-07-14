import time
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpi4py import MPI

from SandblasterMasterOptimizer import BFGSoptimizer
from OperationServer import SandblasterOpServer

from sys import path
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)
from Logger import DataLogger
p = getcwd()[0:getcwd().rfind("/")]+"/hostfile"
w =getcwd()[0:getcwd().rfind("/")]+"/BFGS_SANDBLASTER/DistributedBFGSOptimizer.py"

class DistriputedBFGSOptimizer:

    def __init__(self,logger,TrainData,TestData,learn_rate=.001,epochs=15,batch_size=100,nodes=256,layers=2,workers=3):
        self.workers = workers
        self.logger= logger
        if workers != None:
            mpi_info = MPI.Info.Create()
            mpi_info.Set('add-hostfile', p)
            self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                               args=[w],
                               maxprocs=workers, info=mpi_info)
            for i in range(workers):
                args = (logger,TrainData,TestData,learn_rate,epochs,batch_size,nodes,layers,None)
                self.comm.send(args,dest=i)
                self.rank = self.comm.Get_rank()
                self.size = self.comm.Get_size()+workers
        else:
            self.comm = MPI.Comm.Get_parent()
            self.rank = self.comm.Get_rank()+1
            self.size = self.comm.Get_size()+1

        self.train_data = TrainData
        self. test_data = TestData
        # Parameters
        self.learning_rate = learn_rate
        self.training_epochs = epochs
        self.batch_size = batch_size
        self.n_layer = layers
        self.input_size = len(TrainData[0][0])
        self.output_size =  len(TrainData[1][1])
        self.n_nodes = nodes
        self.x = tf.placeholder("float", [None,self.input_size])
        self.y = tf.placeholder("float", [None, self.output_size])
        self.biases = []
        self.weights = []
        self.weights.append(tf.Variable(tf.random_normal([self.input_size, self.n_nodes])))
        for i in range(1, self.n_layer):
            self.biases.append(tf.Variable(tf.random_normal([self.n_nodes])))
            self.weights.append(tf.Variable(tf.random_normal([self.n_nodes, self.n_nodes])))
        self.weights.append(tf.Variable(tf.random_normal([self.n_nodes, self.output_size])))
        self.biases.append(tf.Variable(tf.random_normal([self.n_nodes])))
        self.biases.append(tf.Variable(tf.random_normal([self.output_size])))
        self.pred = self.multilayer_perceptron(self.x, self.weights, self.biases)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        # Initializing the variables
        self.init = tf.initialize_all_variables()
        self.correct_prediction = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # Launch the graph
        self.config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 1},
                                    inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1)
    # Create model
    def multilayer_perceptron(self,x,weights,biases):
        prevlayer = x
        for w,b in zip(weights,biases):
            prevlayer = tf.nn.relu(tf.add(tf.matmul(prevlayer, w), b))
        return prevlayer
    def run_graph(self):
        data_x,data_y = self.train_data
        test_feed = dict()
        test_feed[self.x],test_feed[self.y] = self.test_data   
        with tf.Session(config=self.config) as sess:
            sess.run(self.init)
            if self.workers != None:
                feed={self.x:data_x[0:len(data_x)/self.size],self.y:data_y[0:len(data_x)/self.size]}
                mini=BFGSoptimizer(self.cost,feed,sess,self.rank,"xdat",self.comm) 
                start=time.time()
                for ep in range(self.training_epochs):
                    start=time.time()
                    mini.minimize(alpha=self.learning_rate)
                    end=time.time()                    
                    #test_c=cost.eval({x: mnist.test.images, y:mnist.test.labels},session=sess)
                    #train_c=cost.eval({x: data_x, y:data_y},session=sess)
                    test_acc=self.accuracy.eval(test_feed,session=sess)
                    train_acc=self.accuracy.eval({self.x: data_x, self.y:data_y},session=sess)
                    now=time.time()
                    self.logger.writeData((now-start, train_acc,test_acc))
                self.comm.scatter(["KILL" for x in range(self.comm.Get_size())],root=MPI.ROOT)
                print "Average Gradient Computation Time:", mini.get_average_grad_time()
                print "Core 0 finished."
            else:

                feed={self.x:data_x[len(data_x)/self.size*self.rank:len(data_x)/self.size*(self.rank+1)],self.y:data_y[len(data_x)/self.size*self.rank:len(data_x)/self.size*(self.rank+1)]}
                Operator=SandblasterOpServer(self.rank, "xdat", feed, sess, self.cost,self.comm)
                total_time=0
                while (True):
                    data="None"
                    data=comm.scatter(["GP" for x in range(self.comm.Get_size())],root=0)
                    if data=="None":
                        continue
                    elif data=="GP":
                        start=time.time()
                        g=Operator.Compute_Gradient()
                        c=Operator.Compute_Cost()
                        new_data = comm.gather((g,c),root=0)
                        end=time.time()
                        total_time=total_time+end-start
                    elif data=="KILL":
                        break
                print "Core,", self.rank, "Computation Cost:", total_time
                

if __name__ == "__main__":
    comm = MPI.Comm.Get_parent()
    optimize = DistriputedBFGSOptimizer(*comm.recv(source=0))
    optimize.run_graph()