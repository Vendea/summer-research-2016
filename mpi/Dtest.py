import tensorflow as tf 
from mpi4py import MPI

comm_global = MPI.COMM_WORLD
rank = comm_global.Get_rank()
size = comm_global.Get_size()
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=rank)
with tf.device("/job:local/task:0"):
     global_step = tf.Variable(0)
     saver = tf.train.Saver()
     summary_op = tf.merge_all_summaries()
init_op = tf.initialize_all_variables()
sv = tf.train.Supervisor(is_chief=(rank==0),
                         logdir="/tmp/train_logs",
                         init_op=init_op,
                         summary_op=summary_op,
                         saver=saver,
                         global_step=global_step,
                         save_model_secs=600)
with sv.PrepareSession(server.target) as sess:
	print"worked",rank


