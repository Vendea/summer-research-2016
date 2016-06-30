import os.path
import datetime


class DataLogger:
	def __init__(self,ExperimentTitle,layers,nodes):
		current_time = datetime.datetime.now().time()
		self.f = open(
			ExperimentTitle+"-"+str(current_time.isoformat())+".csv",
			'w')
		self.f.write("Layers: "+str(layers)+"\n")
		self.f.write("Nodes: "+str(nodes)+"\n")
		self.f.write("========================================================"+"\n")
		self.f.write("Epoch,Train_Cost,Test_Cost,Computation_Time,Train_Accuracy,Test_Accuracy"+"\n")
		self.f.flush()
	def writeData(self,e,trainc,testc,t,trainacu,testacu):
		self.f.write(str(e)+","+str(trainc)+","+str(testc)+","+str(t)+","+str(trainacu)+","+str(testacu)+"\n")
		self.f.flush()
