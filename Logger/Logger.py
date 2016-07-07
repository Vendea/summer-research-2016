import os.path
import datetime
import datetime as dt 


class DataLogger:
	def __init__(self,ExperimentTitle,layers,nodes,header):
		self.f = open(
			ExperimentTitle+ dt.datetime.now().strftime("%Y%m%d%H%M%S")+".csv",
			'w')
		self.ExperimentTitle = ExperimentTitle
		# self.f.write("Layers: "+str(layers)+"\n")
		# self.f.write("Nodes: "+str(nodes)+"\n")
		# self.f.write("========================================================"+"\n")
		self.f.write(header+"\n")
		self.f.flush()
	def writeData(self,cols):
		if self.f.closed:
			self.f = open(
			self.ExperimentTitle+".csv",
			'a')
		data = str(cols)[1:]
		data = data[:-1]
		self.f.write(data+"\n")
		self.f.flush()

