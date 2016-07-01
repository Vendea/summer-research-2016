import os.path
import datetime
 

class DataLogger:
	def __init__(self,ExperimentTitle,layers,nodes,testing,header):
		self.f = open(
			ExperimentTitle+".csv",
			'w')
		self.f.write("Layers: "+str(layers)+"\n")
		self.f.write("Nodes: "+str(nodes)+"\n")
		self.f.write("========================================================"+"\n")
		self.f.write(header+"\n")
		self.f.flush()
	def writeData(self,cols):
		data = str(cols)[1:]
		data = data[:-1]
		self.f.write(data+"\n")
		self.f.flush()