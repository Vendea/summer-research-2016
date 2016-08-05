__author__ = 'billywu'

import numpy as np
class ParamServer:
    def __init__(self,param,comm):
        self.param=np.array(param)
        self.comm=comm

    def update(self,delta_param):
        self.param=self.param+delta_param

    def next_request(self,cores):
        waiting=True
        while  (waiting):
            for core in cores:
                if  self.comm.Iprobe(source=1, tag=13):
                    return  0,0
                waiting=not self.comm.Iprobe(source=core, tag=11)
                if not waiting:
                    break
        data=self.comm.recv(source=core, tag=11)
        return core, data

    def handle_request(self,core,data):
        self.update(data)
        self.comm.send(self.param,core,tag=11)
