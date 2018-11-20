#from enum import Enum
import copy
import subprocess
#class Metric(Enum):
#	MEDIA_DIFF = 1

class Strategy():
    def __init__(self,name,pathExe,args,params):
        #path_exe = '/home/investigador/Documentos/algoritmo100real/Metasolver/./BSG_CLP'
        self.pathExe=pathExe
        self.args=args
        self.params=params
        #self.id = id
        self.name = name
        
    def __hash__(self):
        params = tuple(self.params.values())
        return hash((self.pathExe,self.args,self.name)+params)

    def __eq__(self,other):
        return self.pathExe == other.pathExe and self.args == other.args and self.params == other.params and self.name == self.name 

    def run(self, instance, PI):
        #PI = '/home/investigador/Documentos/algoritmo100real/Metasolver/extras/fw4exps/instancesBR.txt'
        aux=copy.copy(PI)
        aux=aux.split('/')
        aux.pop()
        aux.pop(0)
        PI=""
        for e in aux:
            PI=PI+"/"+e
        PI=PI+"/"
        instance=PI+instance
        args = self.args
        for k, v in self.params.items():
            args = args.replace(k, str(v))

        commando = self.pathExe + " " + instance.rstrip() + " " + args
        output= subprocess.getoutput(commando)
        output = output.splitlines()
        print ("resultado: " + output[-1])
        return float(output[-1])


