#from enum import Enum
import copy
import commands

#class Metric(Enum):
#	MEDIA_DIFF = 1

class Strategy():
    def __init__(self,name,pathExe,args,params):
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
        #instance=self.repairInstancePath(instance)

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
        #print args
        #print "args (before replace): "+args
        for k, v in self.params.items():
            #print "k: "+(k)
            #print "v: "+str(v)
            args = args.replace(k, str(v))
        #print "args (after replace): "+args
        #print self.params.items()
        #args = args + " " + str(instance)
        #print self.pathExe+" "+args
        #result = subprocess.run([self.pathExe, args], stdout=subprocess.PIPE)
        #pathNoExe = self.pathNoExe()


        #commando = self.pathExe + " " + instance.replace('\r\n', '') + " " + args
        commando = self.pathExe + " " + instance.rstrip() + " " + args
        #print commando
        output= commands.getoutput(commando)
        output = output.splitlines()
        print "resultado: " + output[-1]
        return float(output[-1])


'''
    def pathNoExe(self):

        aux = copy.copy(self.pathExe)
        aux = aux.split("/")
        #print aux
        aux.pop()
        aux.pop()
        aux.pop(0)
        #print aux
        absolutePath = ""
        for e in aux:
            absolutePath = absolutePath + "/" + e
        return absolutePath
   
    def repairInstancePath(self, path):
        aux = copy.copy(path)
        aux = aux.split("/")
        #print aux
        aux.pop(0)
        aux.pop(0)
        #print aux
        absolutePath = ""
        for e in aux:
            absolutePath = absolutePath + "/" + e
        
        return absolutePath 
'''