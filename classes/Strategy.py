from enum import Enum

class Metric(Enum):
	MEDIA_DIFF = 1

class Strategy():
    def __init__(self,name,exe,args,params,id):
        self.exe=exe
        self.args=args
        self.params=params
        self.id = id
        self.name = name

    def run(self, instance):
    	args = self.args
    	for k, v in params.items():
    		args.replace(k, v)

    	result = subprocess.run([self.exe, args], stdout=subprocess.PIPE)


