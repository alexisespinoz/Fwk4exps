from fwk4exps.SpeculativeMonitor import *
from fwk4exps.classes.Strategy import Strategy
import copy

def bestStrategyList(S0, SList, pi, delta):
    Sbest=S0
    for S in SList:
      Sbest=bestStrategy(Sbest, S, pi, delta)
      if Sbest is not S0: delta=0.0
    return Sbest
    
def bestParam(S0, param, values, pi, delta):
    SList=[]
    for p in values:
       S = copy.deepcopy(S0)
       S.params[param]=p
       S.name='Algo-'+param+"="+str(p)
       SList.append(S)
     
    return bestStrategyList(S0, SList, pi, delta)

def experimentalDesign():
   	
    params = {"__a" : 0.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S0 = Strategy('Algo0', '../Metasolver/BSG_CLP', '--alpha=__a --beta=__b --gamma=__g -p __p -t 2 --min_fr=0.98', params)

    #pi = '/home/iaraya/clp/instances.txt'
    pi = range(800,900)
         
    Sbest=bestParam(S0, "__a", [1.0, 4.0], pi, 0.00)
    print ("El mejor valor de a es: " + str(Sbest.params["__a"]))
    
    Sbest=bestParam(Sbest, "__b", [1.0, 4.0], pi, 0.00)
    print ("El mejor valor de b es: " + str(Sbest.params["__b"]))
    
    '''
    Sbest=bestParam(Sbest, "__g", [0.1, 0.4], pi, 0.00)
    print ("El mejor valor de g es: " + str(Sbest.params["__g"]) ) 
    
    Sbest=bestParam(Sbest, "__p", [0.01, 0.04], pi, 0.00)
    print ("El mejor valor de p es: " + str(Sbest.params["__p"]))
    '''
    ##print("______________________")
PI='../Metasolver/extras/fw4exps/instancesBR.txt'

#PI='/home/iaraya/clp/extras/fw4exps/instancesBR.txt'
#/home/investigador/Documentos/algoritmo100real/Metasolver/extras/fw4exps/../../problems/...
sm = SpeculativeMonitor (experimentalDesign,PI)

run()
