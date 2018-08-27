from fwk4exps.SpeculativeMonitor import *
from fwk4exps.classes.Strategy import Strategy
import copy

def bestStrategyList(S0, SList, pi, delta):
    Sbest=S0
    for S in SList:
      Sbest=bestStrategy(S0, S, pi, delta)
      if Sbest is not S0: delta=0.0
    return Sbest

def experimentalDesign():
   	
    params = {"__a" : 0.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S0 = Strategy('Algo0', '/home/iaraya/clp/BSG_CLP', '--alpha=__a --beta=__b --gamma=__g -p __p -t 2', params)

    ##print("Creando algoritmo1")
    S1 = copy.deepcopy(S0)
    S1.params["__a"]=1.0
    S1.name='Algo1'

    ##print("Creando algoritmo2")
    S2 = copy.deepcopy(S0)
    S2.params["__a"]=2.0
    S2.name='Algo2'

    ##print("Creando algoritmo2")
    S3 = copy.deepcopy(S0)
    S3.params["__a"]=3.0
    S3.name='Algo3'

    ##print("Creando algoritmo2")
    S4 = copy.deepcopy(S0)
    S4.params["__a"]=4.0
    S4.name='Algo4'

    #pi = '/home/iaraya/clp/instances.txt'
    pi = range(500,800)
    
    SList = {S1,S2,S3,S4}
    
    Sbest=bestStrategyList(S0, SList, pi, 0.00)
    
    #Sbest = bestStrategy(S0, S1, pi, 0.00)
    #Sbest = bestStrategy(Sbest, S2, pi, 0.00)

    print "El mejor algoritmo es: " + Sbest.name
    ##print("______________________")

PI='/home/iaraya/clp/extras/fw4exps/instancesBR.txt'
#/home/investigador/Documentos/algoritmo100real/Metasolver/extras/fw4exps/../../problems/...
sm = SpeculativeMonitor (experimentalDesign,PI)

run()
