from fwk4exps.SpeculativeMonitor import *
from fwk4exps.classes.Strategy import Strategy
import copy

def experimentalDesign():
   	
    params = {"__a" : 0.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S0 = Strategy('Algo0', '/home/investigador/Documentos/algoritmo100real/Metasolver/./BSG_CLP', '--alpha=__a --beta=__b --gamma=__g -p __p -t 5', params)

    ##print("Creando algoritmo1")
    S1 = copy.deepcopy(S0)
    S1.params = {"__a" : 2.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    #S1.id=1
    S1.name='Algo1'

    ##print("Creando algoritmo2")
    S2 = copy.deepcopy(S0)
    S2.params = {"__a" : 4.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    #S2.id=2
    S2.name='Algo2'

    #pi = '/home/iaraya/clp/instances.txt'
    pi = range(500,800)
    
    Sbest = bestStrategy(S0, S1, pi, 0.00)
    Sbest = bestStrategy(Sbest, S2, pi, 0.00)

    print "El mejor algoritmo es: " + Sbest.name
    ##print("______________________")

PI='/home/investigador/Documentos/algoritmo100real/Metasolver/extras/fw4exps/instancesBR.txt'
#/home/investigador/Documentos/algoritmo100real/Metasolver/extras/fw4exps/../../problems/...
sm = SpeculativeMonitor (experimentalDesign,PI)

run()