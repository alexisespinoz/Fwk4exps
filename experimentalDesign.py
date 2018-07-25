from fwk4exps.SpeculativeMonitor import SpeculativeMonitor

def experimentalDesign():
   	
    params = {"__a" : 0.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S0 = Strategy('Algo0', '/home/iaraya/clp/BSG_CLP', '--alpha=__a --beta=__b --gamma=__g -p __p', params, 0)

    ##print("Creando algoritmo1")
    S1 = copy.deepcopy(S0)
    S1.params = {"__a" : 2.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S1.id=1
    S1.name='Algo1'

    ##print("Creando algoritmo2")
    S2 = copy.deepcopy(S0)
    S2.params = {"__a" : 4.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S2.id=2
    S2.name='Algo2'

    pi = '/home/iaraya/clp/instances.txt'

    Sbest = bestStrategy(S0, S1, pi, Metric.MEDIA_DIFF, 0.00)
    Sbest = bestStrategy(Sbest, S2, pi, Metric.MEDIA_DIFF, 0.00)

    print "El mejor algoritmo es: " + Sbest.name
    ##print("______________________")


sm = SpeculativeMonitor (experimentalDesign)

sm.run()