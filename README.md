# Framework Diseño de Experimentos 
Requerimientos: 
    
    python >= 3

Descargar:

    git clone https://github.com/alexisespinoz/Fwk4exps.git
   
Instalacion:

   Windows:
    En la carpeta raiz ejecutar el siguiente comando:
        
        python -mpip install requirements.txt
        
   Linux:
    En la carpeta raiz ejecutar el siguiente comando:
        
        pip install -r requirements.txt

Editar archivo Fwk4exps/experimentalDesign.py

````
from fwk4exps.SpeculativeMonitor import *
from fwk4exps.classes.Strategy import Strategy
import copy

def experimentalDesign():
   	
    params = {"__a" : 0.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S0 = Strategy('Algo0', '/home/iaraya/clp/BSG_CLP', '--alpha=__a --beta=__b --gamma=__g -p __p -t 2', params)

    pi = range(500,800)
         
    Sbest=bestParamValue(S0, "__a", [1.0,2.0,3.0,4.0], pi, 0.04)
    print "El mejor valor de a es: " + str(Sbest.params["__a"])
   
    Sbest=bestParamValue(Sbest, "__b", [1.0,2.0,3.0,4.0], pi, 0.04)
    print "El mejor valor de b es: " + str(Sbest.params["__b"])
    
    Sbest=bestParamValue(Sbest, "__g", [0.1,0.2,0.3,0.4], pi, 0.04)
    print "El mejor valor de g es: " + str(Sbest.params["__g"])   
 
    Sbest=bestParamValue(Sbest, "__p", [0.01,0.02,0.03,0.04], pi, 0.04)
    print "El mejor valor de p es: " + str(Sbest.params["__p"])
    

PI='/home/iaraya/clp/extras/fw4exps/instancesBR.txt'
sm = SpeculativeMonitor (experimentalDesign,PI)

run()
````
