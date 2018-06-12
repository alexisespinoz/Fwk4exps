import matplotlib.pyplot as plt #graficos
from pprint import pprint
#from classes.Algorithm import Algorithm
from classes.Tree import Tree
from classes.Strategy import Strategy
from classes.Strategy import Metric
from scipy import stats
from scipy.stats import norm,t
from prettytable import PrettyTable
import numpy as np
import copy

selected_vs_run = []
__count = None
__msg = None
__speculativeNode = None
root = None

def bestStrategy(S1, S2, pi, metric, delta_sig):
    print("______________________")
    print("entra en bestStrategy")
    global __count, __speculativeNode, __msg
    print("count: "+str(__count))
    if __count < len(__msg):
        if __msg[__count]==0:
            __count=__count+1
            return S1
        else:
            __count=__count+1
            return S2
    print "creating speculative node"
    __speculativeNode=Tree(S1,S2,None,0)
    raise ValueError

def experimentalDesign():
    print("______________________")
    print("entra en Experimental Design")

    #print("Creando algoritmo0")
    params = {"__a" : 0.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S0 = Strategy('Algo0', '/home/iaraya/clp/BSG_CLP', '--alpha=__a --beta=__b --gamma=__g -p __p', params, 0)

    #print("Creando algoritmo1")
    S1 = copy.deepcopy(S0)
    S1.params = {"__a" : 2.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S1.id=1
    S1.name='Algo1'

    #print("Creando algoritmo2")
    S2 = copy.deepcopy(S0)
    S2.params = {"__a" : 4.0, "__b" : 0.0, "__g" : 0.0, "__p" : 0.0}
    S2.id=2
    S2.name='Algo2'

    pi = '/home/iaraya/clp/instances.txt'

    Sbest = bestStrategy(S0, S1, pi, Metric.MEDIA_DIFF, 0.00)
    Sbest = bestStrategy(Sbest, S2, pi, Metric.MEDIA_DIFF, 0.00)

    print "El mejor algoritmo es: " + Sbest.name
    print("______________________")

def retrieveNode(aux):
    print("______________________")
    print("retrieve node")
    print("______________________")
    global __count, __speculativeNode, __msg

    __msg=[]

    if aux:
        __msg=aux.getMsg()
    try:
    	print "mensaje: "
        print __msg
        __count=0
        __speculativeNode = None
        experimentalDesign()
    except ValueError as x:
        print "escapando de diseno experimental"
    print("______________________")
    return __speculativeNode

def readData(results):
    f=[]
    f=np.genfromtxt(results)
    return f

def speculativeExecute():
    print("-------------------------------------------------------------")
    print("---------------------speculativeExecute----------------------")
    print("-------------------------------------------------------------")
    global root
    #v = null
    s = set()
    root = retrieveNode(None)
    if root:
        print("raiz agregada :)")
        s.add(root)

    i=0
    while (len(s)>0):
        print("-------------------------------------------------------------")
        print("iteracion: "+str(i))
        print("-------------------------------------------------------------")
        n = best(s)

        if n==None:
            print("se cumple criterio de salida")
            break
        s.add(n)
        run(n)
        update(s)
        #addChildrens(s,n)

        i=i+1
        input('Enter your input:')
        print("-------------------------------------------------------------")

def best(s):
    print("_____________________________________________")
    print("selecting best node to run")
    print("_____________________________________________")
    print("-> Largo de s(coleccion de nodos del arbol):"+str(len(s)))
    ####Parte1###
    # Selecciona el nodo no visitado u hoja con max probabilidad conjunta
    best=None
    max_pc=-1
    print("buscando rama con mayor probabilidad conjunta")
    for nod in s:
        if ((nod.isLeaf())and nod.jointProbability()>max_pc ):
            best=nod
            max_pc=nod.jointProbability()
    # porqueeeeee este if?
    if best==None:
        print("no se encontro nodo hoja?")
        return None
    print("mejor probabilidad conjunta:"+str(max_pc))
    #maxpc_vs_iter.append(max_pc)
    ####Parte2###
    #debemos encontrar el minimo de la rama
    #en primera instancia es el nodo hoja que maximiza la pj
    min_p=best.bestp1p2()
    #best2 guardara el
    #best2=Tree(None,None,None)
    #aux=Tree(None,None,None)
    #aux=best2
    aux=best
    best2=best
    print("buscando nodo de la rama con menor probabilidad")
    while best.parent!=None:
        if best.parent.left == best:
            val = best.parent.p1
        if best.parent.right == best:
            val = best.parent.p2
        if val< min_p:
            min_p = val
            best2= best.parent
        best=best.parent

    print("probabilidad minima encontrada en la rama:"+str(min_p))

    if(min_p>0.5):
        print("Solicitando nuevo nodo al arbol")
        if(aux.p1>aux.p2 and aux.left==None) or (aux.p1<=aux.p2 and aux.right==None):
            nod=retrieveNode(aux)
            if nod:
                print("nodo recibido")
                print nod.alg1.name + " " +nod.alg2.name
                if aux.p1>aux.p2:
                    aux.addLeft(nod)
                    print("id nodo agregado: ",aux.left.id)
                    return aux.left
                else:
                    aux.addRight(nod)
                    print("id nodo agregado y seleccionado: ",id(aux.right))
                    return aux.right
    print("-------------------------------------------------------------")
    if min_p == 1:
        print("min_p alcanzo valor 1")
        return None
    else:
        print("")
        selected_vs_run.append(best2.id)
        print("nodo seleccionado : ",best2.id)
        return best2

def run(n):
    print "______________________"
    print("corriendo algoritmos en nodo"+str(n.id))
    if n.visited:
        id1=n.alg1.id
        id2=n.alg2.id
        i = selectInstance(n)
        print("selected Instance: "+str(i))
        resultado_a1=resultados_experimentos[i][id1]
        resultado_a2=resultados_experimentos[i][id2]
        global_results[i][id1]=resultado_a1
        global_results[i][id2]=resultado_a2
        print("Resultado algoritmo "+str(id1)+" :"+str(resultado_a1))
        print("Resultado algoritmo "+str(id2)+" :"+str(resultado_a2))
        diferencia=resultado_a1-resultado_a2
        print("Diferencia:"+str(diferencia))

        #n.save_jp()
        #maxpc_vs_run[n.id][n.lastInstanceIndex]=n.jointProbability()
    else:
        for j in range(1,4):
            id1=n.alg1.id
            id2=n.alg2.id
            i = selectInstance(n)
            print("selected Instance: "+str(i))
            resultado_a1=resultados_experimentos[i][id1]
            resultado_a2=resultados_experimentos[i][id2]
            global_results[i][id1]=resultado_a1
            global_results[i][id2]=resultado_a2
            print("Resultado algoritmo "+str(id1)+" :"+str(resultado_a1))
            print("Resultado algoritmo "+str(id2)+" :"+str(resultado_a2))
            diferencia=resultado_a1-resultado_a2
            print("Diferencia:"+str(diferencia))
        n.visited=True
        #n.save_jp()

    if n.lastInstanceIndex == 1599:
        n.setMaxp()
    print "______________________"

            #maxpc_vs_run[n.id][n.lastInstanceIndex]=n.jointProbability()

def addChildrens(s,n):
    if n.visited==False:
        print("agregando hijos")
        if n.left!=None:
            s.add(n.left)
        if n.right!=None:
            s.add(n.right)
    n.visited=True

def selectInstance(n): #mejorar para el caso de que se haya ejecutado uno y el otro no
    print("seleccionando instancia")
    n.lastInstanceIndex=n.lastInstanceIndex + 1
    index=n.lastInstanceIndex
    i=instance_order[index]
    return i

def inicializarResultadosGlobales():
    print("inicializarResultadosGlobales")
    dimensiones=np.shape(resultados_experimentos)
    filas=dimensiones[0]
    columnas=dimensiones[1]
    globalr=[]
    for i in range(filas):
        row=[]
        for j in range(columnas):
            row.append(-1)
        globalr.append(row)
    return globalr
    #print(globalr)

def update(s):
    global root
    print("recalculando probabilidades del arbol")
    dimensiones=np.shape(resultados_experimentos)
    filas=dimensiones[0]
    #columnas=dimensiones[1]
    for n in s:
        if n.p1==1 or n.p2==1:
            continue
        recalculateMetric(n, filas)
        #n.data = n.alg1.name+"("+str(n.p1)+") vs "+n.alg2.name+"("+str(n.p2)+")"
        #n.left.pj=n.p1*
    root.printTree()

def recalculateMetric(n, filas):
    id1=n.alg1.id
    id2=n.alg2.id
    data1=[]
    data2=[]
    for i in range(0,filas):
        if(global_results[i][id1] != -1 and global_results[i][id2] != -1):
            data1.append(global_results[i][id1])
            data2.append(global_results[i][id2])
    print ("resultados alg"+str(id1)+" : ")
    print data1
    print ("resultados alg"+str(id2)+" : ")
    print data2

    mean2=np.mean(data2)
    mean1=np.mean(data1)
    variance1=np.var(data1)
    variance2=np.var(data2)
    media=mean1-mean2
    desviacion_tipica = np.sqrt((variance1/len(data1))+(variance2/len(data2)))
    print("numero de instancias:"+str(len(data1)))
    print("Media: "+str(media))
    print("Desviacion estandar:"+str(desviacion_tipica))
    if len(data1)<30:
        estadistico_t=(0-media)/desviacion_tipica
        grados_de_libertad=len(data1)-1
        p=1-t.cdf(estadistico_t,grados_de_libertad)
    else:
        p=1-norm.cdf(0, media, desviacion_tipica)
    print("calculated probability"+str(p))
    n.p1=p
    n.p2=1-p
    n.save_p()


########### Leer Archivo con Resultados ################
namedata = 'data.txt'
resultados_experimentos=readData(namedata)

########### Inicializar Resultados Globales en -1 ################

global_results=inicializarResultadosGlobales()


########### Seleccion de metrica #######################


########### Generar orden de instancias ################

np.random.seed(0)
instance_order=np.random.permutation(len(global_results))


print(instance_order)

########### Creacion de algoritmos a comparar ################

####################################################################
medias_0 = []
medias_1 = []
medias_2 = []
probability_0 = []
probability_1 = []
probability_2 = []
########### Ejecucion ################
speculativeExecute()

########### Vectores para hacer plots ####################
#maxpc_vs_iter=[]
'''
maxpc_vs_run = []

#if(root.left.jointProbability()>root.right.jointProbability()):


#Pretty PrettyTable
t = PrettyTable(['Diferencia Media','Probabilidad'])
for i in range(0,(len(medias_0)-1)):
    t.add_row([medias_0[i],probability_0[i]])
print t
'''
'''
print("num times right node was run: "+str(root.right.lastInstanceIndex))
print("num times left node was run: "+str(root.left.lastInstanceIndex))
print("num times root node was run: "+str(root.lastInstanceIndex))
print("lenght of jp calculated in right node:"+str(len(root.right.get_jp_vs_run())))
print("lenght of jp calculated in left node:"+str(len(root.left.get_jp_vs_run())))
'''

'''
(a , b )=root.left.get_p_vs_run()
plt.figure()
plt.plot(a)
plt.plot(b)
plt.title("p1 and p2 vs runs (1_left node a1 a3)")

plt.figure()
a , b =root.right.get_p_vs_run()
plt.plot(a)
plt.plot(b)
plt.title("p1 and p2 vs runs (2_right node a2 a3)")

plt.figure()
a , b =root.get_p_vs_run()
plt.plot(a)
plt.plot(b)
plt.title("p1 and p2 vs runs (0_root node a1 a2)")

plt.figure()
plt.plot(selected_vs_run)
plt.title("selected node vs iterarion")
'''
'''
plt.figure()
plt.plot(medias_0)
plt.plot(probability_0)
plt.title("Diferencia Medias y probabilidad nodo raiz(A1 VS A2)")


plt.figure()
plt.plot(medias_1)
plt.plot(probability_1)
plt.title("Diferencia Medias y probabilidad nodo IZQ(A1 VS A3)")


plt.figure()
plt.plot(medias_2)
plt.plot(probability_2)
plt.title("Diferencia Medias y probabilad nodo DERE(A2 VS A3)")


plt.show()


print(root.stat_vs_run)
#maxpc_vs_run=root.get_jp_vs_run()
#plt.plot(maxpc_vs_run)
#plt.title("max joint vs runs (right node)")
#plt.show()
'''
