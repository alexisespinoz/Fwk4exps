import matplotlib.pyplot as plt #graficos
#from p#print import p#print
#from classes import noprint
from classes.detectInput import KeyPoller
#from classes.Plotter import Plotter
#from classes.Algorithm import Algorithm
from classes.Tree import Tree
from classes.Strategy import Strategy
from classes.Strategy import Metric
from scipy import stats
from scipy.stats import norm,t
from prettytable import PrettyTable
import numpy as np
import copy
import time
from random import randint
##para evitar que imprima los print
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
#####################################
class SpeculativeMonitor(object):
    """docstring for SpeculativeMonitor"""
    def __init__(self, experimentalDesign):
        self.arg = arg
        self.selected_vs_run = []
        self.__count = None
        self.__msg = None
        self.__speculativeNode = None
        self.root = None
        self.experimentalDesign = experimentalDesign

def bestStrategy(S1, S2, pi, metric, delta_sig):
    ##print("______________________")
    ##print("entra en bestStrategy")
    global __count, __speculativeNode, __msg
    ##print("count: "+str(__count))
    if __count < len(__msg):
        if __msg[__count]==0:
            __count=__count+1
            return S1
        else:
            __count=__count+1
            return S2
    ##print "creating speculative node"
    __speculativeNode=Tree(S1,S2,None,0)
    raise ValueError

def retrieveNode(aux):
    ##print("______________________")
    ##print("retrieve node")
    ##print("______________________")
    global __count, __speculativeNode, __msg

    __msg=[]

    if aux:
        __msg=aux.getMsg()
    try:
    	##print "mensaje: "
        ##print __msg
        __count=0
        __speculativeNode = None
        experimentalDesign()
    except ValueError as x:
        print ""
        ##print "escapando de diseno experimental"
    ##print("______________________")
    return __speculativeNode

def readData(results):
    f=[]
    f=np.genfromtxt(results)
    return f

def speculativeExecute():
    ##print("-------------------------------------------------------------")
    ##print("---------------------speculativeExecute----------------------")
    ##print("-------------------------------------------------------------")
    global root
    #v = null
    s = set()
    root = retrieveNode(None)
    if root:
        ##print("raiz agregada :)")
        s.add(root)
    runNode(root)
    update(s)
    i=0
    with KeyPoller() as keyPoller:
        while (len(s)>0):
            ##print("-------------------------------------------------------------")
            ##print("iteracion: "+str(i))
            ##print("-------------------------------------------------------------")

            c = keyPoller.poll()
            #print("lalallalaalala")
            pressed=False
            #pressed = True
            if not c is None:
                if c=="\n":
                    pressed=True

            if pressed:
                n = best(s)
                if n==None:
                    ##print("se cumple criterio de salida")
                    break
                s.add(n)
                runNode(n)
                update(s)
                i=i+1
                print("iteracion: "+str(i))
                #root.printTree()
            else:
                with suppress_stdout():
                    n = best(s)
                    if n==None:
                        ##print("se cumple criterio de salida")
                        break
                    s.add(n)
                    runNode(n)
                    update(s)
                    i=i+1
                    print("iteracion: "+str(i))
                    root.printTree()

        ##print("-------------------------------------------------------------")

def best(s):
    ##print("_____________________________________________")
    ##print("selecting best node to run")
    ##print("_____________________________________________")
    ##print("-> Largo de s(coleccion de nodos del arbol):"+str(len(s)))
    ####Parte1###
    # Selecciona el nodo no visitado u hoja con max probabilidad conjunta
    best=None
    max_pc=-1
    ##print("buscando rama con mayor probabilidad conjunta")
    for nod in s:
        if ((nod.isLeaf())and nod.jointProbability()>max_pc ):
            best=nod
            max_pc=nod.jointProbability()
    # porqueeeeee este if?
    if best==None:
        ##print("no se encontro nodo hoja?")
        return None
    ##print("mejor probabilidad conjunta:"+str(max_pc))
    #maxpc_vs_iter.append(max_pc)
    ####Parte2###
    #debemos encontrar el minimo de la rama
    #en primera instancia es el nodo hoja que maximiza la pj
    max_pvalue=99999999999999#best.lastInstanceIndex#pvalue
    #best2 guardara el
    #best2=Tree(None,None,None)
    #aux=Tree(None,None,None)
    #aux=best2
    aux=best
    best2=best
    ##print("buscando nodo de la rama con mayor pvalue")# con menos ejecuciones
    while best!=None:
        val=best.lastInstanceIndex#randint(0,9)# best.parent.pvalue
        #print "val:" + str(val)
        #input()
        #if val == -1:
        #    val=0
        if val <= max_pvalue:# and best.parent.pvalue!=0:
            max_pvalue = val
            best2= best#.parent
        best=best.parent

    ##print("Pvalue maximo encontrado en la rama:"+str(max_pvalue))

    #if(aux.p1>aux.p2 and aux.left==None) or (aux.p1<=aux.p2 and aux.right==None):
    if aux.left==None or aux.right==None:
        ##print("Solicitando nuevo nodo al arbol")
        nod=retrieveNode(aux)
        if nod:
            ##print("nodo recibido")
            ##print nod.alg1.name + " " +nod.alg2.name
            if aux.p1>aux.p2:
                aux.addLeft(nod)
                ##print("id nodo agregado: ",aux.left.id)
                return nod
            else:
                aux.addRight(nod)
                ##print("id nodo agregado y seleccionado: ",aux.right.id)
                return nod
    ##print("-------------------------------------------------------------")

    if best2.pvalue==0:
        ##print("Se corrieron todas las instancias del nodo seleccionado")
        return None
    else:
        ##print("")
        selected_vs_run.append(best2.id)
        ##print("nodo seleccionado : ",best2.id)
        return best2

def runNode(n):
    ##print "______________________"
    ##print("corriendo algoritmos en nodo"+str(n.id))
    time.sleep(0.01)
    if n.visited:
        id1=n.alg1.id
        id2=n.alg2.id
        i = selectInstance(n)
        ##print("selected Instance: "+str(i))
        resultado_a1= n.executeAlgorithm1(i)#resultados_experimentos[i][id1]
        resultado_a2= n.executeAlgorithm2(i)#resultados_experimentos[i][id2]
        global_results[i][id1]=resultado_a1
        global_results[i][id2]=resultado_a2
        ##print("Resultado algoritmo "+str(id1)+" :"+str(resultado_a1))
        ##print("Resultado algoritmo "+str(id2)+" :"+str(resultado_a2))
        diferencia=resultado_a1-resultado_a2
        ##print("Diferencia:"+str(diferencia))

        #n.save_jp()
        #maxpc_vs_run[n.id][n.lastInstanceIndex]=n.jointProbability()
    else:
        for j in range(1,4):
            id1=n.alg1.id
            id2=n.alg2.id
            i = selectInstance(n)
            #print("selected Instance: "+str(i))
            resultado_a1=n.executeAlgorithm1(i)#resultados_experimentos[i][id1]
            resultado_a2=n.executeAlgorithm2(i)#resultados_experimentos[i][id2]
            global_results[i][id1]=resultado_a1
            global_results[i][id2]=resultado_a2
            #print("Resultado algoritmo "+str(id1)+" :"+str(resultado_a1))
            #print("Resultado algoritmo "+str(id2)+" :"+str(resultado_a2))
            diferencia=resultado_a1-resultado_a2
            #print("Diferencia:"+str(diferencia))
        n.visited=True
        #n.save_jp()

    if n.lastInstanceIndex == 1599:#cambiar!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        n.setMaxp()
        n.setPvalueZero()
    #print "______________________"

            #maxpc_vs_run[n.id][n.lastInstanceIndex]=n.jointProbability()

def addChildrens(s,n):
    if n.visited==False:
        #print("agregando hijos")
        if n.left!=None:
            s.add(n.left)
        if n.right!=None:
            s.add(n.right)
    n.visited=True

def selectInstance(n): #mejorar para el caso de que se haya ejecutado uno y el otro no
    #print("seleccionando instancia")
    n.lastInstanceIndex=n.lastInstanceIndex + 1
    index=n.lastInstanceIndex
    i=instance_order[index]
    return i

def inicializarResultadosGlobales():
    #print("inicializarResultadosGlobales")
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
    ##print(globalr)

def update(s):
    global root
    #print("recalculando probabilidades del arbol")
    dimensiones=np.shape(resultados_experimentos)
    filas=dimensiones[0]
    #columnas=dimensiones[1]
    for n in s:
        if n.p1==1 or n.p2==1:
            continue
        recalculateMetric(n, filas)
        #n.data = n.alg1.name+"("+str(n.p1)+") vs "+n.alg2.name+"("+str(n.p2)+")"
        #n.left.pj=n.p1*
    clearScreen()
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
    ##print ("resultados alg"+str(id1)+" : ")
    ##print data1
    ##print ("resultados alg"+str(id2)+" : ")
    ##print data2

    mean2=np.mean(data2)
    mean1=np.mean(data1)
    variance1=np.var(data1)
    variance2=np.var(data2)
    media=mean1-mean2
    desviacion_tipica = np.sqrt((variance1/len(data1))+(variance2/len(data2)))
    ##print "media1: "+str(mean1)
    ##print "media2: "+str(mean2)
    ##print "variance1: "+str(variance1)
    ##print "variance2: "+str(variance2)
    ##print("numero de instancias:"+str(len(data1)))
    ##print("Media: "+str(media))
    ##print("Desviacion estandar:"+str(desviacion_tipica))
    #if len(data1)<30:
    estadistico_t=(0-media)/desviacion_tipica
    ##print "estadistico t:" + str(estadistico_t)
    grados_de_libertad=len(data1)-2
    #print "grados_de_libertad: " + str(grados_de_libertad)
    p=1-t.cdf(estadistico_t,grados_de_libertad)
    #print "probabilidad a>b:" + str(p)
    #else:
    #p=1-norm.cdf(0, media, desviacion_tipica)
    ##print("calculated probability"+str(p))
    t_test=stats.ttest_ind(data1,data2)
    n.pvalue=t_test[1]
    n.p1=p
    n.p2=1-p
    #n.actualize_#print()
    n.save_p()
    n.save_jp()
    n.save_pvalue()
    #Plotter.save_p(n)
    #Plotter.save_pvalue(n)#n.save_pvalue()
    #Plotter.save_jp(n)

def clearScreen():
    for i in range(1,25):
        print "."

def run():
    ########### Leer Archivo con Resultados ################

    namedata = 'data.txt'
    resultados_experimentos=readData(namedata)

    ########### Inicializar Resultados Globales en -1 ################

    global_results=inicializarResultadosGlobales()


    ########### Seleccion de metrica #######################


    ########### Generar orden de instancias ################

    np.random.seed(13)
    instance_order=np.random.permutation(len(global_results))


    #print(instance_order)

    ########### Creacion de algoritmos a comparar ################

    ####################################################################
    medias_0 = []
    medias_1 = []
    medias_2 = []
    probability_0 = []
    probability_1 = []
    probability_2 = []
    ########### Ejecucion ################
    #Plotter= Plotter()
    speculativeExecute()

    ########### Vectores para hacer plots ####################
    #maxpc_vs_iter=[]

    if root.right:
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
    plt.xlabel('iteration')
    plt.ylabel('jointProbability')
    plt.plot(root.get_jp_vs_run2())
    plt.plot(root.left.get_jp_vs_run())
    plt.plot(root.left.get_jp_vs_run2())
    plt.title("joint Probability")

    plt.figure()
    plt.plot(root.get_pvalue_vs_run())
    plt.title("pvalue vs run root")
    plt.show()

