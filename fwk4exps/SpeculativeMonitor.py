import matplotlib.pyplot as plt #graficos
#from p#print import p#print
#from classes import noprint
from classes.detectInput import KeyPoller
#from classes.Plotter import Plotter
#from classes.Algorithm import Algorithm
from classes.Tree import Tree
from classes.Strategy import Strategy
#from classes.Strategy import Metric
from scipy import stats
from scipy.stats import norm,t
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


pi = None
resultados_experimentos=None
#selected_vs_run = []
__count = None
__msg = None
__speculativeNode = None
root = None
pifile=None
experimentalDesign = None
instance_order = None
instancias = None
global_results =None
quality_vs_iteration = []
__totalSimulations = 2000

s2id = {}
s_id =0

class SpeculativeMonitor(object):
    """docstring for SpeculativeMonitor"""
    def __init__(self, expDesign,PI):
        global experimentalDesign
        experimentalDesign = expDesign
        global pifile
        pifile = PI


def bestStrategy(S1, S2, range, delta_sig):
    ##print("______________________")
    #print("entra en bestStrategy")
    #if self.pi is None:
    #   self.pi = pi 

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
    ins_ord = instanceOrderGenerator(range)
    __speculativeNode=Tree(S1,S2,None,0,ins_ord)
    raise ValueError

def retrieveNode(aux):
    ##print("______________________")
    print("retrieve node")
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

def speculativeExecute():
    ##print("-------------------------------------------------------------")
    print("---------------------speculativeExecute----------------------")
    ##print("-------------------------------------------------------------")
    global root
    #v = null
    s = set()
    root = retrieveNode(None)
    if root:
        print("raiz agregada :)")
        #print root.alg1.pathExe
        s.add(root)
    runNode(root)
    update()
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
                n = select(s)
                if n==None:
                    print("se cumple criterio de salida")
                    break
                s.add(n)
                runNode(n)
                update()
                i=i+1
                print("iteracion: "+str(i))
                saveConditionalProbability(root,s)
                #root.printTree()
            else:
                #with suppress_stdout():
                n = select(s)
                if n==None:
                    print("se cumple criterio de salida")
                    break
                s.add(n)
                runNode(n)
                update()
                i=i+1
                print("iteracion: "+str(i))
                saveConditionalProbability(root,s)
                root.printTree()

        ##print("-------------------------------------------------------------")

def select(s):
    print("_____________________________________________")
    print("selecting best node to run")
    print("________________________________________________________")
    ####Parte1###
    global root
    aux = root
    best = None
    pj = 2
    while aux is not None:#not aux.isLeaf(): 

        if aux.conditionalProb() < pj:
            best = aux
            pj = aux.conditionalProb()

        if aux.p1 > aux.p2:
            if aux.left is not None:
                aux = aux.left
            else:
                break
        else:
            if aux.right is not None:
                aux = aux.right
            else:
                break
        '''
        #si hay hijo izquierdo
        if aux.left:
            #si hay ambos hijos
            if aux.right:
                if aux.left.simulationVisitCount > aux.right.simulationVisitCount: #conditionalProb()>aux.right.conditionalProb():
                    aux = aux.left
                else:
                    aux = aux.right
                if aux.conditionalProb() < pj:
                    best = aux
                    pj = aux.conditionalProb()                    
            #si hay solo hijo izquierdo
            else:
                if aux.left.simulationVisitCount > aux.simulationVisitCount - aux.left.simulationVisitCount:
                    aux = aux.left
                    if aux.conditionalProb() < pj:
                        best = aux
                        pj = aux.conditionalProb()
                else:
                    break
                    #aux = aux.right

        else:
        #si hay solo hijo derecho
            if aux.right:
                if aux.right.simulationVisitCount >aux.simulationVisitCount - aux.left.simulationVisitCount:
                    aux = aux.right
                else:
                    break
                    #aux = aux.left
                #    if aux.conditionalProb() < pj:
                #        best = aux
                #        pj = aux.conditionalProb()
        #si no hay ninguno
        #else:
        #    break
    '''

    '''         
            if aux.p1 > aux.p2:
                if aux.left is None:
                    break
                else:
                    aux = aux.left
                    if aux.conditionalProb() < pj:
                        best = aux
                        pj = aux.conditionalProb()
            else:
                if aux.right is None:
                    break
                else:
                    aux = aux.right
                    if aux.conditionalProb() < pj:
                        best = aux
                        pj = aux.conditionalProb()
    '''  
    ###############
    #   expansion #
    ###############
    nod = retrieveNode(aux)

    if nod:

        if aux.left == None and aux.p1 > aux.p2:
            aux.addLeft(nod)
            return nod

        if aux.right == None and aux.p2 > aux.p1:
            aux.addRight(nod)
            return nod
    ###################
    #   end expansion #
    ###################   
    if best.isTerminated():
        print("Se corrieron todas las instancias del nodo seleccionado")
        return None
    else:

        print("nodo seleccionado : ",best.id)
        return best   

def mapa(alg):
    global s2id, s_id

    exists = False

    for key in s2id:
        if key == alg:
            exists = True

    if exists:
        return s2id[alg]
    else:
        s2id[alg] = s_id
        s_id = s_id + 1
        return s2id[alg]

def reshape(id1,id2):
    global global_results
    
    dimensiones=np.shape(global_results)

    num_alg = dimensiones[1] #numero de algoritmos
    num_ins = dimensiones[0] #numero instancias
    if (id1 >= num_alg or id2 >= num_alg):
        if(id1>id2):
            while num_alg<=id1 :
                new_column = np.ones((num_ins,1))*-1
                global_results = np.hstack((global_results,new_column))
                num_alg =num_alg+1
        else:
            while num_alg<=id2 :
                new_column = np.ones((num_ins,1))*-1
                global_results = np.hstack((global_results,new_column))
                num_alg =num_alg+1
        print dimensiones   

def runNode(n):
    global instancias, pifile
    print "______________________"
    print("corriendo algoritmos en nodo"+str(n.id))
    ###_Agregar columnas si el numero de algoritmos e mayor que el numero de columnas actual de la matriz de resultados globales
    
    #time.sleep(0.01)
    if n.visited:
        id1 = mapa(n.alg1)#.id
        id2 = mapa(n.alg2)#.id
        
        reshape(id1,id2)
        i = n.selectInstance()
        instancia = instancias[i]
        ##print("selected Instance: "+str(i))
        if global_results[i][id1] == -1:
            resultado_a1 = n.executeAlgorithm1(instancia,pifile)#resultados_experimentos[i][id1]
            global_results[i][id1] = resultado_a1

        if global_results[i][id2] == -1:
            resultado_a2 = n.executeAlgorithm2(instancia,pifile)#resultados_experimentos[i][id2]
            global_results[i][id2] = resultado_a2

        ##print("Resultado algoritmo "+str(id1)+" :"+str(resultado_a1))
        ##print("Resultado algoritmo "+str(id2)+" :"+str(resultado_a2))
        #diferencia=resultado_a1-resultado_a2
        ##print("Diferencia:"+str(diferencia))

        #n.save_jp()
        #maxpc_vs_run[n.id][n.lastInstanceIndex]=n.jointProbability()
    else:
        for j in range(1,4):
            id1 = mapa(n.alg1)#.id
            id2 = mapa(n.alg2)#.id
            
            reshape(id1,id2)
            #i = selectInstance(n)
            #print("selected Instance: "+str(i))
            i = n.selectInstance()
            instancia = instancias[i]
            if global_results[i][id1] == -1:
                resultado_a1 = n.executeAlgorithm1(instancia,pifile)#resultados_experimentos[i][id1]
                global_results[i][id1] = resultado_a1

            if global_results[i][id2] == -1:
                resultado_a2 = n.executeAlgorithm2(instancia,pifile)#resultados_experimentos[i][id2]
                global_results[i][id2] = resultado_a2
            #print("Resultado algoritmo "+str(id1)+" :"+str(resultado_a1))
            #print("Resultado algoritmo "+str(id2)+" :"+str(resultado_a2))
            #diferencia=resultado_a1-resultado_a2
            #print("Diferencia:"+str(diferencia))
        n.visited=True
        #n.save_jp()

    if n.isTerminated():#lastInstanceIndex == 1599:#cambiar!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        n.setMaxp()
        n.setPvalueZero()
    #print "______________________"

            #maxpc_vs_run[n.id][n.lastInstanceIndex]=n.jointProbability()__

def update():
    global root,__totalSimulations
    root.refreshSimulations()
    for x in xrange(1,__totalSimulations + 1):
        simulation(root)

def simulation(nod):
    global global_results
    n = nod
    sumDict = {}
    while n is not None:   
        n.addSimulationVisit()
        total = len(n.ins_ord)
        current = n.lastInstanceIndex
        c = total - current
        delta = 0#.01
        #print c
        #obtener resultados parciales
        id1 = mapa(n.alg1)#.id
        id2 = mapa(n.alg2)#.id
        data1=[]
        data2=[]
        filas , columnas = np.shape(global_results)
        for i in range(0,filas):
            if(global_results[i][id1] != -1 and global_results[i][id2] != -1):
                data1.append(global_results[i][id1])
                data2.append(global_results[i][id2]) 

        #print data1
        mean1 = np.mean(data1)
        n.savemean1(mean1)
        #print mean1
        sd1 = np.std(data1)
        #print sd1
        parcial_sum1 = sum(data1)
        complementsum1 = np.random.normal( c*mean1, np.sqrt(c)*sd1)

        mean2 = np.mean(data2)
        n.savemean2(mean2)
        sd2 = np.std(data2)
        parcial_sum2 = sum(data2)
        complementsum2 = np.random.normal(c * mean2, np.sqrt(c)* sd2)
        
        if ( (parcial_sum1 + complementsum1 )/total*1.0 + delta> (parcial_sum2 +complementsum2)/total*1.0 ):
            n.p1 = n.p1+1#(parcial_sum1 + complementsum1 + delta)/total
            #n.p2 = #(parcial_sum2 +complementsum2 )/total
            n = n.left
        else:
            n.p2 = n.p2+1#(parcial_sum1 + complementsum1 + delta)/total
            #n.p1 = #(parcial_sum2 +complementsum2)/total 
            n = n.right    
    '''
    delta = 0
    id1 = mapa(n.alg1)#.id
    id2 = mapa(n.alg2)#.id
    data1=[]
    data2=[]
    for i in range(0,filas):
        if(global_results[i][id1] != -1 and global_results[i][id2] != -1):
            data1.append(global_results[i][id1])
            data2.append(global_results[i][id2])
    dif = np.array(data1)-np.array(data2)
    n_0 = len(dif)
    N = len(n.ins_ord)
    varianza = np.var(dif)
    sd = np.sqrt(varianza/n_0)
    mean_dif = np.mean(dif)
    val = N*(delta - (n_0*mean_dif)/N)
    factor = N - n_0
    estadistico_t = (val - mean_dif)/(sd * factor)
    grados_de_libertad = n_0-1
    p = 1 - t.cdf(estadistico_t,grados_de_libertad)
    #n.pvalue=t_test[1]
    n.p1=p
    n.p2=1-p
    #n.actualize_#print()
    n.save_p()
    n.save_jp()
    '''

def readData(path):
    #f=[]
    #f=np.genfromtxt(results)
    #return f
    '''
    aux = copy.copy(path)
    aux = aux.split("/")
    aux.pop()
    aux.pop(0)
    absolutePath = ""
    for e in aux:
        absolutePath = absolutePath + "/" + e
    absolutePath = absolutePath + "/"
    print path
    print absolutePath
    '''
    with open(path) as f:
        content = f.readlines()

    return content

def instanceOrderGenerator(range):
    np.random.seed(13)
    instance_order=np.random.permutation(len(range))+range[0]
    return instance_order

def createGlobalResults():
    global global_results
    global instancias
    length = len(instancias)
    global_results = np.ones(shape =(length,2))*-1

def saveConditionalProbability(root,s):
    global quality_vs_iteration, __totalSimulations
    maxCP = 0
    #leafs = 
    ###############################
    # encontrar algoritmo ganador #
    ###############################
    aux = root
    bestAlgId = None
    sumSimulations = 0
    while aux != None:
        if aux.p1 > aux.p2:
            if aux.left is not None:
                aux = aux.left
            else:
                break
        else:
            if aux.right is not None:
                aux = aux.right
            else:
                break

    if aux.p1 > aux.p2:
        bestAlgId = aux.alg1.name
    else:
        bestAlgId = aux.alg2.name

    print "mejor algoritmo: " + bestAlgId
    ###################################
    # fin encontrar algoritmo ganador #
    ###################################

    ###################################
    # recorrer nodos hojas            #
    ###################################
    for n in s:
        if n.isLeaf():
            if n.alg1.name == bestAlgId: #and n.p1 >= n.p2:
                print "entraaaaa_A"
                sumSimulations = sumSimulations + n.p1

            if n.alg2.name == bestAlgId: #and n.p2 >= n.p1:
                print "entraaaaa_B"
                sumSimulations = sumSimulations + n.p2    
    ###################################
    # fin recorrer nodos hojas        #
    ###################################
    print "sumSimulations:"
    print sumSimulations 

    maxCP = sumSimulations / (__totalSimulations*1.0)

    print "maxCP:"
    print maxCP
    quality_vs_iteration.append(maxCP) 

def run():

    ########### Generar orden de instancias ################
    global instancias


    instancias=readData(pifile)
    #print pifile
    #print pi[-1]
    createGlobalResults()
    #instance_order=np.random.permutation(len(pi))


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
    plotQuality()
    ########### Vectores para hacer plots ####################
    #maxpc_vs_iter=[]
    '''
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
    '''

def plotQuality():
    global quality_vs_iteration

    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('max conditional Probability')
    plt.plot(quality_vs_iteration)
    plt.title("quality of results over iterations")
    plt.show()