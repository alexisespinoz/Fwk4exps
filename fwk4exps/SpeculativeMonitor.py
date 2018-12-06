import matplotlib.pyplot as plt #graficos
import matplotlib.animation as animation #graficos animados

#from p#print import p#print
from fwk4exps.classes import noprint
from fwk4exps.classes.detectInput import KeyPoller
#from classes.Plotter import Plotter
#from classes.Algorithm import Algorithm
from fwk4exps.classes.noprint import * 
from fwk4exps.classes.Tree import Tree
from fwk4exps.classes.Strategy import Strategy
#from classes.Strategy import Metric
from scipy import stats
from scipy.stats import norm,t





import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

import pymc3 as pm


import copy
import time
from random import randint
##para evitar que imprima los print
from contextlib import contextmanager
import sys, os
import multiprocessing



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
__totalSimulations = 1000
iteration = 0

ax1 = None
ann_list = []

__numOfExecutions=0


s2id = {}
s_id =0

parametersAlgos = dict()

class SpeculativeMonitor(object):
    """docstring for SpeculativeMonitor"""
    def __init__(self, expDesign,PI):
        global experimentalDesign
        experimentalDesign = expDesign
        global pifile
        pifile = PI

def bestStrategy(S1, S2, __range, delta_sig):
    global __count, __speculativeNode, __msg
    if __count < len(__msg):
        if __msg[__count]==0:
            __count=__count+1
            return S1
        else:
            __count=__count+1
            return S2
    ins_ord = instanceOrderGenerator(__range)
    __speculativeNode=Tree(S1,S2,None,0,ins_ord,__range[0])
    raise ValueError

def retrieveNode(aux):
    print("retrieve node")
    global __count, __speculativeNode, __msg
    __msg=[]
    if aux:
        __msg=aux.getMsg()
    try:
        __count=0
        __speculativeNode = None
        print ("salida mas probable:")
        experimentalDesign()
    except ValueError as x:
        print ("")
    else:
        marcarNodo()
    finally:
        return __speculativeNode

def speculativeExecute():
    print("---------------------speculativeExecute----------------------")
    global root, iteration
    s = set()
    root = retrieveNode(None)
    if root:
        s.add(root)
    runNode(root)
    update(s)
    root.printTree()

    #with KeyPoller() as keyPoller:
    while (len(s)>0):
        a = KeyPoller()
        c = a.poll() 
        pressed=False
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
            update(s)
            iteration=iteration+1
            print("iteracion: "+str(iteration))
            saveConditionalProbability(root,s)
            root.printTree()
        else:
            #with suppress_stdout():
            n = select(s)
            if n==None:
                print("se cumple criterio de salida")
                break
            s.add(n)
            runNode(n)
            update(s)
            iteration=iteration+1
            print("iteracion: "+str(iteration))
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
        #print (dimensiones)   

def runNode(n):
    global instancias, pifile
    print ("______________________")
    print("corriendo algoritmos en nodo"+str(n.id))

    if n.visited:
        id1 = mapa(n.alg1)#.id
        id2 = mapa(n.alg2)#.id
        
        reshape(id1,id2)
        runParallel(n)

    else:
        for j in range(1,4):
            id1 = mapa(n.alg1)#.id
            id2 = mapa(n.alg2)#.id
            
            reshape(id1,id2)
            runParallel(n)

        n.visited=True


    if n.isTerminated():
        n.setMaxp()
        n.setPvalueZero()


def update(s):
    global root,__totalSimulations#,parametersAlgos
    root.refreshSimulations()

    for x in range(1,__totalSimulations + 1):
        #print ("simulation "+str(x)+":")
        simulation(root)

def simulation(nod):
    global parametersAlgos
    n = nod
    sumDict = {}
    while n is not None:   
        n.addSimulationVisit()
        total = len(n.ins_ord)
     
        parametersAlgo1 = parametersAlgos[hash(n.alg1)]
        parametersAlgo2 = parametersAlgos[hash(n.alg2)]
        
        mean1, sd1, data1= getRandomParameters(parametersAlgo1)
        mean2,sd2, data2 = getRandomParameters(parametersAlgo2)   
        current1 = len(data1)
        current2 = len(data2)
        c1 = total - current1
        c2 = total - current2
        delta = 0#.01


        

        n.savemean1(np.mean(data1))
        parcial_sum1 = sum(data1)
        complementsum1 = np.random.normal( c1*mean1, np.sqrt(c1)*sd1)

        
        n.savemean2(np.mean(data2))
        parcial_sum2 = sum(data2)
        complementsum2 = np.random.normal(c2 * mean2, np.sqrt(c2)* sd2)

        ############
        #print info#
        ############
        #print ("visitando nodo:")
        n.getData()

        #print ("media estimada 1:" + str((parcial_sum1 + complementsum1 )/total*1.0))

        #print ("media estimada 2:" + str((parcial_sum2 + complementsum2 )/total*1.0))

        if ( (parcial_sum1 + complementsum1 )/total*1.0 + delta > (parcial_sum2 +complementsum2)/total*1.0 ):
            n.p1 = n.p1+1#(parcial_sum1 + complementsum1 + delta)/total
            #n.p2 = #(parcial_sum2 +complementsum2 )/total
            n = n.left
        else:
            n.p2 = n.p2+1#(parcial_sum1 + complementsum1 + delta)/total
            #n.p1 = #(parcial_sum2 +complementsum2)/total 
            n = n.right    

def readData(path):

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
    global iteration, __totalSimulations, __numOfExecutions

    max_sim = -1

    #cambiar esta parte!!!
    for n in s:
        if n.isLeafLeaf:
            #print ("hay hoja hoja !!!! ")
            if n.p1 > max_sim :
                max_sim = n.p1
                bestId = mapa(n.alg1)
            if n.p2 > max_sim :
                max_sim = n.p2
                bestId = mapa(n.alg2)
        '''
        if n.p1 > max_sim and n.isLeafLeaf:
            max_sim = n.p1
        else:
            if n.p2 >max_sim and n.isLeafLeaf:
                max_sim = n.p2
        '''
    if max_sim > -1:
        print ("guardando probabilidad conjunta en archivo")
        #print maxCP
        maxCP = max_sim / (__totalSimulations*1.0)

        ##########################################
        # guadar calidad en archivo de animacion #
        ##########################################
        with open('fwk4exps/dataanimation.txt','a') as f:
            f.write(str( __numOfExecutions )+","+str(maxCP)+","+str(bestId)+"\n")

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
    clearAnimationData()
    live_plot = multiprocessing.Process(target=livegraph, args=())
    live_plot.start()
    speculativeExecute()
    live_plot.join()
    #plotQuality()
    ########### Vectores para hacer plots ####################
    #maxpc_vs_iter=[]

def animate(i):
    global ax1,ann_list

    for i, a in enumerate(ann_list):
        a.remove()
    ann_list[:] = []

    graph_data = open('fwk4exps/dataanimation.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    zs = []
    for line in lines:
        if len(line) > 1:
            x, y, z = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
            zs.append(z)

    ax1.clear()
    ax1.plot(xs, ys)
    for i, txt in enumerate(zs):
        #print(i,txt)
        ann=ax1.annotate(txt, (xs[i], ys[i]))
        ann_list.append(ann)

def livegraph():
    global ax1
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)     
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

def clearAnimationData():
    open('fwk4exps/dataanimation.txt','w').close()

def marcarNodo():
    #print "marcando nodo como hoja hoja"
    global __msg,root
    #print (__msg)
    aux = root
    #cambiar, llegar hasta el penultimo o si no se vuelve nulo
    ##
    __msg.pop()

    for i in __msg:
        #print i
        if i == 0:
            #print "left"
            aux = aux.left
            continue
        if i == 1:
            #print "right"
            aux = aux.right
            continue
    #print "done"

    #print aux
    aux.isLeafLeaf = True
    #print "es hoja hoja?"
    #print aux.isLeafLeaf
def sampleParameters(data):
    #https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
    np.random.seed(123)


    #extracted means and sigmas from 
    __medias = []
    __sigmas = [] 
    #with suppress_stdout:
    with pm.Model():
        mu = pm.Normal('mu', np.mean(data), 1)
        sigma = pm.Uniform('sigma', lower=0.001, upper=1)
        
        returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)
        
        step = pm.Metropolis()
        trace = pm.sample(5000, step, cores=4)
        
        for t in trace:
          __medias.append(t["mu"])
          __sigmas.append(t["sigma"])

    return __medias, __sigmas, data

def getAlgResults(alg):
    global global_results
    id1 = mapa(alg)#.id
    filas , columnas = np.shape(global_results)
    data1=[]
    for i in range(0,filas):
        if global_results[i][id1] != -1 :
            data1.append(global_results[i][id1])
    return data1

def getRandomParameters(params):
    medias = params[0]
    sigmas = params[1]
    algoResults = params[2]
    largo  = len(medias)
    randomIndex =np.random.randint(largo)
    return medias[randomIndex], sigmas[randomIndex], algoResults
    
def runParallel(node):
    global instancias, pifile, parametersAlgos,__numOfExecutions

    id1 = mapa(node.alg1)#.id
    id2 = mapa(node.alg2)#.id 
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    #seleccionar instancia
    numproc = multiprocessing.cpu_count()
    diff = node.lastInstanceIndex1 - node.lastInstanceIndex2 
    print("numero de procedadores:"+str(numproc))
    print ("esta es la difererncia:"+str(diff))

    if diff>0:
        if diff>numproc:
            for j in range(numproc):
                i=node.selectInstance2()
                if i==None:
                    break
                instancia = instancias[i]

                while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                    i = node.selectInstance2()
                    instancia = instancias[i]

                p = multiprocessing.Process(target=node.executeAlgorithm2, args=(i,id2,instancia,pifile,return_dict))
                jobs.append(p)
                p.start()
        else:
            c = numproc - diff
            for j in range(c//2):
                i=node.selectInstance1()
                if i==None:
                    break
                instancia = instancias[i]
                
                while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                    i = node.selectInstance1()
                    instancia = instancias[i]

                p = multiprocessing.Process(target=node.executeAlgorithm1, args=(i,id1,instancia,pifile,return_dict))
                jobs.append(p)
                p.start()

            for j in range(c//2+diff):
                i=node.selectInstance2()
                if i==None:
                    break
                instancia = instancias[i]

                while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                    i = node.selectInstance2()
                    instancia = instancias[i]

                p = multiprocessing.Process(target=node.executeAlgorithm2, args=(i,id2,instancia,pifile,return_dict))
                jobs.append(p)
                p.start()

    elif(diff < 0):
        diff = abs(diff)
        if diff>numproc:
            for j in range(numproc):
                i=node.selectInstance1()
                if i==None:
                    break
                instancia = instancias[i]

                while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                    i = node.selectInstance1()
                    instancia = instancias[i]

                p = multiprocessing.Process(target=node.executeAlgorithm1, args=(i,id2,instancia,pifile,return_dict))
                jobs.append(p)
                p.start()
        else:
            c = numproc - diff
            for j in range(c//2+diff):
                i=node.selectInstance1()
                if i==None:
                    break
                instancia = instancias[i]
                
                while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                    i = node.selectInstance1()
                    instancia = instancias[i]

                p = multiprocessing.Process(target=node.executeAlgorithm1, args=(i,id1,instancia,pifile,return_dict))
                jobs.append(p)
                p.start()

            for j in range(c//2):
                i=node.selectInstance2()
                if i==None:
                    break
                instancia = instancias[i]

                while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                    i = node.selectInstance2()
                    instancia = instancias[i]

                p = multiprocessing.Process(target=node.executeAlgorithm2, args=(i,id2,instancia,pifile,return_dict))
                jobs.append(p)
                p.start()
    else:
        print("hola entre")
        for j in range(numproc//2):
            i=node.selectInstance1()
            if i==None:
                break
            instancia = instancias[i]
            
            while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                i = node.selectInstance1()
                instancia = instancias[i]

            p = multiprocessing.Process(target=node.executeAlgorithm1, args=(i,id1,instancia,pifile,return_dict))
            jobs.append(p)
            p.start()

        for j in range(numproc//2):
            i=node.selectInstance2()
            if i==None:
                break
            instancia = instancias[i]

            while global_results[i][id1]!=-1 and global_results[i][id2] != -1: 
                i = node.selectInstance2()
                instancia = instancias[i]

            p = multiprocessing.Process(target=node.executeAlgorithm2, args=(i,id2,instancia,pifile,return_dict))
            jobs.append(p)
            p.start()
        
    for proc in jobs:
        proc.join()
        __numOfExecutions = __numOfExecutions + 1

    #cambiar por guardar (LLave doble)

    #returned=return_dict.items()

    #print(returned)
    keys = [key for key,value in return_dict.items()]
    #print(keys)
    for k in keys:
        global_results[k[0]][k[1]] = return_dict[k]
    #print (return_dict.values())



    #parametersAlgos.clear()
    if hash(node.alg1) in parametersAlgos.keys():
        parametersAlgos.pop(hash(node.alg1))
    if hash(node.alg2) in parametersAlgos.keys():
        parametersAlgos.pop(hash(node.alg2))

    with suppress_stdout():
        parametersAlgos[hash(node.alg1)] = sampleParameters(getAlgResults(node.alg1))
        parametersAlgos[hash(node.alg2)] = sampleParameters(getAlgResults(node.alg2))

    '''
    for n in s:
        if hash(n.alg1) not in parametersAlgos:

            parametersAlgos[hash(n.alg1)] = sampleParameters(getAlgResults(n.alg1))

        if hash(n.alg2) not in parametersAlgos:

            parametersAlgos[hash(n.alg2)] = sampleParameters(getAlgResults(n.alg2))
    '''