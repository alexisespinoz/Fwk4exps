from classes.Algorithm import Algorithm
from classes.Tree import Tree
from scipy import stats
from scipy.stats import norm
import numpy as np

def readData(results):
    f=[]
    f=np.genfromtxt(results)
    return f

def speculativeExecute():
    #v = null
    s = set()
    s.add(root)
    i=0
    while (i<100):
        print("-------------------------------------------------------------")
        print("iteracion: "+str(i))
        n = best(s)
        run(n)
        update(s)
        i=i+1
        print("-------------------------------------------------------------")

def best(s):
    ####Parte1###
    # Selecciona el nodo no visitado con max probabilidad conjunta

    ####Parte2###
    #bestp=-1
    #bestnode=root
    #leafs=root.leafs()
    return root

def run(n):
    n.visited=True
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

def selectInstance(n): #mejorar para el caso de que se haya ejecutado uno y el otro no
    n.lastInstanceIndex=n.lastInstanceIndex + 1
    index=n.lastInstanceIndex
    i=instance_order[index]
    return i

def inicializarResultadosGlobales():
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
    dimensiones=np.shape(resultados_experimentos)
    filas=dimensiones[0]
    columnas=dimensiones[1]
    for n in s:
        id1=n.alg1.id
        id2=n.alg2.id
        data1=[]
        data2=[]
        #difference = []
        for i in range(0,filas):
            if(global_results[i][id1] != -1 and global_results[i][id2] != -1):
                data1.append(global_results[i][id1])
                data2.append(global_results[i][id2])
                #difference.append(global_results[i][id1]-global_results[i][id2])
        mean1=np.mean(data1)
        mean2=np.mean(data2)
        variance1=np.var(data1)
        variance2=np.var(data2)
        media=mean1-mean2
        desviacion_tipica=np.sqrt((variance1/len(data1))+(variance2/len(data2)))
        print("numero de instancias:"+str(len(data1)))
        print("Media: "+str(media))
        print("Desviacion estandar:"+str(desviacion_tipica))
        p=1-norm.cdf(0,media,desviacion_tipica)
        print("calculated probability"+str(p))
        n.p1=p
        n.p2=1-p
        n.data = n.alg1.name+"("+str(n.p1)+") vs "+n.alg2.name+"("+str(n.p2)+")"
        #n.left.pj=n.p1*
        n.printTree()

def insert(s,v):
    s.append(v[0])
    s.append(v[1])

        #t_test=stats.ttest_ind(data1,data2)
        #print(t_test)


########### Leer Archivo con Resultados ################
namedata = 'data.txt'
resultados_experimentos=readData(namedata)

########### Inicializar Resultados Globales en -1 ################

global_results=inicializarResultadosGlobales()

########### Generar orden de instancias ################

instance_order=np.random.permutation(len(global_results))
print(instance_order)

########### Creacion de algoritmos a comparar ################

alg1=Algorithm("algoritmo1",0)
alg2=Algorithm("algoritmo2",1)
alg3=Algorithm("algoritmo3",2)


########### Creacion de arbol de procesos ################

root=Tree(alg1,alg2,None)
root.left=Tree(alg2,alg3,root)
root.right=Tree(alg3,alg2,root)

#algoritmos=[alg1, alg2, alg3]

########### Ejecucion ################
speculativeExecute()
