import matplotlib.pyplot as plt #graficos
from pprint import pprint
from classes.Algorithm import Algorithm
from classes.Tree import Tree
from scipy import stats
from scipy.stats import norm,t
from prettytable import PrettyTable
import numpy as np

selected_vs_run = []

def readData(results):
    f=[]
    f=np.genfromtxt(results)
    return f

def speculativeExecute():
    #v = null
    s = set()
    s.add(root)
    i=0
    while (len(s)>0):
        print("-------------------------------------------------------------")
        print("iteracion: "+str(i))
        n = best(s)
        if n==None:
            print("se cumple criterio de salida")
            break
        run(n)
        update(s)
        addChildrens(s,n)

        i=i+1
        input('Enter your input:')
        print("-------------------------------------------------------------")

def best(s):
    print("-------------------------------------------------------------")
    print("selecting best node to run")
    print("largo de s"+str(len(s)))
    ####Parte1###
    # Selecciona el nodo no visitado u hoja con max probabilidad conjunta
    best=None
    max_pc=-1
    for nod in s:
        if ((not nod.visited or nod.isLeaf())and nod.jointProbability()>max_pc ):
            best=nod
            max_pc=nod.jointProbability()
    # porqueeeeee este if?
    if best==None:
        return None
    #print("probabilidadconjunta:"+str(max_pc))
    #maxpc_vs_iter.append(max_pc)
    ####Parte2###
    min_p=best.bestp1p2()
    best2=Tree(None,None,None,None)
    best2=best
    while best.parent!=None:
        if best.parent.left == best:
            val = best.parent.p1
        if best.parent.right == best:
            val = best.parent.p2
        if val< min_p:
            min_p = val
            best2= best.parent
        best=best.parent
    print()
    print("-------------------------------------------------------------")
    if min_p == 1:
        return None
    else:
        selected_vs_run.append(best2.id)
        print("nodo seleccionado : ",best2.id)
        return best2

def run(n):
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

            #n.save_jp()

    if n.lastInstanceIndex == 1599:
        n.setMaxp()


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

def insert(s,v):
    s.append(v[0])
    s.append(v[1])


def recalculateMetric(n, filas):
    print(" recalculando metrica en nodo",n.getData())
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
    if len(data1) > 2 and len(data2) > 2 :
        print ("resultados alg"+str(id1)+" : ")
        print data1
        print ("resultados alg"+str(id2)+" : ")
        print data2

        mean2=np.mean(data2)
        mean1=np.mean(data1)
        variance1=np.var(data1)
        variance2=np.var(data2)
        media=mean1-mean2
        if n.id==0:
            medias_0.append(media)
        if n.id==1:
            medias_1.append(media)
        if n.id==2:
            medias_2.append(media)
        #medias_0.append(media)
        desviacion_tipica = np.sqrt((variance1/len(data1))+(variance2/len(data2)))
        print("numero de instancias:"+str(len(data1)))
        print("Media: "+str(media))
        print("Desviacion estandar:"+str(desviacion_tipica))
        if len(data1)<30:
            estadistico_t=(0-media)/desviacion_tipica
            grados_de_libertad=len(data1)-1
            p=1-t.cdf(estadistico_t,grados_de_libertad)
            if n.id==0:
                probability_0.append(p)
            if n.id==1:
                probability_1.append(p)
            if n.id==2:
                probability_2.append(p)
            #n.save_test(estadistico_t,p)
        else:
            p=1-norm.cdf(0, media, desviacion_tipica)

            if n.id==0:
                probability_0.append(p)
            if n.id==1:
                probability_1.append(p)
            if n.id==2:
                probability_2.append(p)
            #n.save_test(estadistico_t,p)
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

alg1=Algorithm("algoritmo0",0)
alg2=Algorithm("algoritmo1",1)
alg3=Algorithm("algoritmo2",2)


########### Creacion de arbol de procesos ################

root=Tree(alg1,alg2,None,0)
root.left=Tree(alg1,alg3,root,1)
root.right=Tree(alg2,alg3,root,2)

root.save_p
root.left.save_p
root.right.save_p
#algoritmos=[alg1, alg2, alg3]

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
maxpc_vs_run = []

#if(root.left.jointProbability()>root.right.jointProbability()):


#Pretty PrettyTable
t = PrettyTable(['Diferencia Media','Probabilidad'])
for i in range(0,(len(medias_0)-1)):
    t.add_row([medias_0[i],probability_0[i]])
print t

'''
print("num times right node was run: "+str(root.right.lastInstanceIndex))
print("num times left node was run: "+str(root.left.lastInstanceIndex))
print("num times root node was run: "+str(root.lastInstanceIndex))
print("lenght of jp calculated in right node:"+str(len(root.right.get_jp_vs_run())))
print("lenght of jp calculated in left node:"+str(len(root.left.get_jp_vs_run())))
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
plt.title("Diferencia Medias y probabilidad nodo DERE(A2 VS A3)")
'''
plt.show()


print(root.stat_vs_run)
#maxpc_vs_run=root.get_jp_vs_run()
#plt.plot(maxpc_vs_run)
#plt.title("max joint vs runs (right node)")
#plt.show()
