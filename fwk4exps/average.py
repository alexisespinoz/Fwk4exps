from scipy.stats import norm,t
import numpy as np
#import speculativemonitor


def readData(results):
    f=[]
    f=np.genfromtxt(results)
    return f

namedata = 'data.txt'
resultados_experimentos=readData(namedata)

delta = 0

N= len(resultados_experimentos)
vec=resultados_experimentos[0:75,1] - resultados_experimentos[0:75,2]

n_0= len(vec)

print vec

variance=np.var(vec)
print variance

desviacion_tipica = np.sqrt(variance/len(vec))
print "sd"+str(desviacion_tipica)
diferencia_de_medias = np.mean(vec)
#print diferencia_de_medias
valor = N*(delta-(n_0*np.mean(vec))/N)
print valor
factor = N-n_0
estadistico_t = (valor-diferencia_de_medias*factor)/(desviacion_tipica*factor)  # media de las diferencias

grados_de_libertad = n_0 -1
p = 1 - t.cdf(estadistico_t,grados_de_libertad)

print p
