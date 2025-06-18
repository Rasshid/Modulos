# importar biliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import numpy.random as nrd
import math
# definir función 'getCentroids()'
def getCentroids(X_data, y_data):
  m=X_data.shape[1] #numero de caracteristicas
  y_unique=np.unique(y_data) #vector con valores de y_data sin repetir
  n=len(y_unique) # numero de etiquetas
  C=np.zeros([n,m]) #matriz C de centroides
  for i in range (n):
    posiciones = np.where(y_data == y_unique[i])  # buscamos las posiciones de cada etiqueta
    C[i]= np.mean(X_data[posiciones], 0) #obtenemos el promedio de cada etiqueta y lo guardamos en una fila
  return C, y_unique
# definir función 'minkowski()'
def disMinkowski(x,y,p):
  #obtener sumatoria
  suma=np.sum([np.abs([x-y])**p])
  #elevar a la potencia
  dis=suma**(1/p)
  return dis
# definir función 'getDistances()'
def getDistances(x0,X_data,y_data,p):
  #obtener tamaño de la matriz
  n=len(X_data)
  #crear una matriz d
  d=np.zeros([n,2])
  for i in range (n):
    #obtener distancias
    d[i,0]=disMinkowski(x0,X_data[i],p)
    #obtener etiquetas
    d[i,1]=y_data[i]
  #regresar una lista de tuplas en lugar de la matriz numpy
  tupla = tuple([tuple(e) for e in d])
  lista= list(d)
  return lista
#funcion para obtener la distancia de un punto nuevo a los centroides
def NearCentroide(X, y, Xnew,p):
  X_centro, y_centro= getCentroids(X,y)
  Mxnew=len(Xnew)
  ynew=np.zeros(Mxnew, dtype=int)
  for i in range (Mxnew):
    dist= getDistances(Xnew[i], X_centro, y_centro,p)
    dist.sort(key=lambda pair:pair[0])
    ynew[i]=dist[0][1]
  return ynew
#funcion 2 para el split del data set
def SplitData(X,y,size_test):
  N=len(X)
  i_arr=np.arange(N)
  nrd.shuffle(i_arr)
  N_test=int(N*size_test)
  X_test=X[i_arr[:N_test]]
  y_test=y[i_arr[:N_test]]
  X_train=X[i_arr[N_test:]]
  y_train=y[i_arr[N_test:]]
  return X_train,X_test,y_train, y_test
#calcular exactitud del clasificador
def accuracy(y_test, y_pred):
  N_test=len(y_test)
  acc=0
  for i in range(N_test):
    if y_test[i]==y_pred[i]:
      acc+=1
  return acc/N_test
#funcion para repetir la exactitud varias veces
def randomSampling(X,y,rep):
  acc=np.zeros(rep)
  for i in range (rep):
    X_train,X_test,y_train,y_test=SplitData(X,y,0.20)
    y_pred=NearCentroide(X_train, y_train, X_test,p=2)
    acc[i]=accuracy(y_test,y_pred)
  return np.mean(acc)
# Graficar puntos y centroides
# Entrada: datos - X_data, centroides - X_centroids
# Salida:  gráfica los puntos con los centroides
def plotCentroids( X_data, y_data, X_centroids, y_centroids):
    labels = np.unique(y_data)  # conjunto de etiquetas
    clases =  ['setosa','versicolor','virginica']
    colors  = ['tab:red', 'tab:green', 'tab:blue']
    markers = ['.', '.', '.']

    for k in labels:
        # seleccionar filas etiquetadas con la k-ésima categoría
        X_points = X_data[ y_data == k]
        plt.plot( X_points[:,0], X_points[:,1], ".", color=colors[k])

        # graficar centroids
        X_centro = X_centroids[ y_centroids == k][0]
        plt.plot( X_centro[0], X_centro[1], "o", color=colors[k], label=clases[k] )

        plt.legend(loc="best")


