# importar biliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import numpy.random as nrd
import math
import sys

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

        
# mostrar imagen de dígito
def image_show(i, data, label):
    print('The image label of index %d is %d.' %(i, label[i]))

    digit = data[i]                        # get the vectorized image
    digit_pixels = digit.reshape(28, 28)   # reshape it into 28x28 format
    plt.imshow(digit_pixels, cmap='gray')  # show the image

    
#grafica puntos
def plotPoints( X_data, y_data, color=True ):
    if color == True:
        labels = np.unique(y_data)  # conjunto de etiquetas
        for k in labels:
            # seleccionar filas etiquetadas con la k-ésima categoría
            X_points = X_data[ y_data == k]
            plt.plot( X_points[:,0], X_points[:,1], ".", label=k )

            plt.legend(loc="best")
    else:
        plt.plot( X_data[:,0], X_data[:,1], ".")

        
#paso 1
#inicializacion
#Obtener aleatoriamente k centroides del conjunto de datos
def randomCentroids(X,k):
  N = len(X)
  filas = np.arange(N)
  k_filas = nrd.choice(filas, k, replace=False)
  X_centroids = np.copy(X[k_filas])
  y_centroids = np.arange(k)
  return X_centroids, y_centroids


# Graficar puntos y centroides
def plotPointsCentroids( X_points, X_centroids, y_centroids):
    # graficar todos los puntos con "."
    plt.plot( X_points[:,0], X_points[:,1], "." )

    # graficar centroides con "o"
    for x, y in zip(X_centroids, y_centroids):
        plt.plot( x[0], x[1], "o", label="Clase " + str(y))

    plt.legend(loc="best")

    
#paso 2
#asignacion
#funcion para asignar centroides
def AssignCentroide(X_centroids, y_centroids, X,p=2):
  Mxnew=len(X)
  ynew=np.zeros(Mxnew, dtype=int)
  for i in range (Mxnew):
    dist= getDistances(X[i], X_centroids, y_centroids,p)
    dist.sort(key=lambda pair:pair[0])
    ynew[i]=dist[0][1]
  return ynew


#paso 4
#repeticion
#repetir los pasos anteriores
def k_means(X,k,MAXITE,p=2):
  X_centroids,y_centroids=randomCentroids(X,k)
  for i in range(MAXITE):
    y_pred=AssignCentroide(X_centroids, y_centroids, X,p)
    X_centroids,y_centroids=getCentroids(X,y_pred)
  return y_pred

# Generar aleatoriamente los k centroides
# Entrada: datos sin etiquetas (X_data), número de clusters (k)
# Salida: k centroides seleccionados con su propia etiqueta (X_centroids, y_centroids)
def initCentroids( X_data, k ):
    N, n = X_data.shape
    X_centroids = np.zeros([k,n])   # crear k centroides
    y_centroids = np.arange(k)      # crear etiquetas de los clusters

    # primer centroide
    X_centroids[0] = X_data[nrd.randint(0,N+1)]

    # calcular los k-1 centroides restantes
    for c in range(1,k):

        # calcular distancia mínima de los puntos a los centroides
        dist = np.zeros(N)
        for i in range(N):
            # valor máximo
            min_dist = sys.maxsize
            # recorrer centroides seleccionados
            for j in range(c):
                d_x = disMinkowski(X_data[i], X_centroids[j],p=2)
                if d_x < min_dist:
                    min_dist = d_x
            dist[i] = min_dist

        # calcular la distancia máxima al centroide más cercano
        X_centroids[c] = X_data[np.argmax(dist)]

    return X_centroids, y_centroids

  
# método de k-means++
# Entrada: conjunto de puntos a agrupar (X_data), número de clusters (k), máximo de iteraciones (MAXITE)
# Salida:  etiquetas de los puntos indicando el grupo asignado (y_pred)
def k_meanspp(X_data, k, MAXITE):
    X_centroids, y_centroids = initCentroids(X_data, k)              # inicialización

    for i in range(MAXITE):
        X_copy=np.copy(X_centroids)
        y_pred = AssignCentroide(X_centroids, y_centroids, X)      # asignación
        X_centroids, y_centroids = getCentroids(X_data, y_pred)   # actualización
        #criterio de paro
        if np.array_equal(X_copy, X_centroids)==True:
            break
    print("iteraciones", i)
    return y_pred
  
# definir función 'confusionMatrix()'
def confusionMatrix(y_pred,y_test):
  y_unique = np.unique(y_test)
  n = len(y_unique)
  m = len(y_pred)
  MatrixC = np.zeros([n,n])
  for i in range(n):
     for j in range(m):
        if y_pred[j] == y_unique[i]:
          for k in range(n):
            if y_test[j] == y_unique[k]:
              MatrixC[i,k] = MatrixC[i,k]+1
  return MatrixC


# definir método de k-means con criterios de paro
def k_means_Criterio_1(X,k,p=2):
  X_centroids,y_centroids=randomCentroids(X,k)
  for i in range(100):
    X_copy=np.copy(X_centroids)
    y_pred=AssignCentroide(X_centroids, y_centroids, X,p)
    X_centroids,y_centroids=getCentroids(X,y_pred)
    #criterio de paro
    if np.array_equal(X_copy, X_centroids)==True:
      break
  print("iteraciones", i)
  return y_pred
def k_means_Criterio_2(X,k,p=2):
  X_centroids,y_centroids=randomCentroids(X,k)
  y_pred=AssignCentroide(X_centroids, y_centroids, X,p)
  for i in range(100):
    y_copy=np.copy(y_pred)
    X_centroids,y_centroids=getCentroids(X,y_pred)
    y_pred=AssignCentroide(X_centroids, y_centroids, X,p)
    #criterio de paro
    if np.array_equal(y_pred, y_copy)==True:
      break
  print("iteraciones", i)
  return y_pred


#obtener distancias
def ObtenerDistancias(x0,X_data,p):
  #obtener tamaño de la matriz
  n=len(X_data)
  #crear una matriz d
  d=np.zeros(n)
  for i in range (n):
    #obtener distancias
    d[i]=disMinkowski(x0,X_data[i],p)
  return d


#definir función 'elbow_method()'
def elbow(X, N_clusters):
  #definir vector de inercia
  inercia=np.zeros(N_clusters)
  suma=np.zeros(N_clusters)
  #ciclo exterior para el k_means
  for i in range(1,N_clusters+1):
        y_pred=k_means(X,i,MAXITE=10,p=2)
        X_centroids,y_centroids=getCentroids(X, y_pred)
        #ciclo interno para calcular la distancia a los centroides de acuerdo a su clasificacion
        for j in range (i):
          posiciones=np.where(y_pred==y_centroids[j-1])
          dist=(ObtenerDistancias(X_centroids[j-1],X[posiciones],p=2))**2
          suma[j-1]=np.sum(dist)
        inercia[i-1]=np.sum(suma)
  return inercia


from sklearn.metrics import silhouette_score
# definir función 'silhouette()'
def silhouette(X, N_clusters):
  coefSilueta=np.zeros(N_clusters-1)
  for i in range (2, N_clusters+1):
    y_pred=k_means(X,i,MAXITE=30,p=2)
    coefSilueta[i-2] = silhouette_score(X, y_pred,metric='euclidean')
  return coefSilueta

#definir get_neighbors()
def get_neighbors(X_train, y_train, new_point, k):
    #obtener distancias
    distances = np.array([disMinkowski(new_point, x,p=2) for x in X_train])
    #ordenar los indices
    sorted_indices = np.argsort(distances)
    k_neighbors_indices = sorted_indices[:k]
    # Retorna los vecinos  y las  etiquetas de los vecinos
    return X[k_neighbors_indices], y_train[k_neighbors_indices]

# definir plotNeighbours()
def plotNeighbours(X_train, y_train, new_point, k):
    # Crea una gráfica de dispersión
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Conjunto de Entrenamiento')
    # Marca el punto a clasificar con un color diferente
    plt.scatter(new_point[0], new_point[1], color='red', marker='x', s=100, label='Nuevo Punto a Clasificar')
    # Marca los k vecinos más cercanos con un círculo
    k_neighbors, k_etiquetas = get_neighbors(X_train, y_train, new_point, k)
    plt.scatter(k_neighbors[:, 0], k_neighbors[:, 1], color='green', marker='o', s=100, label=f'{k} Vecinos Más Cercanos')

    # Configura la leyenda y muestra la gráfica
    plt.legend()
    plt.title("K-vecinos mas cercanos")
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.show()

# definir getPrediction()
def getPrediction(X_train, y_train, new_point, k):
    #llamar a la funcion
    k_vecinos, k_etiquetas = get_neighbors(X_train, y_train, new_point, k)
    # Determina la etiqueta más común entre los k vecinos
    most_common = np.bincount(k_etiquetas)
    most_common_etiqueta=np.argmax(most_common)
    return most_common_etiqueta
  
# definir función knn()
def knn(X_train,y_train,X_new,k):
    n=len(X_new)
    y_pred=np.zeros(n)
    for i in range (n):
      y_pred[i] = getPrediction(X_train, y_train, X_new[i], k)
    return y_pred
  
# definir plotKAccuracy()
def plotKAccuracy(X_train,y_train, X_test,y_test, k_max):
  accuracy_vector=np.zeros(k_max)
  for i in range (1,k_max+1):
    y_pred=knn(X_train,y_train,X_test,i)
    accuracy_vector[i-1]=ML.accuracy(y_test,y_pred)
  plt.plot(range(1, k_max+1), accuracy_vector)
  plt.plot(range(1, k_max+1), accuracy_vector,"bo")
  plt.title('k_ideal')
  plt.xlabel('k_valores')
  plt.ylabel('k_accuracy')
  plt.grid()
  plt.show()

  
