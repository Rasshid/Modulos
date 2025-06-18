###########################################################################################################################################
############################################# Mi módulo ###################################################################################
#22/10/2024
# Módulo sobre machine learning v2
#############################################Librerías#####################################################################################
# importar biliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import numpy.random as nrd
import math
import sys

#################################################### Deteccion valores atipicos

# definir función 'drop_outliers()' usando desviacion estandar
def drop_outliers(df, column):
  # obtener desviación estándar de la columna
  DesvE = df[column].std()
  # calcular el umbral superior
  UmbSup= 3 * DesvE
  #obtener mediana
  med= df[column].median()
  #reemplazar por la mediana los valores atipicos

  ### SE DEBE REALIZAR CON FILTROS Y VECTORIZACIÓN
  for x in df.index:
    if df.loc[x, column] > UmbSup:
      df.loc[x, column]=med
  #regresar el data frame
  return df

# Usando cuartiles

def drop_outliers2(df, column):
  # obtener cuartil superior e inferior
  q75, q25 =np.percentile(df[column], [75, 25])
  #rango intercuartìlico
  S=1.5*(q75-q25)
  # calcular el umbral superior e inferior
  UmbSup= q75+S
  UmbInf= q25-S
  #obtener mediana
  med= df[column].median()
  #reemplazar por la mediana los valores atipicos

  ### SE DEBE REALIZAR CON FILTROS Y VECTORIZACIÓN
  for x in df.index:
     ### MUY BIEN POR APLICAR LOS DOS UMBRALES
    if (df.loc[x, column] > UmbSup) or (df.loc[x, column] < UmbInf):
      df.loc[x, column]=med
  #regresar el data frame
  return df
##################################################### Funciones centroide y distancia

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

###############################################Distancia entre puntos con etiqueta

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


#Separación en prueba y entrenamiento
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

############################################### Accuracy

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

###################################################### Graficacion

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

################################################## K-means

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

#################################################### k-means++

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

################################################## Matriz de confusion  
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

#################################################### k-means criterio de paro

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

# criterio 2
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

########################################### Método del codo

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

########################################### Método de la silueta

from sklearn.metrics import silhouette_score
# definir función 'silhouette()'
def silhouette(X, N_clusters):
  coefSilueta=np.zeros(N_clusters-1)
  for i in range (2, N_clusters+1):
    y_pred=k_means(X,i,MAXITE=30,p=2)
    coefSilueta[i-2] = silhouette_score(X, y_pred,metric='euclidean')
  return coefSilueta

############################################################# KNN

#definir get_neighbors()
def get_neighbors(X_train, y_train, new_point, k):
    #obtener distancias
    distances = np.array([disMinkowski(new_point, x,p=2) for x in X_train])
    #ordenar los indices
    sorted_indices = np.argsort(distances)
    k_neighbors_indices = sorted_indices[:k]
    # Retorna los vecinos  y las  etiquetas de los vecinos
    return X_train[k_neighbors_indices], y_train[k_neighbors_indices]

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

########################################################## estandarizacion

# función que realiza el escalado estándar
def escEstandar(X_data):
    X_norm = np.zeros_like(X_data)
    X_mean = np.mean(X_data, axis=0)      # media de las columnas
    X_std  = np.std(X_data, axis=0)       # desviación estándar de las columnas
    X_norm = (X_data - X_mean)/X_std      # escalar datos

    return X_norm

######################################################### Regresion logística gradiente descendente
# Regresion lineal múltiple
def regress_mul(X,w):
    z = np.dot(X,w[1:]) + w[0]
    return z

# función sigmoide
def sigmoid(z):
    phi_z = 1.0 / (1.0 + np.exp(-np.clip(z,-250,250)))
    return phi_z

 # predicción de la regresión logística
def predictLogic(X, w):
    z = regress_mul(X,w)
    y_pred = np.where(z>=0.0,1,0)
    return y_pred

# función de costo entropía cruzada
def costLogic(y, phi_z):
    n = len(y)
    loss = -(1/n)* ( np.sum( y*np.log(phi_z) + (1-y)*np.log(1-phi_z) ) )
    return loss

# regresión logística con descenso del gradiente
def fitLogic(X,y,w,alpha,epochs):

    loss = np.zeros(epochs)
    n = len(X)                     # cantidad de datos

    for i in range(epochs):
        # calcular el valor de la predicción
        z = regress_mul(X,w)
        phi_z = sigmoid(z)

        # derivadas parciales
        error = (phi_z - y)
        dw0 = np.sum(error)/n        # dJ/dw0
        dwj = np.dot(X.T,error)/n    # dJ/dwj

        # actualizar los pesos usando el gradiente descendente
        w[0]  = w[0]  - alpha*dw0
        w[1:] = w[1:] - alpha*dwj

        # actualizar el valor del costo logístico
        loss[i] = costLogic(y, phi_z)

    return w, loss

###################################################### Regresion logística descenso del gradiente estocástico v1

#funcion que separa en lotes a X e y
def lote_X_y(X,y,n,size=1):
      if size > n:
             size=n
      idx=np.random.choice(n,size,replace=False)
      return size, X[idx],y[idx]

#Minimizar la funcion de costos
def fit_Logit_estoch(X, y, w, alpha, epochs,size=1):
    cost = np.zeros(epochs)
    m = len(y)  # Número de muestras

    for i in range(epochs):
        # seleccionar un lote de datos
        n_batch, lote_x, lote_y = lote_X_y(X, y, m, size)
        z = regress_mul(lote_x, w)
        phi = sigmoid(z)
        errors = phi - lote_y  # Calcular los errores como la diferencia entre la predicción y la verdad

        #Minimizar funcion de costos (entropía cruzada) usando el descenso del gradiente estocástico

        w[1:] -= alpha * (1/n_batch) * np.dot(lote_x.T, errors)  # Actualizar pesos
        w[0] -= alpha * (1/n_batch) * np.sum(errors)        # Actualizar el sesgo

        # Calcular la función de costo (log-verosimilitud negativa)
        phi=sigmoid(regress_mul(X,w)) #actualizar phi
        cost[i] = -np.mean(y * np.log(phi) + (1 - y) * np.log(1 - phi))

    return w, cost

#funcion de prediccion binaria
def predict_logit(X, w):
    z = regress_mul(X, w)
    phi = sigmoid(z)
    return (phi >= 0.5).astype(int)

################################################ Regresión logística gradiente descendente estocástico v2
#funcion para obtener lotes del mismo tamaño
def getBatch(len_x, batch_size):
  idx=np.arange(len_x)
  #revolver índices
  np.random.shuffle(idx)
  return len_x//batch_size, idx

#Minimizar la funcion de costos
def fit_Logit_estoch_new(X, y, w, alpha, epochs,num_batch):
    cost = np.zeros(epochs)
    m = len(y)  # Número de muestras
    batch_size,idx=getBatch(m, num_batch)
    for i in range(epochs):
      for k in range(num_batch):
        X_batch, y_batch = X[k*batch_size:(k+1)*batch_size], y[k*batch_size:(k+1)*batch_size]
        z = regress_mul(X_batch, w)
        phi = sigmoid(z)
        errors = phi - y_batch  # Calcular los errores como la diferencia entre la predicción y la verdad

        #Minimizar funcion de costos (entropía cruzada) usando el descenso del gradiente estocástico

        w[1:] -= alpha * (1/batch_size) * np.dot(X_batch.T, errors)  # Actualizar pesos
        w[0] -= alpha * (1/batch_size) * np.sum(errors)        # Actualizar el sesgo

      # Calcular la función de costo (log-verosimilitud negativa)
      phi=sigmoid(regress_mul(X,w)) #actualizar phi
      cost[i] = -np.mean(y * np.log(phi) + (1 - y) * np.log(1 - phi))

    return w, cost

#################################################### Regresion lineal múltiple descenso del gradiente

# función que calcula los valores de la función de costo J BASADO EN EL MSE (mean square error)
def cost(X, y, w):
    y_pred = regress_mul(X, w)
    error = (y - y_pred)**2
    loss = np.mean(error)
    return loss

# método del gradiente descendente múltiple
def gradient(X, y, w, alpha, epochs):
  costos = np.zeros(epochs)
  n = len(X) # cantidad de datos
  for k in range(epochs):
    # calcular el valor de la predicción
    y_pred = regress_mul(X, w)
    # gradientes: derivadas de la función de error
    error = (y - y_pred)
    dw_0 = -(2/n) * np.sum(error) # dJ/d w_0
    dw_j = -(2/n) * np.dot(X.T,error) # dJ/d w_1
    # actualizar los pesos usando el gradiente descendente
    w[0] = w[0] - alpha * dw_0
    w[1:] = w[1:] - alpha*dw_j
    # actualizar el valor del costo
    costos[k] = cost(X, y, w)
  return w, costos, k+1

################################################### Regresión lineal múltiple descenso del gradiente estocástico v1
#funcion que separa en lotes a X e y
def lote_X_y(X,y,n,size=1):
      if size > n:
             size=n
      idx=np.random.choice(n,size,replace=False)
      return size, X[idx],y[idx]

# método del gradiente descendente versión estocástica
def gradient_estochastic(X, y, w, alpha, size, epochs):
    costos = np.zeros(epochs)
    n = len(X)                               # cantidad de datos
    for k in range(epochs):
        # seleccionar un lote de datos
        n_batch, lote_x, lote_y = lote_X_y(X, y, n, size)
        # calcular el valor de la predicción
        y_pred = regress_mul(lote_x, w)

        # gradientes: derivadas de la función de error
        error = (lote_y - y_pred)
        dw_0 = -(2/n_batch) * np.sum(error)         # dJ/d w_0
        dw_j = -(2/n_batch) * np.dot(lote_x.T,error)       # dJ/d w_1

        # actualizar los pesos usando el gradiente descendente
        w[0] = w[0] - alpha * dw_0
        w[1:] = w[1:] - alpha*dw_j

        # actualizar el valor del costo
        costos[k] = cost(X, y, w)
    return w, costos, k+1
###################################################  Regresion lineal multiple usando Descenso del gradiente estcástico v2
#funcion de lote
def getBatch(len_x, batch_size):
  idx=np.arange(len_x)
  #revolver índices
  np.random.shuffle(idx)
  return len_x//batch_size, idx

# método del gradiente descendente múltiple
def gradient_estoch_new(X, y, w, alpha, epochs,num_batch):
  costos = np.zeros(epochs)
  n = len(X) # cantidad de datos
  batch_size,idx=getBatch(n, num_batch)
  #ciclo para las épocas
  for k in range(epochs):
    #ciclo para los lotes
    for i in range(num_batch):
        X_batch=X[i*batch_size:(i+1)*batch_size]
        y_batch=y[i*batch_size:(i+1)*batch_size]
        # calcular el valor de la predicción
        y_pred = regress_mul(X_batch, w)
        # gradientes: derivadas de la función de error
        error = (y_batch - y_pred)
        dw_0 = -(2/batch_size) * np.sum(error) # dJ/d w_0
        dw_j = -(2/batch_size) * np.dot(X_batch.T,error) # dJ/d w_1
        # actualizar los pesos usando el gradiente descendente
        w[0] = w[0] - alpha * dw_0
        w[1:] = w[1:] - alpha*dw_j
    # actualizar el valor del costo
    costos[k] = cost(X, y, w)
  return w, costos, k+1

########################################################### SVM

# distancias
def distances(X,Y, w,b, with_lagrange=True):
    dist = Y * (np.dot(X, w)+b) - 1

    # get distance from the current decision boundary
    # by considering 1 width of margin

    if with_lagrange:  # if lagrange multiplier considered
        # if distance is more than 0
        # sample is not on the support vector
        # Lagrange multiplier will be 0
        dist[dist > 0] = 0

    return dist

#funcion de costo svm
def get_cost_grads(X,Y,w,b,c):
        # Get distances from the decision boundary

        dist = distances(X,Y,w,b)

        # Get current cost
        L = 1 / 2 * np.dot(w, w) - c * np.sum(dist)

        dw = np.zeros(len(w))

        for ind, d in enumerate(dist):
            if d == 0:  # if sample is not on the support vector
                di = w  # (alpha * y[ind] * X[ind]) = 0
            else:
                # (alpha * y[ind] * X[ind]) = y[ind] * X[ind]
                di = w - (c * Y[ind] * X[ind])
            dw += di
        return L, dw / len(X)

#SVM
def fit_SVM(X, Y, epochs, alpha, c,w,b):
        for i in range(epochs):
            L, dw = get_cost_grads(X,Y,w,b,c)
            w = w - alpha * dw
            if i % 1000 == 0:
                print(i, ' | ', L)
        return L,w

#prediccion svm 
def predict_svm(X,w,b):
    return np.sign(np.dot(X, w)+b)

####################################################################### Perceptron

#Prediccion func para perceptron
def predict_perc(X,w):
  return np.where(np.dot(X,w[1:])+w[0]>=0.0,1,-1)

#Implementacion del perceptron
def perceptron(X,y, eta, epochs):
  #inicializar w
  rgen = np.random.RandomState(1)
  w = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])
  errors=[]
  #ciclo de las épocas
  for i in range(epochs):
    er=0
    #ciclo de los xi
    for xi, target in zip(X,y):
      update = eta * (target - predict_perc(xi,w))
      w[1:] += update * xi
      w[0] += update
      er += int(update != 0.0)
    errors.append(er)
    #condicion de paro
    if er==0:
      break
  return w,errors

###################################################################### Adaline

# net input
def net_input(X,w):
  return np.dot(X,w[1:])+w[0]

#funcion de activacion (identidad)
def activation(X):
  return X

#prediccion adaline
def predict_adaline(X,w):
  return np.where(activation(net_input(X,w))>= 0.0, 1, -1)

#Implementacion de Adaline
def Adaline(X,y, eta, epochs):
  #inicializar w
  rgen = np.random.RandomState(1)
  w = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])
  costo=[]
  #ciclo de las épocas
  for i in range(epochs):
    input = net_input(X,w)
    output = activation(input)
    error = (y-output)
    #Actualizar los pesos
    w[1:] += eta * X.T.dot(error)
    w[0] += eta * error.sum()
    cost = (error**2).sum()
    costo.append(cost)
  return w, costo

####################################################################### Validación cruzada
from sklearn.metrics import accuracy_score, r2_score
#funcion de validación cruzada
def cross_validation(X, y, num_folder,fmodel,w0,alpha,epochs,num_batch):
  size, idx=getFolders(len(X), num_folder)
  score=np.zeros(num_folder)
  for k in range(num_folder):
    #separar datos
    X_test=X[idx[k*size:(k+1)*size]]
    y_test=y[idx[k*size:(k+1)*size]]
    X_train=np.delete(X, idx[k*size:(k+1)*size], axis=0)
    y_train=np.delete(y, idx[k*size:(k+1)*size], axis=0)
    #aplicar modelo
    w,loss=f(X_train, y_train, w0, alpha, epochs,num_batch)
    # Predicciones y evaluación
    if np.all(y == y.astype(int)):
      y_pred=predict_logit(X_test, w)
      score[k]=accuracy_score(y_test, y_pred)
    else:
      y_pred=regress_mul(X_test, w)
      score[k]=r2_score(y_test, y_pred)
  return np.mean(score)


