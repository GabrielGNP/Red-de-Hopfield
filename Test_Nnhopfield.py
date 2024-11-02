# libreias
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
## Carga de Datasets
# ============================================================


# Colores para mostrar
colors = ListedColormap(['white', 'black'])

#Formas. Tamaño 5x5 (4 patrones)
dataset1 = np.array(([
[
[0, 0, 0, 0, 0],
[0, 1, 1, 1, 0],
[0, 1, 0, 1, 0],
[0, 1, 1, 1, 0],
[0, 0, 0, 0, 0]],
##################
[
[1, 1, 0, 0, 0],
[0, 1, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 1, 0],
[0, 0, 0, 1, 1]],
##################
[
[1, 1, 1, 1, 1],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[1, 1, 1, 1, 1]],
##################
[
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],
[1, 0, 0, 0, 1]]
]))

#Tamaño de las imagenes 7x5 (10 patrones)
dataset2 = np.array([
[#Nº 1
[0, 0, 1, 0, 0],
[0, 1, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 1, 1, 1, 0]],
##################
[#Nº 2
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1],
[0, 0, 0, 0, 1],
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 0],
[1, 0, 0, 0, 0],
[0, 1, 1, 1, 1]],
##################

[#Nº 3
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1],
[0, 0, 0, 0, 1],
[0, 1, 1, 1, 0],
[0, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[0, 1, 1, 1, 0]],
##################

[#Nº 4
[0, 0, 0, 1, 0],
[0, 0, 1, 1, 0],
[0, 1, 0, 1, 0],
[1, 0, 0, 1, 0],
[1, 1, 1, 1, 1],
[0, 0, 0, 1, 0],
[0, 0, 0, 1, 0]],
##################

[#Nº 5
[1, 1, 1, 1, 1],
[1, 0, 0, 0, 0],
[1, 1, 1, 1, 0],
[0, 0, 0, 0, 1],
[0, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[0, 1, 1, 1, 0]],
##################

[#Nº 6
[0, 0, 1, 1, 0],
[0, 1, 0, 0, 0],
[1, 0, 0, 0, 0],
[1, 1, 1, 1, 0],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[0, 1, 1, 1, 0]],
##################

[#Nº 7
[1, 1, 1, 1, 1],
[0, 0, 0, 0, 1],
[0, 0, 0, 1, 0],
[0, 0, 1, 0, 0],
[0, 1, 0, 0, 0],
[0, 1, 0, 0, 0],
[0, 1, 0, 0, 0]],
##################

[#Nº 8
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[0, 1, 1, 1, 0]],
##################

[#Nº 9
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[0, 1, 1, 1, 1],
[0, 0, 0, 0, 1],
[0, 0, 0, 1, 0],
[0, 1, 1, 0, 0]],
##################

[#Nº 0
[0, 0, 1, 1, 0],
[0, 1, 0, 0, 1],
[0, 1, 0, 0, 1],
[0, 1, 0, 0, 1],
[0, 1, 0, 0, 1],
[0, 1, 0, 0, 1],
[0, 0, 1, 1, 0]],
##################
])

#Tamaño de las imagenes 7x5 (3 patrones)
dataset3 = np.array([
[#Nº 1
[0, 0, 0, 0, 0],
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1],
[1, 1, 1, 1, 1],
[1, 0, 0, 0, 1],
[1, 0, 0, 0, 1]],
##################
[#Nº 2
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 0],
[1, 0, 0, 0, 0],
[1, 0, 0, 0, 0],
[1, 0, 0, 0, 0],
[1, 0, 0, 0, 0],
[0, 1, 1, 1, 0]],
##################

[#Nº 3
[1, 1, 1, 1, 1],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0]]
])

#Tamaño de las imagenes 9x9 (3 patrones)
dataset4 = np.array([
[#Nº 1
[0, 1, 1, 1, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0, 0, 0, 0],
[0, 1, 1, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0]],
##################
[#Nº 2
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 1, 1, 1, 1]],
##################

[#Nº 3
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 0, 0]]
])

## Transformación de datos
# Como los datos están almacenados en formato 0 y 1. para poder generar el ruido despues se modificará multiplicando todos los valores por 2 y luego restandole 1.
# -1 = 0*2-1
# 1 = 1*2-1
# [0, 0, 0,]
# [1, 1, 1,]
# [0, 1, 0,]
# entonces
# [-1, -1, -1,]
# [ 1,  1,  1,]
# [-1,  1, -1,]
dataset1 = dataset1*2-1
dataset2 = dataset2*2-1
dataset3 = dataset3*2-1
dataset4 = dataset4*2-1
# Esto tambien facilitará la manipulación del ruido en la imagen canviando el signo de los valores en lugar de evaluar que valor es.


# ============================================================
## inputs de Datos
# ============================================================

## Dataset1
arraux = np.array(([
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],
[1, 0, 0, 0, 1]]
))

## Dataset2
# arraux = np.array(([
# [0, 1, 1, 1, 0],
# [1, 0, 0, 0, 1],
# [0, 0, 0, 0, 1],
# [0, 1, 1, 1, 0],
# [0, 0, 0, 0, 1],
# [1, 0, 0, 0, 1],
# [0, 1, 1, 1, 0]]
# ))


## Dataset4
# arraux = np.array(([
# [0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 1, 1, 1, 1],
# [0, 0, 0, 0, 0, 1, 1, 1, 1],
# [0, 0, 0, 0, 0, 1, 1, 1, 1],
# [0, 0, 0, 0, 0, 1, 1, 1, 1]]
# ))

arraux = arraux *2 - 1


# ============================================================
## Configuración
# ============================================================
datasetUse = dataset1
lvlRuido = 0.3 #valores entre 0 y 1
Cnt_Iteraciones = 500
num_converg = 20
# ============================================================
## Funciones
# ============================================================

# Funcion para aplicar ruido a los datos
# El porcentaje será el nivel de ruido que tendrá la imagen a generar
def datanoice(data,porcentaje):
  data2 = data.copy() # Hace una copia del dato original para no modificarlo
  for i in range(data2.shape[0]): # range(data2.shape[0]) devuelve el número de filas
    for j in range(data2.shape[1]): # range(data2.shape[0]) devuelve el número de columnas
        if np.random.randint(10)< (porcentaje*10):  #Si el número aleatorio obtenido es menor al porcentaje de ruido dado, entonces se modificará el valor de la imagen en la posicion dada.
          data2[i,j] = -(data[i,j]) # se invierte el signo del dato. (-1 => 1 y 1 => -1)
  return data2
# el .shape devuelve las dimensiones del array. 
# Si el array guarda varias matrices. devolverá la cantidad de matrices que guarda, la cantidad de filas y la cantidad de columnas
# SI el array guarda una única matriz entonces devolverá la cantidad de filas y la cantidad de columnas
# [0] => filas
# [1] => columnas


# graficar el dataset
def drawData(data):
    plt.figure()
    for img in range(data.shape[0]):
        plt.subplot(1,data.shape[0],img+1)
        plt.imshow(data[img], cmap=colors)
    plt.show(block=False)

# drawData(dataset1)
# drawData(dataset2)
# drawData(dataset3)
# drawData(dataset4)
drawData(datasetUse)


# ============================================================
## Red de Hopfield
# ============================================================
class redHopfield():
  # Inicializador de la red de Hopfield
  # n_neurons: El número de neuronas en la red.
  # n_memory: Representa el número de patrones que deseas almacenar. Si el número de patrones supera cierta capacidad (determinado por una fórmula basada en el número de neuronas), el código emite una advertencia de que la red puede no funcionar correctamente.
  # self.w_ij: Es la matriz de pesos sinápticos, que controla cómo las neuronas están conectadas entre sí.
  # self.yout: Representa el estado de salida actual de las neuronas (inicialmente, todas las salidas son 0).
  def __init__(self,n_neurons,n_memory=0):
    self.n_neurons = n_neurons
    if(n_memory!=None):
      check = n_neurons/(2*np.log(n_neurons)) # condición teórica para la capacidad de almacenamiento de las redes de Hopfield.
      if check <= n_memory:
        print("No se cumple la expresion de almacenamiento con 1% de error")
    self.w_ij = np.zeros((self.n_neurons,self.n_neurons))  # Inicializa la matriz de pesos
    self.yout = np.zeros(self.n_neurons)  # Inicializa las salidas de las neuronas

  # método encargado de calcular la salida de las neuronas dado un patrón de entrada.
  # Este método actualiza el estado de las neuronas (self.yout) basándose en el estado anterior y los pesos sinápticos.
  # datain: Es el patrón de entrada que quieres procesar.
  def salidaenN(self,datain):
    dim = datain.shape
    datain = datain[:].squeeze() # Elimina dimensiones unitarias
    dataout = np.zeros_like(datain[0])  # Inicializa un array para la salida
    for it in range(datain.shape[0]):
        if datain[it] != 0:
          dataout[it] = np.sign(datain[it]) # np.sign(): Devuelve el signo de cada número en el array: -1 si es negativo, 1 si es positivo. La utilidad es calcular el signo del valor de entrada
        else:
          # Calcula el valor de salida utilizando los pesos sinápticos
          dataout[it] = np.sign(np.dot(np.remove(self.w_ij[it],it) ,np.remove(self.yout,it))) # np.dot(): Es el producto punto de dos arrays.
          # np.remove(): Elimina un valor específico de un array (aquí se eliminan los pesos y salidas de la neurona actual para evitar bucles de autoconexión).
    self.yout = dataout.squezze()  # Actualiza las salidas
    return dataout.reshape(dim)

  # Este método entrena la red para almacenar patrones, ajustando los pesos sinápticos de acuerdo a los patrones proporcionados.
  # data: Es un conjunto de patrones que la red debe memorizar.
  # Entrenamiento: Se basa en la regla de Hebb, donde los pesos se ajustan en función del producto de las salidas de dos neuronas. Esto hace que los patrones se "almacenen" en la red.
  def train(self,data):
    Pval = data.shape[0] # Número de patrones
    data = data.reshape(Pval,1,-1).squeeze() # Asegura que los datos estén en la forma correcta
    nfeat = data.shape[1] # Número de neuronas o características
    for it1 in range(0,self.w_ij.shape[0]):
      for it2 in range(it1+1,self.w_ij.shape[1]): # Actualiza los pesos sinápticos utilizando el producto punto entre neuronas
        self.w_ij[it1,it2] = self.w_ij[it2,it1] = (1/nfeat)* np.dot(data.T[it1],data.T[it2])


  # Este método intenta recuperar un patrón a partir de una entrada. Es una variante que probablemente tenga algunos problemas (por eso el nombre "BAD").
  # maxit: Número máximo de iteraciones permitidas para alcanzar la convergencia.
  # converg: Número de iteraciones consecutivas que deben pasar para que consideremos que la red ha convergido.
  def predict(self,input,maxit=1000,converg=10):
      #inicializacion de salidas
      inp = input.copy()
      self.yout = inp.reshape(1,-1).squeeze()
      auxconv = 0
      np.random.RandomState(66)
      self.yout = np.array(self.yout,dtype=np.float32)
      for it in range(maxit):
        yold = self.yout.copy()
        j = np.random.randint(self.w_ij.shape[0]) # Selecciona una neurona al azar
        self.yout[j] = np.sign(self.w_ij[j] @ self.yout)
        if np.linalg.norm(np.sign(self.yout)-np.sign(yold))<1e-5:
          auxconv += 1
        else:
          auxconv = 0
        if auxconv>=converg:
          break # Detiene si ha convergido
      if it==1000:
        print("Numero maximo de iteraciones")
      return self.yout.reshape(input.shape),it
  


# =====================================================
## Funcionamiento
# =====================================================

# Entrenamiento
# Se crea una instancia de la clase redHopfield, que es la red de Hopfield en sí.
# hay que tener en cuenta que todos los datos (matrices) del dataset tienen el mismo tamaño.
# El argumento dataset1.shape[1]*dataset1.shape[2] indica el número de neuronas que tendrá la red. cantidad de filas por cantidad de columnas
net = redHopfield(datasetUse.shape[1]*datasetUse.shape[2])
# Se llama al método train() de la clase redHopfield para entrenar la red usando el conjunto de datos dataset1.
# dataset1 contiene los patrones que quieres almacenar en la red. Estos patrones son los que la red de Hopfield aprenderá y tratará de recordar.
net.train(datasetUse)


# imagein e imagepred se usarán para almacenar las imágenes con ruido y las predicciones hechas por la red.
# imagenin para las imagenes con ruido
# imagepred para las imagenes de prediccion.
imagein = []
imagepred = []

# ===== Para cuando se quiere trabajar con una única imagen con ruido =====
auxim = datanoice(arraux,lvlRuido) # genera una nueva imagen con un 40% de ruido (o probabilidad de ruido) utilizando una imagen original.
#auxim = arraux
imagein.append(auxim)  # Guarda la imagen ruidosa
predicho, iteraciones = net.predict(auxim,maxit=Cnt_Iteraciones,converg=num_converg)  # se ejecuta la funcion de predict de la red de hopfield para que genere una predicción
imagepred.append(predicho) # Guarda la imagen predicha
print(iteraciones, "Iteraciones")


# Esto crea una figura donde se muestran las imágenes de entrada (las imágenes con ruido)
plt.figure(figsize=(5,5))
plt.title("Entrada")
for img in range(len(imagein)): # se utiliza un for para leer todas las imagenes con ruido
  plt.subplot(1,len(imagein),img+1)
  plt.imshow(imagein[img])
plt.show(block=False)

# Muestra las imágenes resultantes de la red Hopfield después de intentar "reparar" las imágenes con ruido.
plt.figure(figsize=(5,5))
plt.title("Salida de la red")
for img in range(len(imagepred)): # se utiliza un for apra leer todas las imagenesr con ruido
  plt.subplot(1,len(imagepred),img+1)
  plt.imshow(imagepred[img])
plt.show()