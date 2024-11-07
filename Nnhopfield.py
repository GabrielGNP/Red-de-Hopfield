# libreias
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import os
import sys
import platform

# Importar las bibliotecas necesarias según el sistema operativo
if platform.system() == 'Windows':
    import msvcrt
else:
    import curses
    
from colorama import init, Fore, Back, Style  # Solo en Windows

# ============================================================
#region Colores para las graficas
# ============================================================
colors = ListedColormap(['white', 'black'])
colorsInput = ListedColormap(['white', 'green'])
colorsNoice = ListedColormap(['white', 'red'])
colorsOutput = ListedColormap(['white', 'blue'])



# ============================================================
#region Configuración
# ============================================================

datasetSelected = "1"

def get_base_path():
    return os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))


base_path = get_base_path()
dataset_path = os.path.join(base_path, 'datasets', "dataset"+datasetSelected, "dataset"+datasetSelected+'.csv')
input_path = os.path.join(base_path, 'datasets', "dataset"+datasetSelected, "input"+datasetSelected+'.csv')

# print(dataset_path)
# print(input_path)

lvlNoice = 0.3 #valores entre 0 y 1
Cnt_Iteraciones = 500
num_converg = 20



# ============================================================
#region Funciones
# ============================================================

# Funcion de Carga de datasets
def loadDatasetsCsv(file_path):
    matrices = []
    current_matrix = []

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            
            for row in reader:
                if not row:  # Si la fila está vacía, es el fin de una matriz
                    if current_matrix:
                        matrices.append(current_matrix) 
                        current_matrix = []
                elif "Matrix" not in row[0]:  # Evita las líneas de identificación
                    try:
                        current_matrix.append([int(x) for x in row])
                    except ValueError:
                        print("Error: El archivo contiene datos no numéricos.")
                        return np.array([])
            
            if current_matrix:  # Asegura que la última matriz se guarde
                matrices.append(current_matrix)
        
        # Convertimos la lista de matrices a un array de numpy con 3 dimensiones
        return np.array(matrices)

    except FileNotFoundError:
        print("Error: El archivo no se encontró en la ruta especificada.")
        return np.array([])
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return np.array([])

# Funcion de Carga de Input
def loadInputCsv(file_path):
    matrix = []
    found_first_matrix = False  # Bandera para identificar el primer patrón
    
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            
            for row in reader:
                # Verifica si la fila contiene "Matrix"
                if row and "Matrix" in row[0]:
                    # Si ya se ha encontrado el primer patrón, se rompe el bucle
                    if found_first_matrix:
                        break
                    found_first_matrix = True  # Marca que el primer patrón fue encontrado
                    continue
                
                # Ignora filas vacías o las filas con identificador de matriz
                if row and found_first_matrix:
                    try:
                        matrix.append([int(x) for x in row])
                    except ValueError:
                        print("Error: El archivo contiene datos no numéricos en una fila.")
                        return np.array([])

        # Convierte la lista de listas en un array de NumPy bidimensional
        return np.array(matrix)

    except FileNotFoundError:
        print("Error: El archivo de entrada no se encontró en la ruta especificada.")
        return np.array([])
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return np.array([])


# Funcion para aplicar ruido a los datos
# El porcentaje será el nivel de ruido que tendrá la imagen a generar
def datanoice(data,porcentaje):
  data2 = data.copy() 
  for i in range(data2.shape[0]): 
    for j in range(data2.shape[1]): 
        if np.random.randint(10)< (porcentaje*10):  
          data2[i,j] = -(data[i,j]) 
  return data2


# graficar el dataset
def drawData(data):
    plt.figure(figsize=(7,3))
    plt.title("DataSet")
    for img in range(data.shape[0]):
        plt.subplot(1,data.shape[0],img+1)
        plt.imshow(data[img], cmap=colors)
    plt.show(block=False)

# para limpiar la pantalla de la consola
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')




# ============================================================
#region Red de Hopfield
# ============================================================
class redHopfield():
  # Inicializador de la red de Hopfield
  # n_neurons: El número de neuronas en la red.
  # n_memory: Representa el número de patrones que deseas almacenar. Si el número de patrones supera cierta capacidad (determinado por una fórmula basada en el número de neuronas), el código emite una advertencia de que la red puede no funcionar correctamente.
  # self.w_ij: Es la matriz de pesos sinápticos, que controla cómo las neuronas están conectadas entre sí.
  # self.yout: Representa el estado de salida actual de las neuronas (inicialmente, todas las salidas son 0).
  def __init__(self,n_neurons,n_memory=3):
    self.n_neurons = n_neurons
    if(n_memory!=None):
      check = n_neurons/(2*np.log(n_neurons)) # condición teórica para la capacidad de almacenamiento de las redes de Hopfield.
      if check <= n_memory:
        print("No se cumple la condición teórica de capacidad de almacenamiento")
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
  def train(self, data):
    # Aplanar cada patrón en data
    data = np.array([pattern.flatten() for pattern in data])

    Pval = data.shape[0]  # Número de patrones
    nfeat = data.shape[1]  # Número de neuronas o características

    # Ajustamos los pesos sinápticos usando los patrones en data
    for it1 in range(self.w_ij.shape[0]):
        for it2 in range(it1 + 1, self.w_ij.shape[1]):
            # Calculamos el producto punto entre neuronas para los pesos
            self.w_ij[it1, it2] = self.w_ij[it2, it1] = (1 / nfeat) * np.dot(data.T[it1], data.T[it2])



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


# ============================================================
#region Carga de datos
# ============================================================
dataOK = True

inputData = loadInputCsv(input_path)
if inputData.size == 0:
    print("   El array de entrada está vacío o no se pudo cargar correctamente.")
    dataOK = False
else:
    inputData = inputData*2-1

datasetUsed = loadDatasetsCsv(dataset_path)
if datasetUsed.size == 0:
    print("   El array de entrada está vacío o no se pudo cargar correctamente.")
    dataOK = False
else:
    datasetUsed = datasetUsed*2-1  

# =====================================================
#region Funcionamiento de Hopfield
# =====================================================
def startHopfield():
    clear_screen()
    dataOK = True
    infoFail = ""
    inputData = loadInputCsv(input_path)
    if inputData.size == 0:
        infoFail = infoFail + "   El input está vacío o no se pudo cargar correctamente.\n"
        # print("   El array de entrada está vacío o no se pudo cargar correctamente.")
        dataOK = False
    else:
        inputData = inputData*2-1

    datasetUsed = loadDatasetsCsv(dataset_path)
    if datasetUsed.size == 0:
        infoFail = infoFail + "   El dataset está vacío o no se pudo cargar correctamente.\n"
        # print("   El array de entrada está vacío o no se pudo cargar correctamente.")
        dataOK = False
    else:
        datasetUsed = datasetUsed*2-1  

    if dataOK:
        for idx, matrix in enumerate(datasetUsed):
            if matrix.shape != datasetUsed[0].shape:
                infoFail = infoFail + "   No todos los patrones del dataset tienen el mismo tamaño.\n"
                dataOK = False
                break

    if dataOK:
        for idx, matrix in enumerate(datasetUsed):
            if matrix.shape != inputData.shape:
                infoFail = infoFail + "   El input y los datasets tienen tamaños distintos.\n"
                dataOK = False
                break

    # imagein, imageinNoice e imagepred se usarán para almacenar la imagen input, imágene con ruido y la prediccion hechas por la red.
    # imagenin para la imagen input.
    # imageninNoice para la con ruido.
    # imagepred para las imagenes de prediccion.
    imagein = []
    imageinNoice = []
    imagepred = []

    if dataOK:
        # Grafica el Dataset cargado
        drawData(datasetUsed)

        
        # Entrenamiento
        # Se crea una instancia de la clase redHopfield, que es la red de Hopfield en sí.
        # hay que tener en cuenta que todos los datos (matrices) del dataset tienen el mismo tamaño.
        # El argumento dataset1.shape[1]*dataset1.shape[2] indica el número de neuronas que tendrá la red. cantidad de filas por cantidad de columnas
        net = redHopfield(datasetUsed.shape[1]*datasetUsed.shape[2],datasetUsed.shape[0])
        # Se llama al método train() de la clase redHopfield para entrenar la red usando el conjunto de datos datasetUsed.
        # datasetUsed contiene los patrones que quieres almacenar en la red. Estos patrones son los que la red de Hopfield aprenderá y tratará de recordar.
        net.train(datasetUsed)


        # ===== Para cuando se quiere trabajar con una única imagen con ruido =====
        auxImp = datanoice(inputData,lvlNoice) # genera una nueva imagen con un 40% de ruido (o probabilidad de ruido) utilizando una imagen original.
        imagein.append(inputData)
        imageinNoice.append(auxImp)  # Guarda la imagen ruidosa
        predicho, iteraciones = net.predict(auxImp,maxit=Cnt_Iteraciones,converg=num_converg)  # se ejecuta la funcion de predict de la red de hopfield para que genere una predicción
        imagepred.append(predicho) # Guarda la imagen predicha
        print(iteraciones, "Iteraciones")




        # Esto crea una figura donde se muestran las imágenes de entrada
        plt.figure(figsize=(4,3))
        plt.title("Entrada sin ruido")
        for img in range(len(imagein)): 
            plt.subplot(1,len(imagein),img+1)
            plt.imshow(imagein[img], cmap=colorsInput)
        plt.show(block=False)

        # Esto crea una figura donde se muestran las imágenes de entrada con ruido
        plt.figure(figsize=(4,3))
        plt.title("Entrada con ruido")
        for img in range(len(imageinNoice)): 
            plt.subplot(1,len(imageinNoice),img+1)
            plt.imshow(imageinNoice[img], cmap=colorsNoice)
        plt.show(block=False)

        # Muestra las imágenes resultantes de la red Hopfield después de intentar "reparar" las imágenes con ruido.
        plt.figure(figsize=(4,3))
        plt.title("Salida de la red")
        for img in range(len(imagepred)): 
            plt.subplot(1,len(imagepred),img+1)
            plt.imshow(imagepred[img], cmap=colorsOutput)
        plt.show()

    else:
        print("")
        print("No se puede realizar la ejecución porque hay un problema con los datos:")
        print(infoFail)

    input("---Presione Enter para continuar---")





# =====================================================
#region Menú
# =====================================================

# Inicializa colorama
init(autoreset=True)

# función para detectar las teclas en window
def get_key_windows():
    return msvcrt.getch()

# función para detectar las teclas en linux
def get_key_unix(stdscr):
    return stdscr.getch()

# ================ Funciones de opciones del menú ================
# función de opción de seleccionar dataset
def optionSelectDataset():
    global dataset_path
    global input_path
    global datasetSelected
    global dataOK
    optionsSelectDatasetMenu = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]
    selected_index = 0
    if platform.system() == 'Windows':
        while True:
            clear_screen()
            for i, option in enumerate(optionsSelectDatasetMenu):
                prefix = Fore.BLACK + Back.WHITE + "-> " if i == selected_index else "   "
                print(f"{prefix}{option}")
            print()
            print()
            print("flecha ↓ para cambiar")
            print("flecha ↑ para cambiar")
            print("flecha Enter para confirmar")
            print("flecha Esc para salir")

            key = get_key_windows()
            if key == b'\xe0':  # Tecla especial
                key = msvcrt.getch()
                if key == b'H':  # Flecha arriba
                    selected_index = (selected_index - 1) % len(optionsSelectDatasetMenu)
                elif key == b'P':  # Flecha abajo
                    selected_index = (selected_index + 1) % len(optionsSelectDatasetMenu)
            elif key == b'\r':  # Enter
                if selected_index == 1:
                    datasetSelected = "2"
                    break
                elif selected_index == 2:
                    datasetSelected = "3"
                    break
                elif selected_index == 3:
                    datasetSelected = "4"
                    break
                else:
                    datasetSelected = "1"
                    break
            elif key == b'\x1b':  # Escape
                break
    else:  # Para Linux o macOS
        def wrapper(stdscr):
            nonlocal selected_index
            curses.curs_set(0)
            while True:
                clear_screen()
                for i, option in enumerate(optionsSelectDatasetMenu):
                    prefix = Fore.BLACK + Back.WHITE + "-> " if i == selected_index else "   "
                    stdscr.addstr(i, 0, f"{prefix}{option}")

                key = get_key_unix(stdscr)
                if key == curses.KEY_UP:  # Flecha arriba
                    selected_index = (selected_index - 1) % len(optionsSelectDatasetMenu)
                elif key == curses.KEY_DOWN:  # Flecha abajo
                    selected_index = (selected_index + 1) % len(optionsSelectDatasetMenu)
                elif key == curses.KEY_ENTER or key in [10, 13]:  # Enter
                    if selected_index == 1:
                        datasetSelected = "2"
                        break
                    elif selected_index == 2:
                        datasetSelected = "3"
                        break
                    elif selected_index == 3:
                        datasetSelected = "4"
                        break
                    else:
                        datasetSelected = "1"
                        break
                        
                elif key == 27:  # Escape
                    datasetSelected = "1"
                    break

        curses.wrapper(wrapper)

    dataset_path = os.path.join(base_path, 'datasets', "dataset"+datasetSelected, "dataset"+datasetSelected+'.csv')
    input_path = os.path.join(base_path, 'datasets', "dataset"+datasetSelected, "input"+datasetSelected+'.csv')

# función de cambiar el ruido
def optionChangeNoice():
    global lvlNoice
    noice = lvlNoice * 100
    if platform.system() == 'Windows':
        while True:
            clear_screen()
            print("Cuanto ruido quiere agregar a la imagen?")
            print("            " + str(noice) + "%")
            print()
            print()
            print("flecha ← para reducir 1")
            print("flecha → para aumentar 1")
            print("flecha ↓ para reducir 10")
            print("flecha ↑ para aumentar 10")
            print("flecha Enter o Esc para salir")
            
            key = get_key_windows()
            if key == b'\xe0':  # Tecla especial
                key = msvcrt.getch()
                if key == b'H':  # Flecha arriba
                    if noice<90: noice = noice + 10 
                    else: noice = 100
                elif key == b'P':  # Flecha abajo
                    if noice>10: noice = noice - 10
                    else: noice = 0
                elif key == b'K':  # Flecha izquierda
                    if noice>0: noice = noice - 1
                elif key == b'M':  # Flecha derecha
                    if noice<100: noice = noice + 1
            elif key == b'\r':  # Enter
                break
            elif key == b'\x1b':  # Escape
                break

    else:  # Para Linux o macOS
        def wrapper(stdscr):
            while True:
                clear_screen()
                print("Cuanto ruido quiere agregar a la imagen?")
                print("            " + str(noice) + "%")
                print("flecha ← para reducir 1")
                print("flecha → para aumentar 1")
                print("flecha ↓ para reducir 10")
                print("flecha ↑ para aumentar 10")
                print("flecha Enter o Esc para salir")

                key = get_key_windows()
                if key == b'\xe0':  # Tecla especial
                    key = msvcrt.getch()
                    if key == b'H':  # Flecha arriba
                        if noice<90: noice = noice + 10 
                        else: noice = 100
                    elif key == b'P':  # Flecha abajo
                        if noice>10: noice = noice - 10
                        else: noice = 0
                    elif key == b'K':  # Flecha izquierda
                        if noice>0: noice = noice - 1
                    elif key == b'M':  # Flecha derecha
                        if noice<100: noice = noice + 1
                elif key == b'\r':  # Enter
                    break
                elif key == b'\x1b':  # Escape
                    break
        curses.wrapper(wrapper)
    
    lvlNoice = noice/100

# función de cambiar la cantidad de iteraciones
def optionChangeIterations():
    global Cnt_Iteraciones
    if platform.system() == 'Windows':
        while True:
            clear_screen()
            print("Cuantas iteraciones máximas tendrá?")
            print("            " + str(Cnt_Iteraciones))
            print()
            print()
            print("flecha ← para reducir 10")
            print("flecha → para aumentar 10")
            print("flecha ↓ para reducir 100")
            print("flecha ↑ para aumentar 100")
            print("flecha Enter o Esc para salir")

            key = get_key_windows()
            if key == b'\xe0':  # Tecla especial
                key = msvcrt.getch()
                if key == b'H':  # Flecha arriba
                    if Cnt_Iteraciones<9900: Cnt_Iteraciones = Cnt_Iteraciones + 100
                elif key == b'P':  # Flecha abajo
                    if Cnt_Iteraciones>150: Cnt_Iteraciones = Cnt_Iteraciones - 100
                elif key == b'K':  # Flecha izquierda
                    if Cnt_Iteraciones>50: Cnt_Iteraciones = Cnt_Iteraciones - 10
                elif key == b'M':  # Flecha derecha
                    if Cnt_Iteraciones<10000: Cnt_Iteraciones = Cnt_Iteraciones + 10
            elif key == b'\r':  # Enter
                break
            elif key == b'\x1b':  # Escape
                break

    else:  # Para Linux o macOS
        def wrapper(stdscr):
            while True:
                clear_screen()
                print("Cuantas iteraciones máximas tendrá?")
                print("            " + str(Cnt_Iteraciones))
                print("flecha ← para reducir 10")
                print("flecha → para aumentar 10")
                print("flecha ↓ para reducir 100")
                print("flecha ↑ para aumentar 100")
                print("flecha Enter o Esc para salir")
                key = get_key_windows()
                if key == b'\xe0':  # Tecla especial
                    key = msvcrt.getch()
                    if key == b'H':  # Flecha arriba
                        if Cnt_Iteraciones<9900: Cnt_Iteraciones = Cnt_Iteraciones + 100
                    elif key == b'P':  # Flecha abajo
                        if Cnt_Iteraciones>150: Cnt_Iteraciones = Cnt_Iteraciones - 100
                    elif key == b'K':  # Flecha izquierda
                        if Cnt_Iteraciones>50: Cnt_Iteraciones = Cnt_Iteraciones - 10
                    elif key == b'M':  # Flecha derecha
                        if Cnt_Iteraciones<10000: Cnt_Iteraciones = Cnt_Iteraciones + 10
                elif key == b'\r':  # Enter
                    break
                elif key == b'\x1b':  # Escape
                    break
        curses.wrapper(wrapper)

# función para iniciar la ejecución de la red de Hopfield
def optionStartHopfield():
    startHopfield()

# función principal del menús
def main_menu():
    # Opciones del menú
    optionsMainMenu = ["Elegir Dataset", "Cambiar ruido agregado al input", "Cambiar cantidad de iteraciones máximas", "Ejecutar Hopfield", "Salir"]
    selected_index = 0
    global datasetSelected
    if platform.system() == 'Windows':
        while True:
            clear_screen()
            for i, option in enumerate(optionsMainMenu):
                prefix = Fore.BLACK + Back.WHITE + "-> " if i == selected_index else "   "
                print(f"{prefix}{option}")

            print()
            print(f"dataset usado: {Fore.GREEN}dataset{datasetSelected}")
            print(f"ruido agregado: {Fore.GREEN}{lvlNoice}")
            print(f"Cantidad de iteraciones: {Fore.GREEN}{Cnt_Iteraciones}")
            print()
            print()
            print("para controlar el menú ↓ y ↑")
            print('Pulse "Enter" para elejir')
            print('"Esc o salir" para salir del programa')
            # print("dataset_path: "+ dataset_path)
            # print("input_path: "+ input_path)

            key = get_key_windows()
            if key == b'\xe0':  # Tecla especial
                key = msvcrt.getch()
                if key == b'H':  # Flecha arriba
                    selected_index = (selected_index - 1) % len(optionsMainMenu)
                elif key == b'P':  # Flecha abajo
                    selected_index = (selected_index + 1) % len(optionsMainMenu)
            elif key == b'\r':  # Enter
                if selected_index == len(optionsMainMenu) - 1:  # Opción "Salir"
                    print("")
                    print(Fore.BLACK + Back.WHITE + "       Saliendo...")
                    break
                else:
                    if selected_index == 0:
                        optionSelectDataset()
                    elif selected_index == 1:
                        optionChangeNoice()
                    elif selected_index == 2:
                        optionChangeIterations()
                    elif selected_index == 3:
                        optionStartHopfield()  
            elif key == b'\x1b':  # Escape
                break

    else:  # Para Linux o macOS
        def wrapper(stdscr):
            nonlocal selected_index
            curses.curs_set(0)
            while True:
                clear_screen()
                for i, option in enumerate(optionsMainMenu):
                    prefix = Fore.BLACK + Back.WHITE + "-> " if i == selected_index else "   "
                    stdscr.addstr(i, 0, f"{prefix}{option}")

                print()
                print(f"dataset usado: {Fore.GREEN}dataset{datasetSelected}")
                print(f"ruido agregado: {Fore.GREEN}{lvlNoice}")
                print(f"Cantidad de iteraciones: {Fore.GREEN}{Cnt_Iteraciones}")
                print()
                print()
                print("para controlar el menú ↓ y ↑")
                print('Pulse "Enter" para elejir')
                print('"Esc o salir" para salir del programa')
                # print("dataset_path: "+dataset_path)
                # print("input_path: "+input_path)

                key = get_key_unix(stdscr)
                if key == curses.KEY_UP:  # Flecha arriba
                    selected_index = (selected_index - 1) % len(optionsMainMenu)
                elif key == curses.KEY_DOWN:  # Flecha abajo
                    selected_index = (selected_index + 1) % len(optionsMainMenu)
                elif key == curses.KEY_ENTER or key in [10, 13]:  # Enter
                    if selected_index == len(optionsMainMenu) - 1:  # Opción "Salir"
                        print("")
                        print(Fore.BLACK + Back.WHITE + "       Saliendo...")
                        break
                    else:
                        if selected_index == 0:
                            optionSelectDataset()
                        elif selected_index == 1:
                            optionChangeNoice()
                        elif selected_index == 2:
                            optionChangeIterations()
                        elif selected_index == 3:
                            optionStartHopfield()    
                elif key == 27:  # Escape
                    break
        curses.wrapper(wrapper)

print()
print()
print("     ██     ██    █████████    █████████    █████████    ███    █████████    ██          ████████")
print("     ██     ██    ██     ██    ██     ██    ██                  ██           ██          ██     ██")
print("     █████████    ██     ██    █████████    ██████       ███    █████████    ██          ██     ██")
print("     ██     ██    ██     ██    ██           ██           ███    ██           ██          ██     ██")
print("     ██     ██    █████████    ██           ██           ███    █████████    ████████    ████████")
print()
print()
print("by: Gabriel Nicolás Perero")
print()
input("                             --- pulsa enter para continuar ---")
main_menu()