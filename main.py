import numpy as np
import random
import time
import random
import matplotlib.pyplot as plt
import statistics

semillas = [123456, 832591, 938175, 201935, 134599]

numero_hormigas = 30
numero_hormigas_elitista = 15
evaporacion = 0.1

numero_iteraciones = 5

alpha = 1
beta = 2

semillas=[123456,758293,672341,835776,917325]

def parse_tsp_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    metadata = {}
    coordenadas = []

    for line in lines:
        if line.startswith("EOF"):
            break

        if ':' in line:
            key, value = line.strip().split(':')
            metadata[key.strip()] = value.strip()
        elif "NODE_COORD_SECTION" in line:
            continue
        else:
            dato = line.strip().split()
            coordenadas.append((float(dato[1]), float(dato[2])))

    return metadata, coordenadas


def distancia_euclidea(punto1, punto2):
    # return round(np.sqrt((punto1[0] - punto2[0]) ** 2 + (punto1[1] - punto2[1]) ** 2))
    punto1=np.array(punto1)
    punto2=np.array(punto2)
    return round(np.linalg.norm(punto1-punto2))

def matriz_distancia(coords):
    n = len(coords)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matriz[i][j] = matriz[j][i] = distancia_euclidea(coords[i], coords[j])
    return matriz


def evaluacion(camino, matriz_distancia):
    coste = 0
    for i in range(len(camino) - 1):
        coste += matriz_distancia[camino[i]][camino[i + 1]]
    return coste


def greedy(coordenadas, matriz_distancia):
    n = len(coordenadas)
    camino = [0]
    visitados = set([0])

    for _ in range(n - 1):
        nodo_actual = camino[-1]
        distancia_minima = float('inf')
        siguiente_nodo = None

        for i in range(n):
            if i not in visitados and matriz_distancia[nodo_actual][i] < distancia_minima:
                distancia_minima = matriz_distancia[nodo_actual][i]
                siguiente_nodo = i
        camino.append(siguiente_nodo)
        visitados.add(siguiente_nodo)
    camino.append(0)
    return camino


def matriz_feromonas(n, feromonaInicial):
    return np.full((n, n), feromonaInicial)


def regla_transicion(nodo_actual, nodos_no_visitados, matriz_feromonas, matriz_distancia, alpha, beta):
    probabilidades = []
    denominador = 0
    thao_dict={}
    eta_dict={}
    for nodo in nodos_no_visitados:
        thao=matriz_feromonas[nodo_actual][nodo] ** alpha
        eta=(1/(matriz_distancia[nodo_actual][nodo]+1e-10)) ** beta
        thao_dict[nodo] = thao
        eta_dict[nodo] = eta
        denominador += thao * eta

    denominador = denominador if denominador>0 else 1e-10
    for nodo in nodos_no_visitados:
        # thao = matriz_feromonas[nodo_actual][nodo] ** alpha
        # eta = (1/(matriz_distancia[nodo_actual][nodo]+1e-10)) ** beta
        prob=((thao_dict[nodo]*eta_dict[nodo])/(denominador))
        probabilidades.append(prob)
    nodos_no_visitadosList = list(nodos_no_visitados)
    return random.choices(nodos_no_visitadosList, weights=probabilidades,k=1)[0]



def actualizarFeromonas(matriz_feromonas, caminos, costes, evaporacion):
    matriz_feromonas*=(1-evaporacion)

    for indice,camino in enumerate(caminos):
        coste=costes[indice]
        aporte=1/coste
        for i in range(len(camino)-1):
            matriz_feromonas[camino[i]][camino[i+1]] += aporte
            matriz_feromonas[camino[i+1]][camino[i]] = matriz_feromonas[camino[i]][camino[i+1]]


def actualizarFeromonasElitista(matriz_feromonas, caminos, costes, evaporacion,costeMejor,e,caminoMejor):
    matriz_feromonas*=(1-evaporacion)

    for indice, camino in enumerate(caminos):
        coste = costes[indice]
        aporte=1/coste
        for i in range(len(camino)-1):
            matriz_feromonas[camino[i]][camino[i + 1]] += aporte
            matriz_feromonas[camino[i + 1]][camino[i]] = matriz_feromonas[camino[i]][camino[i + 1]]

    for i in range(len(caminoMejor)-1):
        matriz_feromonas[caminoMejor[i]][caminoMejor[i+1]]+=e*(1/costeMejor)
        matriz_feromonas[caminoMejor[i+1]][caminoMejor[i]]=matriz_feromonas[caminoMejor[i]][caminoMejor[i+1]]

def mostrar_camino(camino, node_coords,semilla,SH,CH):
    # Extraer las coordenadas x e y de los nodos
    x_coords = [coord[0] for coord in node_coords]
    y_coords = [coord[1] for coord in node_coords]

    # Dibujar los nodos como puntos en el gráfico
    plt.title(f"Camino obtenido con semilla {semilla}")
    plt.scatter(x_coords, y_coords, color='blue', zorder=1)

    # Dibujar el recorrido entre los nodos
    for i in range(len(camino) - 1):
        inicio = node_coords[camino[i]]
        fin = node_coords[camino[i + 1]]
        plt.plot([inicio[0], fin[0]], [inicio[1], fin[1]], color='red', zorder=0)

    # Conectar el último nodo al primer nodo para cerrar el ciclo
    inicio = node_coords[camino[-1]]
    fin = node_coords[camino[0]]
    plt.plot([inicio[0], fin[0]], [inicio[1], fin[1]], color='red', zorder=0)

    # Mostrar la ventana con el gráfico
    if semilla=="greedy":
        if CH:
            plt.savefig("Greedy CH130")
        else:
            plt.savefig("Greedy A280")

    else:
        if SH:
            if CH:
                plt.savefig(f"CH130 {semilla} con SH")
            else:
                plt.savefig(f"A280 {semilla}  con SH")
        else:
            if CH:
                plt.savefig(f"CH130 {semilla}  con SHE")
            else:
                plt.savefig(f"A280 {semilla}  con SHE")

    plt.close()
    plt.show()


def aumentarFeromonas(camino,matriz_feromonas,coste):
    aporte=1/coste
    for i in range(len(camino)-1):
        matriz_feromonas[camino[i]][camino[i+1]]=1/coste
        matriz_feromonas[camino[i+1]][camino[i]]=matriz_feromonas[camino[i]][camino[i+1]]

def SH(file_path, numero_hormigas, evaporacion, alpha, beta, semillas, numero_iteraciones,segundos):

    metadata, coordenadas = parse_tsp_file(file_path)
    matrizD = matriz_distancia(coordenadas)
    caminoGreedy = greedy(coordenadas, matrizD)
    costeGreedy = evaluacion(caminoGreedy, matrizD)
    tamanio_instancia = len(coordenadas)
    inicialFeromona = 1/(tamanio_instancia*costeGreedy)
    matrizFeromonas = matriz_feromonas(tamanio_instancia, inicialFeromona)
    caminos=[]
    caminos.append(caminoGreedy)
    costes=[]
    costes.append(costeGreedy)

    print("COSTE DEL CAMINO GREEDY: ",costeGreedy)
    if file_path=="./Datos/ch130.tsp":
        mostrar_camino(caminoGreedy, coordenadas,"greedy",True,True)
    else:
        mostrar_camino(caminoGreedy, coordenadas, "greedy", True, False)
    Costes=[]
    Evaluaciones=[]
    Caminos=[]
    for i in range(numero_iteraciones):
        print("ITERACION: ",i)
        print("SEMILLA: ", semillas[i])
        matrizFeromonas = matriz_feromonas(tamanio_instancia, inicialFeromona)
        random.seed(semillas[i])
        camino_mejor = caminoGreedy
        coste_camino_mejor = costeGreedy
        evaluaciones=0
        tiempo_inicio = tiempo_final = time.time()
        while tiempo_final - tiempo_inicio < segundos:
            caminos = []
            costes = []
            costeMejorTodas = float('inf')
            caminoMejorTodas = None
            for _ in range(numero_hormigas):

                #Opcion aleatorio
                indice_inicio=random.randint(0,tamanio_instancia-1)
                nodos_noVisitados = set(range(0, tamanio_instancia))
                camino=[indice_inicio]
                nodos_noVisitados.remove(indice_inicio)

                #Opcion
                # camino = [0]  # Quizas aleatorio
                # nodos_noVisitados = set(range(1, tamanio_instancia))
                for _ in range(tamanio_instancia - 1):
                    nodoActual = camino[-1]
                    siguiente_Nodo = regla_transicion(nodoActual, nodos_noVisitados, matrizFeromonas, matrizD, alpha,
                                                      beta)
                    camino.append(siguiente_Nodo)
                    nodos_noVisitados.remove(siguiente_Nodo)

                camino.append(indice_inicio)
                # camino.append(0)
                coste = evaluacion(camino, matrizD)
                evaluaciones+=1
                costes.append(coste)
                caminos.append(camino)
                if coste < costeMejorTodas:
                    costeMejorTodas = coste
                    caminoMejorTodas = camino
            if costeMejorTodas < coste_camino_mejor:
                coste_camino_mejor = costeMejorTodas
                print("MEJOR CAMINO: ", coste_camino_mejor)
                camino_mejor = caminoMejorTodas
            # Actualizar Feromona
            actualizarFeromonas(matrizFeromonas, caminos, costes, evaporacion)
            tiempo_final = time.time()
        print(f"CAMINO OBTENIDO EN ITERACION {i}: ",camino_mejor)
        Caminos.append(camino_mejor)
        if file_path=="./Datos/ch130.tsp":
            mostrar_camino(camino_mejor,coordenadas,semillas[i],True,True)
        else:
            mostrar_camino(camino_mejor, coordenadas, semillas[i], True, False)
        print(f"Su coste es de: ",coste_camino_mejor)
        print(f"Evaluaciones: ",evaluaciones)
        Costes.append(coste_camino_mejor)
        Evaluaciones.append(evaluaciones)

    return Caminos,Costes,Evaluaciones

def SHE(file_path, numero_hormigas, evaporacion, alpha, beta, semillas, numero_iteraciones,e,segundos):

    metadata, coordenadas = parse_tsp_file(file_path)
    matrizD = matriz_distancia(coordenadas)
    caminoGreedy = greedy(coordenadas, matrizD)
    costeGreedy = evaluacion(caminoGreedy, matrizD)
    tamanio_instancia = len(coordenadas)
    inicialFeromona = 1/(tamanio_instancia*costeGreedy)
    matrizFeromonas = matriz_feromonas(tamanio_instancia, inicialFeromona)
    caminos=[]
    caminos.append(caminoGreedy)
    costes=[]
    costes.append(costeGreedy)
    # for _ in range(20):
    #     actualizarFeromonas(matrizFeromonas,caminos,costes,evaporacion)


    print("COSTE DEL CAMINO GREEDY: ",costeGreedy)
    if file_path == "./Datos/ch130.tsp":
        mostrar_camino(caminoGreedy, coordenadas, "greedy", True, True)
    else:
        mostrar_camino(caminoGreedy, coordenadas, "greedy", True, False)
    Costes=[]
    Evaluaciones=[]
    Caminos=[]
    for i in range(numero_iteraciones):
        print("ITERACION: ",i)
        print("SEMILLA: ",semillas[i])
        matrizFeromonas = matriz_feromonas(tamanio_instancia, inicialFeromona)
        random.seed(semillas[i])
        camino_mejor = caminoGreedy
        coste_camino_mejor = costeGreedy
        evaluaciones=0
        tiempo_inicio = tiempo_final = time.time()
        while tiempo_final - tiempo_inicio < segundos:
            caminos = []
            costes = []
            costeMejorTodas = float('inf')
            caminoMejorTodas = None
            for _ in range(numero_hormigas):
                # Opcion aleatorio
                indice_inicio = random.randint(0, tamanio_instancia - 1)
                nodos_noVisitados = set(range(0, tamanio_instancia))
                camino = [indice_inicio]
                nodos_noVisitados.remove(indice_inicio)

                # Opcion
                # camino = [0]  # Quizas aleatorio
                # nodos_noVisitados = set(range(1, tamanio_instancia))

                for _ in range(tamanio_instancia - 1):
                    nodoActual = camino[-1]
                    siguiente_Nodo = regla_transicion(nodoActual, nodos_noVisitados, matrizFeromonas, matrizD, alpha,
                                                      beta)
                    camino.append(siguiente_Nodo)
                    nodos_noVisitados.remove(siguiente_Nodo)
                camino.append(indice_inicio)
                # camino.append(0)
                coste = evaluacion(camino, matrizD)
                evaluaciones+=1
                costes.append(coste)
                caminos.append(camino)
                if coste < costeMejorTodas:
                    costeMejorTodas = coste
                    caminoMejorTodas = camino
            if costeMejorTodas < coste_camino_mejor:
                coste_camino_mejor = costeMejorTodas
                print("MEJOR CAMINO: ", coste_camino_mejor)
                camino_mejor = caminoMejorTodas
            # Actualizar Feromona
            actualizarFeromonasElitista(matrizFeromonas, caminos, costes, evaporacion,coste_camino_mejor,e,camino_mejor)

            tiempo_final = time.time()
        print(f"CAMINO OBTENIDO EN ITERACION {i}: ",camino_mejor)
        Caminos.append(camino_mejor)
        if file_path=="./Datos/ch130.tsp":
            mostrar_camino(camino_mejor,coordenadas,semillas[i],False,True)
        else:
            mostrar_camino(camino_mejor, coordenadas, semillas[i], False, False)
        print(f"Su coste es de: ",coste_camino_mejor)
        print(f"Evaluaciones: ", evaluaciones)
        Costes.append(coste_camino_mejor)
        Evaluaciones.append(evaluaciones)

    return Caminos,Costes,Evaluaciones

print("SH CH130")
Caminos,Costes,Evaluaciones = SH("./Datos/ch130.tsp", numero_hormigas, evaporacion, alpha, beta, semillas,
                                      numero_iteraciones,180)

print("Coste mínimo: ",min(Costes))
print("Camino con coste minimo: ",Caminos[Costes.index(min(Costes))])
print("Media Costes: ",statistics.mean(Costes))
print("Desv. Tipica Costes: ",statistics.stdev(Costes))
print("Media Evaluaciones: ",statistics.mean(Evaluaciones))
print("Desv. Tipica Evaluaciones: ",statistics.stdev(Evaluaciones))

print("SH A280")
Caminos,Costes,Evaluaciones = SH("./Datos/a280.tsp", numero_hormigas, evaporacion, alpha, beta, semillas,
                                      numero_iteraciones,480)
print("Coste mínimo: ",min(Costes))
print("Camino con coste minimo: ",Caminos[Costes.index(min(Costes))])
print("Media Costes: ",statistics.mean(Costes))
print("Desv. Tipica Costes: ",statistics.stdev(Costes))
print("Media Evaluaciones: ",statistics.mean(Evaluaciones))
print("Desv. Tipica Evaluaciones: ",statistics.stdev(Evaluaciones))

print("SHE CH130")
Caminos,Costes,Evaluaciones = SHE("./Datos/ch130.tsp", numero_hormigas, evaporacion, alpha, beta, semillas,
                                      numero_iteraciones,numero_hormigas_elitista,180)
print("Coste mínimo: ",min(Costes))
print("Camino con coste minimo: ",Caminos[Costes.index(min(Costes))])
print("Media Costes: ",statistics.mean(Costes))
print("Desv. Tipica Costes: ",statistics.stdev(Costes))
print("Media Evaluaciones: ",statistics.mean(Evaluaciones))
print("Desv. Tipica Evaluaciones: ",statistics.stdev(Evaluaciones))

print("SHE A280")
Caminos,Costes,Evaluaciones = SHE("./Datos/a280.tsp", numero_hormigas, evaporacion, alpha, beta, semillas,
                                      numero_iteraciones,numero_hormigas_elitista,480)
print("Coste mínimo: ",min(Costes))
print("Camino con coste minimo: ",Caminos[Costes.index(min(Costes))])
print("Media Costes: ",statistics.mean(Costes))
print("Desv. Tipica Costes: ",statistics.stdev(Costes))
print("Media Evaluaciones: ",statistics.mean(Evaluaciones))
print("Desv. Tipica Evaluaciones: ",statistics.stdev(Evaluaciones))