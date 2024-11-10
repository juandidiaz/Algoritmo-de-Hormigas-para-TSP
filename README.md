# Algoritmo de Hormigas para el Problema del Viajante de Comercio (TSP)

Este proyecto implementa el **Algoritmo de Colonia de Hormigas (Ant Colony Optimization, ACO)** para resolver el **Problema del Viajante de Comercio (TSP)**, optimizando el camino más corto para visitar una serie de ciudades y regresar al punto de partida. El código usa el enfoque de feromonas para guiar a las hormigas en la construcción de caminos de manera estocástica, con dos variantes: SH y SHE, las cuales utilizan un enfoque de selección y evaporación de feromonas.

## Descripción del Código

El código está escrito en Python y se compone de las siguientes partes principales:

- **Lectura de datos del archivo TSP:** Usa `parse_tsp_file` para cargar las coordenadas de las ciudades desde archivos `.tsp`.
- **Cálculo de distancias:** `distancia_euclidea` y `matriz_distancia` crean una matriz de distancias usando la distancia euclidiana.
- **Inicialización de feromonas:** `matriz_feromonas` crea una matriz de feromonas inicializadas.
- **Algoritmo Greedy:** Usado para obtener una solución inicial rápida.
- **Regla de Transición:** Las hormigas eligen el siguiente nodo basándose en la feromona y la distancia entre ciudades.
- **Actualización de Feromonas:** Dos variantes de actualización: SH (sin elitismo) y SHE (con elitismo).
- **Visualización de los caminos obtenidos:** `mostrar_camino` grafica el recorrido de cada hormiga.

## Parámetros Principales

- `numero_hormigas`: Número total de hormigas.
- `numero_hormigas_elitista`: Número de hormigas elitistas en SHE.
- `evaporacion`: Tasa de evaporación de feromonas.
- `alpha` y `beta`: Parámetros de importancia de la feromona y la visibilidad (distancia) en la regla de transición.
- `numero_iteraciones`: Número de iteraciones del algoritmo.
- `semillas`: Lista de semillas aleatorias para experimentar con distintos caminos iniciales.

## Ejecución del Algoritmo

### Variantes SH y SHE

Ambas variantes se ejecutan en dos instancias de datos, `CH130.tsp` y `A280.tsp`. Las hormigas construyen sus caminos basándose en la probabilidad de transición, y después de cada iteración, las feromonas en los caminos de menor costo se refuerzan mientras las demás se evaporan.

1. **SH** - Algoritmo sin elitismo:
   ```python
   Caminos, Costes, Evaluaciones = SH("./Datos/ch130.tsp", numero_hormigas, evaporacion, alpha, beta, semillas, numero_iteraciones, 180)

   ## Resultados de las Simulaciones

## Resultados de las Simulaciones

## Resultados de las Simulaciones

### Caminos obtenidos en las diferentes instancias y configuraciones

![A280 123456 con SH](imagenes/A280%20123456%20con%20SH.png)
![A280 123456 con SHE](imagenes/A280%20123456%20con%20SHE.png)
![A280 672341 con SH](imagenes/A280%20672341%20con%20SH.png)
![A280 672341 con SHE](imagenes/A280%20672341%20con%20SHE.png)
![A280 758293 con SH](imagenes/A280%20758293%20con%20SH.png)
![A280 758293 con SHE](imagenes/A280%20758293%20con%20SHE.png)
![A280 835776 con SH](imagenes/A280%20835776%20con%20SH.png)
![A280 835776 con SHE](imagenes/A280%20835776%20con%20SHE.png)
![A280 917325 con SH](imagenes/A280%20917325%20con%20SH.png)
![A280 917325 con SHE](imagenes/A280%20917325%20con%20SHE.png)
![CH130 123456 con SHE](imagenes/CH130%20123456%20con%20SHE.png)
![CH130 123456 con SH](imagenes/CH130%20123456%20con%20SH.png)
![CH130 672341 con SHE](imagenes/CH130%20672341%20con%20SHE.png)
![CH130 672341 con SH](imagenes/CH130%20672341%20con%20SH.png)
![CH130 758293 con SHE](imagenes/CH130%20758293%20con%20SHE.png)
![CH130 758293 con SH](imagenes/CH130%20758293%20con%20SH.png)
![CH130 835776 con SHE](imagenes/CH130%20835776%20con%20SHE.png)
![CH130 835776 con SH](imagenes/CH130%20835776%20con%20SH.png)
![CH130 917325 con SHE](imagenes/CH130%20917325%20con%20SHE.png)
![CH130 917325 con SH](imagenes/CH130%20917325%20con%20SH.png)
![Greedy A280](imagenes/Greedy%20A280.png)
![Greedy CH130](imagenes/Greedy%20CH130.png)


