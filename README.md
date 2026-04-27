# Algoritmo Genético para el Problema de las N-Reinas

Este proyecto implementa un algoritmo genético para resolver el problema de las N-Reinas, 
utilizando una representación basada en permutaciones y distintos operadores genéticos para 
optimizar la búsqueda de soluciones sin conflictos.

## Ejecución
Para evitar errores relacionados con la codificación de caracteres (acentos, símbolos, etc.),
es necesario configurar la terminal antes de ejecutar el programa:

```chcp 65001```

Esto cambia la codificación de la consola a UTF-8, permitiendo que el programa se ejecute 
correctamente sin conflictos de lectura o impresión de caracteres.

## Descripción de archivos
### __pycache__/
Esta es una carpeta generada automaticamente por Python que almacena los archivos compilados para
poder mejorar el rendimiento.

### k_queens_genetic.py
Contiene la implementación principal del algoritmo genético, incluyendo la representación del 
problema, operadores de selección, cruzamiento, mutación y lógica de evolución.

### k_queens_viz.py
Módulo encargado de la visualización de resultados, como la generación de gráficos comparativos y 
la representación de los tableros con las soluciones encontradas.

### runner.py
Archivo principal de ejecución. Se encarga de configurar los experimentos, ejecutar ambas versiones del 
algoritmo (V1 y V2) y coordinar la generación de resultados y visualizaciones.

## Resultados
Todos los resultados generados por el programa (gráficas comparativas y tableros) se guardan automáticamente en el siguiente directorio:

> /mnt/k_queens_ga

Este directorio se crea dentro del entorno del usuario y permite centralizar toda la información generada durante la ejecución,
facilitando su análisis posterior.
