###################################################################################################
# -*- coding: utf-8 -*-
# Autor: Ángel Valera Motos
###################################################################################################
import sys
from pyspark import SparkContext, SparkConf, SQLContext
import os.path

# Función para conectar con el cluster de Spark
#-------------------------------------------------------------------------------
def iniciar_Spark():
    """ Inicializa el Spark Context """
    conf = SparkConf().setAppName("Practica 4 - Angel Valera Motos")
    sc = SparkContext(conf=conf)    
    return sc

# Función para cargar el fichero de datos y asignarle la cabecera
#-------------------------------------------------------------------------------
def cargarDatos(sparkContext):
    # Leemos el fichero con las cabeceras
    cabeceras = sparkContext.textFile(
        "/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    # Extraemos las cabeceras
    cabeceras = list(filter(lambda x: "@inputs" in x, cabeceras))[0]
    # Limpiamos las cebeceras 
    cabeceras = cabeceras.replace(",", "").strip().split()
    # Eliminamos la primera columna que contiene "@input"
    del cabeceras[0]                             
    # Añadimos la columna "class"     
    cabeceras.append("class")       
    # Extraemos los datos
    sqlc = SQLContext(sparkContext)
    dataFrame = sqlc.read.csv(
        "/user/datasets/ecbdl14/ECBDL14_IR2.data", header=False, inferSchema=True)
    # Asignamos la cabecera su columna correspondiente    
    for i, nombreColumna in enumerate(dataFrame.columns):
        dataFrame = dataFrame.withColumnRenamed(nombreColumna, cabeceras[i])
    return dataFrame

# Función para generar un nuevo conjunto de datos con las columnas asignadas
#-------------------------------------------------------------------------------
def generarNuevoDF(dataFrame):
    # Columnas asignadas
    columnasAsignadas = ["PSSM_r2_-1_M", "PSSM_r1_3_K", "AA_freq_global_D",
                         "PSSM_central_0_V", "AA_freq_central_D", "PSSM_r1_-1_A", "class"]
    # Seleccionamos las columnas asignadas
    new_DF = dataFrame.select(columnasAsignadas)
    # Generamos un nuevo csv
    new_DF.write.csv(
        '/user/ccsa14274858/filteredC.small.training', header=True)

# Función para realizar el prepocesamiento de los datos
#-------------------------------------------------------------------------------
def prepocesamiento(dataFrame):    
    print("hello")

# Función para comprobar si el dataframe existe
#-------------------------------------------------------------------------------
def existe_DataFrame(sparkContext, fichero):
    sqlc = SQLContext(sparkContext)
    if (os.path.exists(fichero) and os.path.isfile(fichero)):
        df = sqlc.read.csv(fichero, header=True, sep=",", inferSchema=True)
        print('El fichero ya existe')
    else:
        # Generamos el nuevo dataframe
        print('El fichero no existe. Se crea de nuevo')
        df = cargarDatos(sc)        
        df = generarNuevoDF(df)
    return df


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    # Conectamos con Spark    
    sc = iniciar_Spark()    
    # Comprobamos si existe el dataframe y si no existe lo generamos
    fichero = '/user/ccsa14274858/filteredC.small.training'
    df = existe_DataFrame(sc, fichero)

    sc.stop()
