###################################################################################################
# -*- coding: utf-8 -*-
# Autor: Ángel Valera Motos
###################################################################################################
import sys

from pyspark import SparkContext, SparkConf

# Función para conectar con el cluster de Spark
#-------------------------------------------------------------------------------
def iniciar_Spark():
    """ Inicializa el Spark Context """
    conf = SparkConf().setAppName("Practica 4 - Ángel Valera Motos")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    return sc

# Función para cargar el fichero de datos y seleccionar solo las 6 columnas designadas
#-------------------------------------------------------------------------------
def seleccionarColumnas(sparkContext):
    # Primero debemos leer el fichero con las cabeceras
    cabeceras = sparkContext.textFile(
        "/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    cabeceras = list(filter(lambda x: "@inputs" in x, cabeceras))[0]
    cabeceras = cabeceras.replace(",", "").strip().split()  # Eliminamos el
    del cabeceras[0]                                      # Borrar "@input"
    cabeceras.append("class")
    


if __name__ == '__main__':
    sc = iniciar_Spark()
    
