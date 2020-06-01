###################################################################################################
# -*- coding: utf-8 -*-
# Autor: Ángel Valera Motos
###################################################################################################
import sys
import os.path
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

# Función para conectar con el cluster de Spark
#-------------------------------------------------------------------------------
def iniciar_Spark():
    """ Inicializa el Spark Context """
    conf = SparkConf().setAppName("Practica 4 - Angel Valera Motos")
    sc = SparkContext(conf=conf) 
    sc.setLogLevel('WARN')
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
        '/user/ccsa14274858/filteredC.small.training', header=True, mode="overwrite")
    return new_DF 
    
# Función para realizar el prepocesamiento de los datos
#-------------------------------------------------------------------------------
def prepocesamiento(dataFrame):    
    # Converstimos el conjunto de datos a un formato legible
    assembler = VectorAssembler(inputCols=['PSSM_r2_-1_M', 'PSSM_r1_3_K', 'AA_freq_global_D',
                                           'PSSM_central_0_V', 'AA_freq_central_D', 'PSSM_r1_-1_A'], outputCol='features')
    dataFrame = assembler.transform(dataFrame)
    dataFrame = dataFrame.selectExpr('features as features', 'class as label')
    dataFrame = dataFrame.select('features', 'label')
    # Balanceamos de los datos utilizando Undersampling    
    No = dataFrame.filter('label=0')
    Si = dataFrame.filter('label=1')
    sampleRatio = float(Si.count()) / float(dataFrame.count())
    seleccion = No.sample(False, sampleRatio)
    dataFrame = Si.unionAll(seleccion)
    return dataFrame

# Función para clasificar usando RandomForest 
#-------------------------------------------------------------------------------
def clasificador_RandomForest(dataFrame,numArboles):
    # Ajustamos todo el conjunto de datos para incluir todas las etiquetas en el índice.
    labelIndexer = StringIndexer(
        inputCol='label', outputCol='indexedLabel').fit(dataFrame)
    # Se identifican automáticamente las características categóricas e indexelas.
    # Establecemos el maxCategories para que las características con> 4 valores distintos se traten como continuas.
    featureIndexer =\
        VectorIndexer(inputCol='features',
                      outputCol='indexedFeatures', maxCategories=4).fit(dataFrame)
    # Dividimos los datos en conjuntos de entrenamiento y prueba (30% retenido para la prueba)
    (trainingData, testData) = dataFrame.randomSplit([0.7, 0.3])
    # Entrena un modelo RandomForest.
    rf = RandomForestClassifier(
        labelCol='indexedLabel', featuresCol='indexedFeatures', numTrees=numArboles)
    # Convertir etiquetas indexadas de nuevo a etiquetas originales.
    labelConverter = IndexToString(inputCol='prediction', outputCol='predictedLabel',
                                   labels=labelIndexer.labels)    
    pipeline = Pipeline(
        stages=[labelIndexer, featureIndexer, rf, labelConverter])
    # Modelo de entrenamiento
    model = pipeline.fit(trainingData)
    # Hacer predicciones
    predictions = model.transform(testData)
    # Seleccione filas de ejemplo para mostrar.
    predictions.select('predictedLabel', 'label', 'features').show(5)
    # Seleccionamos (prediction, true label) y calculamos el test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol='indexedLabel', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print('Test Error = %g' % (1.0 - accuracy))
    print('Accuracy = ', accuracy)

    rfModel = model.stages[2]
    print(rfModel)  # summary only

    #Calcular AUC
    evaluator = BinaryClassificationEvaluator()
    evaluation = evaluator.evaluate(model.transform(testData))
    print('AUC:', evaluation)
    
    

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    # Conectamos con Spark    
    sc = iniciar_Spark()    
    # Comprobamos si existe el dataframe y si no existe lo generamos    
    fichero = '/user/ccsa14274858/filteredC.small.training'
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    if fs.exists(sc._jvm.org.apache.hadoop.fs.Path(fichero)):
        sqlc = SQLContext(sc)
        df = sqlc.read.csv(fichero, header=True, inferSchema=True)
    else:       
        # Unimos cabecera y datos en un conjunto de datos 
        df = cargarDatos(sc)
        # Obtenemos un nuevo conjunto de datos con las columnas designadas
        df = generarNuevoDF(df)
    # Realizamos un prepocesamiento a los datos y los balanceamos con undersampling
    df = prepocesamiento(df)    
    # Realizamos una clasificación usando RandomForest     
    clasificador_RandomForest(df, 10)
    #clasificador_RandomForest(df, 75)
    #clasificador_RandomForest(df, 150)


    sc.stop()
