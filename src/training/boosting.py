# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
#
# Se importan las librerías necesarias para:
#
# • Manipulación del sistema de archivos.
# • Almacenamiento del modelo entrenado.
# • Procesamiento de datos.
# • Entrenamiento del modelo Gradient Boosting.
# • División del conjunto de datos.
# • Codificación de variables categóricas.
#

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# IMPORTACIÓN DE LA CONFIGURACIÓN DEL PROYECTO
# ==========================================================
#
# Se importan las constantes utilizadas durante el proceso
# de entrenamiento.
#
# FEATURES
#     Variables predictoras del modelo.
#
# TARGET
#     Variable objetivo.
#
# TEST_SIZE
#     Porcentaje destinado al conjunto de prueba.
#
# RANDOM_STATE
#     Semilla para garantizar resultados reproducibles.
#
# MODEL_PATH
#     Ruta donde se almacenará el modelo entrenado.
#
# ENCODERS_PATH
#     Ruta donde se guardarán los LabelEncoder utilizados
#     durante el preprocesamiento.
#

from src.config.settings import (
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    ENCODERS_PATH
)


# ==========================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ==========================================================
#
# Esta función realiza el proceso completo de entrenamiento
# del modelo Gradient Boosting.
#
# El procedimiento consiste en:
#
# 1. Leer los datos desde la base de datos.
# 2. Eliminar registros inválidos.
# 3. Realizar Feature Engineering.
# 4. Preparar las variables predictoras.
# 5. Codificar variables categóricas.
# 6. Dividir los datos en entrenamiento y prueba.
# 7. Entrenar el modelo Gradient Boosting.
# 8. Guardar el modelo entrenado.
# 9. Guardar los LabelEncoder.
#

def train_model(conn):
    """
    Entrena el modelo de Machine Learning utilizando Gradient Boosting.

    Parámetros
    ----------
    conn : psycopg2.connection
        Conexión a la base de datos.

    Retorna
    -------
    model : GradientBoostingClassifier
    X_test : DataFrame
    y_test : Series
    """

    # ======================================================
    # LECTURA DE LOS DATOS
    # ======================================================
    #
    # Se recupera el conjunto de datos preparado durante el
    # proceso ETL, el cual contiene todas las variables
    # necesarias para el entrenamiento.
    #

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    # ======================================================
    # ELIMINACIÓN DE REGISTROS INVÁLIDOS
    # ======================================================
    #
    # Se eliminan los registros cuyo valor de la variable
    # objetivo es NULL.
    #
    # Estos registros corresponden principalmente a las
    # últimas semanas de cada lote, donde aún no existe
    # suficiente información futura para construir el target.
    #

    filas_antes = len(df)

    df = df[df[TARGET].notna()].copy()

    print(
        f"ℹ️ Filas excluidas por target nulo: {filas_antes - len(df)}"
    )

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================
    #
    # La variable semana_anio posee un comportamiento cíclico.
    #
    # Para representar correctamente esta característica se
    # generan dos nuevas variables mediante funciones seno
    # y coseno.
    #
    # Esto evita que el modelo interprete erróneamente que
    # la semana 1 está muy alejada de la semana 52.
    #

    df["semana_sin"] = np.sin(
        2 * np.pi * df["semana_anio"] / 52
    )

    df["semana_cos"] = np.cos(
        2 * np.pi * df["semana_anio"] / 52
    )

    # ======================================================
    # SEPARACIÓN DE VARIABLES
    # ======================================================
    #
    # X contiene todas las variables predictoras.
    #
    # y contiene la variable objetivo que deberá aprender
    # el modelo.
    #

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    # ======================================================
    # CODIFICACIÓN DE VARIABLES CATEGÓRICAS
    # ======================================================
    #
    # Gradient Boosting trabaja únicamente con variables
    # numéricas.
    #
    # Todas las columnas categóricas se transforman mediante
    # LabelEncoder.
    #
    # Cada encoder se almacena para reutilizar exactamente la
    # misma codificación durante futuras predicciones.
    #

    encoders = {}

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        # Crear encoder para la columna actual.
        le = LabelEncoder()

        # Transformar categorías en valores numéricos.
        X[col] = le.fit_transform(
            X[col].astype(str)
        )

        # Guardar encoder.
        encoders[col] = le

    # ======================================================
    # DIVISIÓN DEL CONJUNTO DE DATOS
    # ======================================================
    #
    # Se divide el conjunto de datos en entrenamiento y
    # prueba.
    #
    # Se utiliza estratificación para mantener la misma
    # proporción de clases en ambos conjuntos.
    #

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ======================================================
    # CREACIÓN DEL MODELO
    # ======================================================
    #
    # Se configura el algoritmo Gradient Boosting utilizando
    # los hiperparámetros definidos para este proyecto.
    #
    # Características principales:
    #
    # • 300 árboles secuenciales.
    # • Learning Rate reducido para mejorar la estabilidad.
    # • Profundidad máxima limitada para reducir sobreajuste.
    # • Restricciones sobre el tamaño mínimo de nodos y hojas.
    #

    model = GradientBoostingClassifier(

        # Número de árboles.
        n_estimators=300,

        # Velocidad de aprendizaje.
        learning_rate=0.05,

        # Profundidad máxima de cada árbol.
        max_depth=4,

        # Muestras mínimas para dividir un nodo.
        min_samples_split=5,

        # Muestras mínimas por hoja.
        min_samples_leaf=5,

        # Semilla para garantizar reproducibilidad.
        random_state=RANDOM_STATE

    )

    # ======================================================
    # ENTRENAMIENTO DEL MODELO
    # ======================================================
    #
    # El algoritmo construye los árboles de forma secuencial,
    # donde cada nuevo árbol intenta corregir los errores
    # cometidos por los anteriores.
    #

    model.fit(X_train, y_train)

    # ======================================================
    # CREACIÓN DE LA CARPETA DE MODELOS
    # ======================================================
    #
    # Si la carpeta "models" no existe, se crea
    # automáticamente.
    #

    os.makedirs("models", exist_ok=True)

    # ======================================================
    # GUARDADO DEL MODELO
    # ======================================================
    #
    # El modelo entrenado se almacena utilizando pickle para
    # poder reutilizarlo posteriormente sin necesidad de
    # volver a entrenarlo.
    #

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # ======================================================
    # GUARDADO DE LOS LABEL ENCODERS
    # ======================================================
    #
    # También se almacenan los encoders utilizados durante el
    # entrenamiento para garantizar que futuras predicciones
    # empleen exactamente la misma codificación.
    #

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    # ======================================================
    # MENSAJE DE CONFIRMACIÓN
    # ======================================================

    print("✅ Modelo Gradient Boosting entrenado correctamente.")

    # ======================================================
    # RETORNO DE LA FUNCIÓN
    # ======================================================
    #
    # Se devuelve:
    #
    # • El modelo entrenado.
    # • El conjunto de prueba.
    # • Las etiquetas reales del conjunto de prueba.
    #
    # Estos elementos serán utilizados posteriormente para
    # evaluar el rendimiento del modelo.
    #

    return model, X_test, y_test