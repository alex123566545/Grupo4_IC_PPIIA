# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
#
# Se importan las librerías necesarias para:
#
# • Manejo del sistema de archivos.
# • Guardado del modelo entrenado.
# • Procesamiento de datos.
# • Entrenamiento del algoritmo Random Forest.
# • División del conjunto de datos.
# • Codificación de variables categóricas.
#

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#from preprocessing.feature_engineering import feature_engineering

# ==========================================================
# IMPORTACIÓN DE LA CONFIGURACIÓN DEL PROYECTO
# ==========================================================
#
# Estas constantes centralizan la configuración utilizada
# durante el entrenamiento del modelo.
#
# FEATURES
#     Variables predictoras.
#
# TARGET
#     Variable objetivo.
#
# TEST_SIZE
#     Porcentaje destinado al conjunto de prueba.
#
# RANDOM_STATE
#     Semilla para obtener resultados reproducibles.
#
# MODEL_PATH
#     Ruta donde se guardará el modelo entrenado.
#
# ENCODERS_PATH
#     Ruta donde se almacenarán los LabelEncoder.
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
# Esta función realiza todo el proceso necesario para entrenar
# el modelo de Machine Learning:
#
# 1. Leer los datos desde la base de datos.
# 2. Eliminar registros inválidos.
# 3. Crear nuevas variables (Feature Engineering).
# 4. Preparar las variables predictoras.
# 5. Codificar variables categóricas.
# 6. Dividir los datos en entrenamiento y prueba.
# 7. Entrenar el Random Forest.
# 8. Guardar el modelo.
# 9. Guardar los encoders.
#

def train_model(conn):
    """
    Entrena el modelo de Machine Learning.

    Versión simple (sin RandomizedSearchCV), elegida como modelo final
    tras comparar contra la versión optimizada con RandomizedSearchCV +
    F-beta(0.5): esta versión simple generaliza mejor bajo GroupKFold
    (validación por lote nunca visto), aunque su métrica en split
    aleatorio sea algo menor. Ver bitácora de cambios metodológicos
    para el detalle de esa comparación.

    Parámetros
    ----------
    conn : psycopg2.connection
        Conexión a la base de datos.

    Retorna
    -------
    model : RandomForestClassifier
    X_test : DataFrame
    y_test : Series
    """

    # ======================================================
    # LECTURA DE LOS DATOS
    # ======================================================
    #
    # Se obtiene el conjunto final de variables previamente
    # construido durante el proceso ETL.
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
    # Se eliminan las filas cuyo target es NULL.
    #
    # Esto ocurre principalmente en las últimas semanas de cada
    # lote, donde aún no existen suficientes semanas futuras
    # para determinar correctamente la variable objetivo.
    #

    filas_antes = len(df)

    df = df[df[TARGET].notna()].copy()

    print(f"ℹ️  Filas excluidas por target nulo (sin futuro suficiente): {filas_antes - len(df)}")

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================
    #
    # Se realiza una transformación cíclica de la variable
    # semana_anio.
    #
    # Debido a que las semanas tienen naturaleza circular,
    # la semana 52 y la semana 1 deberían encontrarse muy
    # próximas entre sí.
    #
    # Para representar correctamente esta relación se generan
    # dos nuevas variables utilizando funciones seno y coseno.
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
    # y contiene la variable objetivo que el modelo deberá
    # aprender a predecir.
    #

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    # ======================================================
    # CODIFICACIÓN DE VARIABLES CATEGÓRICAS
    # ======================================================
    #
    # Random Forest únicamente acepta variables numéricas.
    #
    # Todas las variables de tipo texto se convierten en
    # valores enteros mediante LabelEncoder.
    #
    # Cada encoder se almacena para reutilizar exactamente la
    # misma codificación durante las predicciones futuras.
    #

    encoders = {}

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        # Crear encoder para la columna actual.
        le = LabelEncoder()

        # Transformar categorías en números.
        X[col] = le.fit_transform(
            X[col].astype(str)
        )

        # Guardar encoder.
        encoders[col] = le

    # ======================================================
    # DIVISIÓN DEL CONJUNTO DE DATOS
    # ======================================================
    #
    # Se divide el conjunto de datos en entrenamiento y prueba.
    #
    # Se utiliza estratificación para mantener la misma
    # proporción de clases en ambos conjuntos.
    #
    # Esto evita que alguno de ellos tenga demasiados o muy
    # pocos ejemplos positivos.
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
    # Se construye el modelo Random Forest utilizando los
    # hiperparámetros seleccionados durante la fase de
    # experimentación del proyecto.
    #
    # Características principales:
    #
    # • 300 árboles.
    # • Profundidad máxima limitada.
    # • Pesos balanceados entre clases.
    # • Uso de todos los núcleos del procesador.
    #

    model = RandomForestClassifier(

        # Número de árboles.
        n_estimators=300,

        # Profundidad máxima.
        max_depth=8,

        # Muestras mínimas para dividir un nodo.
        min_samples_split=5,

        # Muestras mínimas por hoja.
        min_samples_leaf=5,

        # Balanceo automático entre clases.
        class_weight="balanced",

        # Semilla aleatoria.
        random_state=RANDOM_STATE,

        # Utilizar todos los núcleos disponibles.
        n_jobs=-1

    )

    # ======================================================
    # ENTRENAMIENTO DEL MODELO
    # ======================================================
    #
    # El Random Forest aprende los patrones presentes en el
    # conjunto de entrenamiento.
    #

    model.fit(

        X_train,

        y_train

    )

    # ======================================================
    # CREACIÓN DE LA CARPETA DE SALIDA
    # ======================================================
    #
    # Si la carpeta "models" aún no existe, se crea
    # automáticamente.
    #

    os.makedirs("models", exist_ok=True)

    # ======================================================
    # GUARDAR EL MODELO ENTRENADO
    # ======================================================
    #
    # El modelo se serializa utilizando pickle para poder
    # reutilizarlo posteriormente sin necesidad de volver a
    # entrenarlo.
    #

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # ======================================================
    # GUARDAR LOS LABEL ENCODERS
    # ======================================================
    #
    # También se almacenan los encoders para garantizar que
    # las futuras predicciones utilicen exactamente la misma
    # codificación aplicada durante el entrenamiento.
    #

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    # ======================================================
    # MENSAJE DE CONFIRMACIÓN
    # ======================================================

    print("✅ Modelo entrenado correctamente.")

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
    # Estos datos serán utilizados posteriormente para evaluar
    # el rendimiento del modelo.
    #

    return model, X_test, y_test