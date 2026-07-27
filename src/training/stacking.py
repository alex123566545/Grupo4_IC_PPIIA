# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
#
# Se importan las librerías necesarias para:
#
# • Manipulación del sistema de archivos.
# • Guardado del modelo entrenado.
# • Procesamiento de datos.
# • Construcción del modelo Stacking.
# • División del conjunto de datos.
# • Codificación de variables categóricas.
#

import os
import pickle

import numpy as np
import pandas as pd

# ==========================================================
# IMPORTACIÓN DE LOS MODELOS BASE
# ==========================================================
#
# El modelo Stacking estará compuesto por dos algoritmos
# diferentes:
#
# • Random Forest
# • Gradient Boosting
#
# Además, se utilizará StackingClassifier para combinar las
# predicciones de ambos modelos.
#

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

# ==========================================================
# IMPORTACIÓN DEL META-MODELO
# ==========================================================
#
# Logistic Regression será el modelo encargado de aprender
# cómo combinar las predicciones generadas por los modelos
# base.
#

from sklearn.linear_model import LogisticRegression

# ==========================================================
# PREPROCESAMIENTO
# ==========================================================
#
# Se importan las herramientas necesarias para dividir el
# conjunto de datos y convertir variables categóricas en
# valores numéricos.
#

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================================
#
# Se importan las constantes utilizadas durante el proceso
# de entrenamiento.
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
#     Semilla para garantizar resultados reproducibles.
#
# MODEL_PATH
#     Ruta donde se almacenará el modelo entrenado.
#
# ENCODERS_PATH
#     Ruta donde se guardarán los LabelEncoder.
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
# Esta función realiza el entrenamiento completo del modelo
# Stacking siguiendo las siguientes etapas:
#
# 1. Leer los datos desde la base de datos.
# 2. Eliminar registros inválidos.
# 3. Realizar Feature Engineering.
# 4. Preparar las variables predictoras.
# 5. Codificar variables categóricas.
# 6. Dividir los datos en entrenamiento y prueba.
# 7. Crear los modelos base.
# 8. Crear el meta-modelo.
# 9. Construir el modelo Stacking.
# 10. Entrenar el modelo.
# 11. Guardar el modelo entrenado.
# 12. Guardar los LabelEncoder.
#

def train_model(conn):
    """
    Entrena un modelo Stacking.

    Modelos base:
        - Random Forest
        - Gradient Boosting

    Meta-modelo:
        - Logistic Regression
    """

    # ====================================
    # LECTURA DE LOS DATOS
    # ====================================
    #
    # Se obtiene desde la base de datos el conjunto de
    # variables preparado durante el proceso ETL.
    #

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    # ====================================
    # ELIMINACIÓN DE REGISTROS INVÁLIDOS
    # ====================================
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
        f"ℹ️ Filas excluidas por target nulo: {filas_antes-len(df)}"
    )

    # ====================================
    # FEATURE ENGINEERING
    # ====================================
    #
    # La variable semana_anio posee naturaleza cíclica.
    #
    # Para representar correctamente esta característica se
    # crean dos nuevas variables utilizando las funciones
    # seno y coseno.
    #

    df["semana_sin"] = np.sin(
        2 * np.pi * df["semana_anio"] / 52
    )

    df["semana_cos"] = np.cos(
        2 * np.pi * df["semana_anio"] / 52
    )

    # ====================================
    # SEPARACIÓN DE VARIABLES
    # ====================================
    #
    # X contiene todas las variables predictoras.
    #
    # y corresponde a la variable objetivo.
    #

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    # ====================================
    # CODIFICACIÓN DE VARIABLES
    # ====================================
    #
    # Los modelos de Machine Learning utilizados requieren
    # variables numéricas.
    #
    # Todas las variables categóricas son convertidas mediante
    # LabelEncoder.
    #
    # Cada encoder es almacenado para reutilizar exactamente
    # la misma codificación durante futuras predicciones.
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

    # ====================================
    # DIVISIÓN DEL CONJUNTO DE DATOS
    # ====================================
    #
    # Se divide el conjunto de datos en entrenamiento y
    # prueba.
    #
    # Se utiliza estratificación para conservar la misma
    # proporción de clases en ambos conjuntos.
    #

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ====================================
    # MODELO BASE 1
    # ====================================
    #
    # Primer algoritmo del Stacking.
    #
    # Random Forest construye múltiples árboles de decisión y
    # combina sus predicciones mediante votación.
    #

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        criterion="gini",
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # ====================================
    # MODELO BASE 2
    # ====================================
    #
    # Segundo algoritmo del Stacking.
    #
    # Gradient Boosting construye árboles de forma secuencial,
    # donde cada nuevo árbol intenta corregir los errores del
    # anterior.
    #

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    # ====================================
    # DEFINICIÓN DE LOS MODELOS BASE
    # ====================================
    #
    # Se agrupan todos los modelos que participarán en el
    # proceso de Stacking.
    #

    estimators = [
        ("RandomForest", rf),
        ("GradientBoosting", gb)
    ]

    # ====================================
    # META-MODELO
    # ====================================
    #
    # Logistic Regression aprenderá a combinar las
    # predicciones generadas por los modelos base para obtener
    # una predicción final.
    #

    meta_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    # ====================================
    # CONSTRUCCIÓN DEL STACKING
    # ====================================
    #
    # El StackingClassifier coordina el entrenamiento de los
    # modelos base y posteriormente utiliza el meta-modelo
    # para combinar sus predicciones.
    #
    # Características:
    #
    # • Validación cruzada de 5 folds.
    # • Utiliza probabilidades como entrada del meta-modelo.
    # • Emplea todos los núcleos disponibles.
    #

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False
    )

    # ====================================
    # ENTRENAMIENTO DEL MODELO
    # ====================================
    #
    # Primero se entrenan los modelos base.
    #
    # Posteriormente, sus predicciones son utilizadas como
    # entrada para entrenar el meta-modelo.
    #

    model.fit(X_train, y_train)

    # ====================================
    # CREACIÓN DE LA CARPETA DE MODELOS
    # ====================================
    #
    # Si la carpeta "models" no existe, se crea
    # automáticamente.
    #

    os.makedirs("models", exist_ok=True)

    # ====================================
    # GUARDADO DEL MODELO
    # ====================================
    #
    # El modelo entrenado se serializa utilizando pickle para
    # poder reutilizarlo posteriormente sin necesidad de
    # volver a entrenarlo.
    #

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # ====================================
    # GUARDADO DE LOS LABEL ENCODERS
    # ====================================
    #
    # También se almacenan los encoders utilizados durante el
    # entrenamiento para garantizar que las futuras
    # predicciones empleen exactamente la misma codificación.
    #

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    # ====================================
    # MENSAJE DE CONFIRMACIÓN
    # ====================================

    print("✅ Modelo Stacking entrenado correctamente.")

    # ====================================
    # RETORNO DE LA FUNCIÓN
    # ====================================
    #
    # Se devuelve:
    #
    # • El modelo Stacking entrenado.
    # • El conjunto de prueba.
    # • Las etiquetas reales del conjunto de prueba.
    #
    # Estos datos serán utilizados posteriormente para
    # evaluar el rendimiento del modelo.
    #

    return model, X_test, y_test