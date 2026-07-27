import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#from preprocessing.feature_engineering import feature_engineering
from src.config.settings import (
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    ENCODERS_PATH
)


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

    # ====================================
    # Leer datos
    # ====================================

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    # ====================================
    # Excluir filas sin target válido
    # (target_riesgo_alto_3sem queda NULL en las últimas semanas de cada
    #  lote, donde no hay 3 semanas completas de futuro para evaluar)
    # ====================================

    filas_antes = len(df)
    df = df[df[TARGET].notna()].copy()
    print(f"ℹ️  Filas excluidas por target nulo (sin futuro suficiente): {filas_antes - len(df)}")

    # ====================================
    # Feature engineering: codificación cíclica de semana_anio
    # (semana 52 y semana 1 deben quedar "cerca" para el modelo,
    #  cosa que un entero plano de 1 a 52 no representa)
    # ====================================

    df["semana_sin"] = np.sin(2 * np.pi * df["semana_anio"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_anio"] / 52)

    # ====================================
    # Variables predictoras y objetivo
    # ====================================

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # ====================================
    # Encoding variables categóricas
    # ====================================

    encoders = {}

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )

        encoders[col] = le

    # ====================================
    # Train Test Split
    # (estratificado por la clase objetivo: conserva la proporción
    #  real ~36% riesgo alto / ~64% riesgo bajo en train y test,
    #  con el target de ventana de 3 semanas)
    # ====================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ====================================
    # Modelo
    # (Random Forest de 300 árboles con poda de profundidad y pesos
    #  de clase balanceados, según la Fase 4 del proyecto)
    # ====================================

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ====================================
    # Crear carpeta models
    # ====================================

    os.makedirs("models", exist_ok=True)

    # ====================================
    # Guardar modelo
    # ====================================

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # ====================================
    # Guardar encoders
    # ====================================

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print("✅ Modelo entrenado correctamente.")

    return model, X_test, y_test