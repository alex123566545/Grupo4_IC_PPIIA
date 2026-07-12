import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

    Parámetros
    ----------
    conn : psycopg2.connection
        Conexión a la base de datos.

    Retorna
    -------
    model : RandomForestRegressor
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
    y = df[TARGET]

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
    # ====================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # ====================================
    # Modelo
    # ====================================

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5,
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