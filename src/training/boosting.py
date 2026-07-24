import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
    # ====================================

    filas_antes = len(df)

    df = df[df[TARGET].notna()].copy()

    print(
        f"ℹ️ Filas excluidas por target nulo: {filas_antes - len(df)}"
    )

    # ====================================
    # Feature Engineering
    # Codificación cíclica de semana
    # ====================================

    df["semana_sin"] = np.sin(
        2 * np.pi * df["semana_anio"] / 52
    )

    df["semana_cos"] = np.cos(
        2 * np.pi * df["semana_anio"] / 52
    )

    # ====================================
    # Variables predictoras
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
    # ====================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ====================================
    # Modelo Gradient Boosting
    # ====================================

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    # ====================================
    # Entrenamiento
    # ====================================

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

    print("✅ Modelo Gradient Boosting entrenado correctamente.")

    return model, X_test, y_test