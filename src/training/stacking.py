import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression
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
    Entrena un modelo Stacking.

    Modelos base:
        - Random Forest
        - Gradient Boosting

    Meta-modelo:
        - Logistic Regression
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
    # Eliminar targets nulos
    # ====================================

    filas_antes = len(df)

    df = df[df[TARGET].notna()].copy()

    print(
        f"ℹ️ Filas excluidas por target nulo: {filas_antes-len(df)}"
    )

    # ====================================
    # Feature Engineering
    # ====================================

    df["semana_sin"] = np.sin(
        2 * np.pi * df["semana_anio"] / 52
    )

    df["semana_cos"] = np.cos(
        2 * np.pi * df["semana_anio"] / 52
    )

    # ====================================
    # Variables
    # ====================================

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    # ====================================
    # Label Encoding
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
    # Modelo Random Forest
    # ====================================

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
    # Modelo Gradient Boosting
    # ====================================

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    # ====================================
    # Modelos base
    # ====================================

    estimators = [
        ("RandomForest", rf),
        ("GradientBoosting", gb)
    ]

    # ====================================
    # Meta-modelo
    # ====================================

    meta_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    # ====================================
    # Stacking
    # ====================================

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False
    )

    # ====================================
    # Entrenamiento
    # ====================================

    model.fit(X_train, y_train)

    # ====================================
    # Crear carpeta
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

    print("✅ Modelo Stacking entrenado correctamente.")

    return model, X_test, y_test