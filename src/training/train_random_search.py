import os
import pickle

import numpy as np
import pandas as pd

from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

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
    Entrena un Random Forest optimizado mediante
    RandomizedSearchCV utilizando ROC-AUC como
    métrica de optimización.
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
    # Eliminar registros sin target
    # ====================================

    filas_antes = len(df)

    df = df[df[TARGET].notna()].copy()

    print(
        f"ℹ️ Filas excluidas por target nulo: "
        f"{filas_antes - len(df)}"
    )

    # ====================================
    # Ingeniería de variables
    # ====================================

    df["semana_sin"] = np.sin(
        2 * np.pi * df["semana_anio"] / 52
    )

    df["semana_cos"] = np.cos(
        2 * np.pi * df["semana_anio"] / 52
    )

    # ====================================
    # Variables predictoras y objetivo
    # ====================================

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    # ====================================
    # Codificación variables categóricas
    # ====================================

    encoders = {}

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

        encoders[col] = encoder

    # ====================================
    # División entrenamiento / prueba
    # ====================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    # ====================================
    # Modelo base
    # ====================================

    rf = RandomForestClassifier(

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

        # ====================================
    # Espacio de búsqueda de hiperparámetros
    # ====================================

    param_dist = {

        # Número de árboles
        "n_estimators": randint(
            200,
            1000
        ),

        # Profundidad máxima
        "max_depth": [

            5,

            8,

            10,

            12,

            15,

            20,

            25,

            None

        ],

        # Criterio de división
        "criterion": [

            "gini",

            "entropy",

            "log_loss"

        ],

        # Número mínimo de muestras
        # para dividir un nodo

        "min_samples_split": randint(
            2,
            20
        ),

        # Número mínimo de muestras
        # en una hoja

        "min_samples_leaf": randint(
            1,
            10
        ),

        # Número de variables candidatas
        # en cada división

        "max_features": [

            "sqrt",

            "log2",

            None

        ],

        # Bootstrap

        "bootstrap": [

            True,

            False

        ],

        # Estrategia para clases desbalanceadas

        "class_weight": [

            None,

            "balanced",

            "balanced_subsample"

        ]

    }


    # ====================================
    # Random Search
    # ====================================

    random_search = RandomizedSearchCV(

        estimator=rf,

        param_distributions=param_dist,

        # Número de combinaciones

        n_iter=300,

        # Validación cruzada

        cv=5,

        # MÉTRICA PRINCIPAL

        scoring="roc_auc",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=2,

        return_train_score=True

    )


    print("\n")
    print("=" * 60)
    print("INICIANDO RANDOMIZED SEARCH")
    print("=" * 60)

    print("Métrica de optimización: ROC-AUC")

    print("Combinaciones a evaluar:", 300)

    print("Folds:", 5)

    print("=" * 60)


    random_search.fit(

        X_train,

        y_train

    )


    # ====================================
    # Mejor modelo encontrado
    # ====================================

    model = random_search.best_estimator_


    print("\n")
    print("=" * 60)
    print("RANDOMIZED SEARCH FINALIZADO")
    print("=" * 60)

    print("\nMejores hiperparámetros:\n")

    for parametro, valor in random_search.best_params_.items():

        print(

            f"{parametro}: {valor}"

        )


    print("\n")

    print(

        f"Mejor ROC-AUC promedio: "

        f"{random_search.best_score_:.4f}"

    )


        # ====================================
    # Crear carpeta models
    # ====================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ====================================
    # Guardar mejor modelo
    # ====================================

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )

    # ====================================
    # Guardar encoders
    # ====================================

    with open(
        ENCODERS_PATH,
        "wb"
    ) as f:

        pickle.dump(
            encoders,
            f
        )

    print("\n")
    print("=" * 60)
    print("MODELO OPTIMIZADO GUARDADO")
    print("=" * 60)

    print(
        f"Combinaciones evaluadas : {random_search.n_iter}"
    )

    print(
        f"Folds utilizados        : {random_search.cv}"
    )

    print(
        f"Modelos entrenados      : "
        f"{random_search.n_iter * random_search.cv}"
    )

    print("=" * 60)

    return (

        model,

        X_test,

        y_test

    )
