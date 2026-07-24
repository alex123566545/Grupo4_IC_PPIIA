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
    RandomizedSearchCV buscando maximizar Precision.
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
    # Feature engineering
    # ====================================

    df["semana_sin"] = np.sin(
        2 * np.pi * df["semana_anio"] / 52
    )

    df["semana_cos"] = np.cos(
        2 * np.pi * df["semana_anio"] / 52
    )


    # ====================================
    # Separar variables
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
    # Random Forest base
    # ====================================

    rf = RandomForestClassifier(

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )
        # ====================================
    # Espacio de búsqueda de hiperparámetros
    # Optimizado para Precision
    # ====================================

    param_dist = {

        # Cantidad de árboles
        "n_estimators": randint(
            200,
            800
        ),


        # Profundidad máxima del árbol
        "max_depth": [
            5,
            8,
            10,
            12,
            15,
            20,
            None
        ],


        # Mínimo de muestras para dividir un nodo
        "min_samples_split": randint(
            2,
            12
        ),


        # Mínimo de muestras en una hoja
        "min_samples_leaf": randint(
            1,
            6
        ),


        # Función de impureza
        "criterion": [

            "gini",

            "entropy",

            "log_loss"

        ],


        # Número de variables consideradas
        # en cada división
        "max_features": [

            "sqrt",

            "log2"

        ],


        # Usar bootstrap
        "bootstrap": [

            True

        ]

    }


    # ====================================
    # Randomized Search
    # ====================================

    random_search = RandomizedSearchCV(

        estimator=rf,


        param_distributions=param_dist,


        # Probar 300 combinaciones diferentes

        n_iter=300,


        # IMPORTANTE:
        # ahora buscamos PRECISION

        scoring="precision",


        # Validación cruzada

        cv=5,


        random_state=RANDOM_STATE,


        verbose=2,


        n_jobs=-1,


        return_train_score=True

    )



    print("\n")
    print("=" * 60)
    print("INICIANDO RANDOMIZED SEARCH")
    print("OBJETIVO: MAXIMIZAR PRECISION")
    print("=" * 60)



    random_search.fit(

        X_train,

        y_train

    )



    # ====================================
    # Obtener mejor modelo
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
        "Mejor Precision promedio CV:"
    )


    print(
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
    # Guardar modelo optimizado
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
    print("MODELO RANDOM FOREST OPTIMIZADO GUARDADO")
    print("=" * 60)



    print(
        f"Combinaciones evaluadas: "
        f"{random_search.n_iter}"
    )


    print(
        f"Folds utilizados: "
        f"{random_search.cv}"
    )


    print(
        f"Total entrenamientos realizados: "
        f"{random_search.n_iter * random_search.cv}"
    )


    print("=" * 60)



    return (
        model,
        X_test,
        y_test
    )