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

from sklearn.metrics import (
    make_scorer,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

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
    RandomizedSearchCV.

    La optimización utiliza F-beta (β=0.5),
    favoreciendo Precision sin ignorar Recall.

    IMPORTANTE: no se sortea bootstrap=False junto con max_samples,
    porque scikit-learn levanta ValueError si max_samples != None y
    bootstrap=False. Combinarlos en el mismo espacio de búsqueda hace
    que ~37% de las combinaciones fallen silenciosamente (score=NaN),
    desperdiciando gran parte del presupuesto de RandomizedSearchCV.
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
    # Variables predictoras
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
    # Función de evaluación
    # Favorece Precision, sin abandonar Recall
    # ====================================

    fbeta = make_scorer(

        fbeta_score,

        beta=0.5

    )
        # ====================================
    # Espacio de búsqueda de hiperparámetros
    #
    # NOTA: bootstrap se fija en True. max_samples solo tiene efecto
    # (y solo es válido) cuando bootstrap=True -- sortearlos por
    # separado con bootstrap=False en el espacio hace fallar ~37% de
    # las combinaciones (ValueError de sklearn). Si quieres explorar
    # bootstrap=False, hazlo en una búsqueda aparte con max_samples
    # fijo en [None], nunca mezclado con valores numéricos.
    # ====================================

    param_dist = {

        # Número de árboles
        "n_estimators": randint(
            300,
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

        # Muestras mínimas para dividir un nodo
        "min_samples_split": randint(
            2,
            20
        ),

        # Muestras mínimas por hoja
        "min_samples_leaf": randint(
            1,
            10
        ),

        # Variables candidatas por división
        "max_features": [

            "sqrt",

            "log2",

            None

        ],

        # Bootstrap fijo en True (ver nota arriba)
        "bootstrap": [

            True

        ],

        # Balanceo de clases
        "class_weight": [

            None,

            "balanced",

            "balanced_subsample"

        ],

        # Poda del árbol
        "ccp_alpha": [

            0.0,

            0.0005,

            0.001,

            0.005,

            0.01

        ],

        # Porcentaje de muestras para cada árbol
        # (válido: bootstrap ya está fijo en True)
        "max_samples": [

            None,

            0.7,

            0.8,

            0.9

        ]

    }


    # ====================================
    # Randomized Search
    # ====================================

    random_search = RandomizedSearchCV(

        estimator=rf,

        param_distributions=param_dist,

        # Puedes subirlo a 500 si tienes tiempo
        n_iter=300,

        cv=5,

        # Favorece Precision
        scoring=fbeta,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=2,

        return_train_score=True

    )


    print("\n")
    print("=" * 60)
    print("INICIANDO RANDOMIZED SEARCH")
    print("=" * 60)

    print("Métrica de optimización : F-beta (β=0.5)")
    print("Prioridad              : Precision > Recall")
    print(f"Combinaciones          : {random_search.n_iter}")
    print(f"Folds                  : {random_search.cv}")

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
        f"Mejor F-beta promedio: "
        f"{random_search.best_score_:.4f}"
    )

    # ====================================
    # Cuántas combinaciones fallaron (diagnóstico)
    # ====================================

    scores = random_search.cv_results_["mean_test_score"]
    n_fallidas = int(np.isnan(scores).sum())
    print(
        f"Combinaciones fallidas (score=NaN): "
        f"{n_fallidas} de {len(scores)}"
    )

    # ====================================
    # Diagnóstico de umbral con piso de recall >= 0.60
    # (el proyecto exige recall >= 0.60; F-beta(0.5) solo no lo
    #  garantiza, así que se verifica explícitamente aquí)
    # ====================================

    proba_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, proba_test)

    print("\n")
    print("=" * 60)
    print("META DEL PROYECTO (Objetivo General, documento SIPREM-BOVINO)")
    print("=" * 60)
    print(f"AUC-ROC obtenido : {auc_test:.4f}  (meta: >= 0.70)  "
          f"{'✅ CUMPLE' if auc_test >= 0.70 else '❌ NO CUMPLE'}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)
    idx_validos = np.where(recalls[:-1] >= 0.60)[0]

    print("\n")
    print("=" * 60)
    print("DIAGNÓSTICO DE UMBRAL (recall mínimo = 0.60)")
    print("=" * 60)

    if len(idx_validos) == 0:
        print("⚠️  Ningún umbral alcanza recall >= 0.60 con este modelo. "
              "F-beta(0.5) puede haber sacrificado demasiado recall -- "
              "revisar si conviene volver a F1 o subir beta.")
    else:
        mejor_idx = idx_validos[np.argmax(precisions[idx_validos])]
        umbral_sugerido = thresholds[mejor_idx]
        pred_sugerida = (proba_test >= umbral_sugerido).astype(int)
        print(f"Umbral sugerido        : {umbral_sugerido:.3f}")
        print(f"Precision en ese umbral: {precision_score(y_test, pred_sugerida, zero_division=0):.4f}")
        print(f"Recall en ese umbral    : {recall_score(y_test, pred_sugerida, zero_division=0):.4f}")
        print(f"F1 en ese umbral        : {f1_score(y_test, pred_sugerida, zero_division=0):.4f}")

    print("=" * 60)

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

    # ====================================
    # Información final
    # ====================================

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

    # ====================================
    # Importancia de variables
    # ====================================

    importancia = pd.DataFrame({

        "Variable": X_train.columns,

        "Importancia": model.feature_importances_

    })

    importancia = importancia.sort_values(

        by="Importancia",

        ascending=False

    )

    print("\n")
    print("=" * 60)
    print("TOP 15 VARIABLES MÁS IMPORTANTES")
    print("=" * 60)

    print(
        importancia.head(15).to_string(index=False)
    )

    # ====================================
    # Retornar resultados
    # ====================================

    return (

        model,

        X_test,

        y_test

    )