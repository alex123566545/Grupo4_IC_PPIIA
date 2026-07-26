import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config.settings import FEATURES, TARGET, RANDOM_STATE

# ==============================================================================
# Hiperparámetros ganadores del RandomizedSearchCV (train_random_search.py).
# Se dejan FIJOS aquí -- no se vuelve a correr la búsqueda dentro de cada
# fold (sería carísimo: 300 combinaciones x 5 folds x 5 sub-folds de CV).
# El objetivo de este script es otro: confirmar si ESTE modelo ya elegido
# generaliza a lotes que nunca vio, no volver a buscar hiperparámetros.
# ==============================================================================

MEJORES_HIPERPARAMETROS = dict(
    n_estimators=815,
    max_depth=15,
    criterion="entropy",
    min_samples_split=6,
    min_samples_leaf=8,
    max_features=None,
    bootstrap=True,
    max_samples=0.7,
    class_weight=None,
    ccp_alpha=0.005,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def evaluar_groupkfold_hiperparametros_fijos(conn, n_splits=5, recall_minimo=0.60):
    """
    Valida, con GroupKFold agrupado por id_lote, si el modelo con los
    hiperparámetros ganadores del RandomizedSearchCV generaliza a lotes
    que nunca vio en entrenamiento -- o si el buen resultado (AUC=0.81,
    precision=0.696, recall=0.611 con split aleatorio) estaba inflado
    por fuga de lote, como ya pasó con versiones anteriores del modelo.

    En cada fold se re-ajusta el umbral de decisión (mismo criterio que
    en buscar_mejor_umbral.py: recall >= recall_minimo como piso,
    maximizando precision dentro de eso), porque el punto óptimo de
    corte puede variar ligeramente según qué lotes caen en cada fold.
    """

    # ====================================
    # Leer datos
    # ====================================

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    filas_antes = len(df)
    df = df[df[TARGET].notna()].copy()
    print(f"ℹ️  Filas excluidas por target nulo: {filas_antes - len(df)}")

    df["semana_sin"] = np.sin(2 * np.pi * df["semana_anio"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_anio"] / 52)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    grupos = df["id_lote"]

    n_lotes = grupos.nunique()
    if n_splits > n_lotes:
        print(
            f"⚠️  Pediste {n_splits} folds pero solo hay {n_lotes} lotes. "
            f"Usando n_splits={n_lotes} (1 lote por fold)."
        )
        n_splits = n_lotes

    # ====================================
    # Encoding (igual que en train_random_search.py, fit sobre todo X)
    # ====================================

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # ====================================
    # GroupKFold: los lotes de cada fold de test no están en su train
    # ====================================

    gkf = GroupKFold(n_splits=n_splits)

    resultados = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=grupos), start=1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        lotes_test = sorted(grupos.iloc[test_idx].unique())

        model = RandomForestClassifier(**MEJORES_HIPERPARAMETROS)
        model.fit(X_train, y_train)

        if y_test.nunique() < 2:
            print(
                f"Fold {fold_i} | lotes test={lotes_test} | "
                f"n_test={len(X_test)} (todos clase {y_test.iloc[0]}) | "
                f"AUC no calculable (una sola clase en test)"
            )
            resultados.append({
                "fold": fold_i, "lotes_test": lotes_test,
                "auc": np.nan, "umbral": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
            })
            continue

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        # --- mismo criterio que buscar_mejor_umbral.py: recall como piso ---
        mejor_umbral, mejor_precision, mejor_recall, mejor_f1 = None, -1, 0, -1
        for umbral in np.arange(0.30, 0.91, 0.01):
            pred = (proba >= umbral).astype(int)
            p = precision_score(y_test, pred, zero_division=0)
            r = recall_score(y_test, pred, zero_division=0)
            f1v = f1_score(y_test, pred, zero_division=0)
            if r >= recall_minimo and (p > mejor_precision or (p == mejor_precision and f1v > mejor_f1)):
                mejor_umbral, mejor_precision, mejor_recall, mejor_f1 = umbral, p, r, f1v

        if mejor_umbral is None:
            # ningún umbral llegó al piso de recall en este fold
            mejores = []
            for umbral in np.arange(0.30, 0.91, 0.01):
                pred = (proba >= umbral).astype(int)
                mejores.append((umbral, recall_score(y_test, pred, zero_division=0),
                                 precision_score(y_test, pred, zero_division=0),
                                 f1_score(y_test, pred, zero_division=0)))
            mejor_umbral, mejor_recall, mejor_precision, mejor_f1 = max(mejores, key=lambda t: t[1])
            print(f"⚠️  Fold {fold_i}: ningún umbral alcanzó recall >= {recall_minimo}")

        tn, fp, fn, tp = confusion_matrix(
            y_test, (proba >= mejor_umbral).astype(int), labels=[0, 1]
        ).ravel()

        print(
            f"Fold {fold_i} | lotes test={lotes_test} | n_test={len(X_test)} "
            f"({int(y_test.sum())} positivos) | AUC={auc:.3f} | "
            f"umbral={mejor_umbral:.2f} -> precision={mejor_precision:.3f} "
            f"recall={mejor_recall:.3f} f1={mejor_f1:.3f} | "
            f"TP={tp} FP={fp} FN={fn} TN={tn}"
        )

        resultados.append({
            "fold": fold_i, "lotes_test": lotes_test,
            "auc": auc, "umbral": mejor_umbral,
            "precision": mejor_precision, "recall": mejor_recall, "f1": mejor_f1,
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n" + "=" * 70)
    print("PROMEDIO ENTRE FOLDS (generalización real a lote nuevo)")
    print("=" * 70)
    print(f"AUC promedio       : {resultados_df['auc'].mean():.3f}  (std {resultados_df['auc'].std():.3f})")
    print(f"Precision promedio : {resultados_df['precision'].mean():.3f}")
    print(f"Recall promedio    : {resultados_df['recall'].mean():.3f}")
    print(f"F1 promedio        : {resultados_df['f1'].mean():.3f}")
    print("=" * 70)

    print("\nComparación con split aleatorio (referencia):")
    print("  Split aleatorio -> AUC=0.810  Precision=0.696  Recall=0.611  F1=0.651")

    return resultados_df


if __name__ == "__main__":
    from src.config.database import get_connection

    conn = get_connection()
    evaluar_groupkfold_hiperparametros_fijos(conn)
    conn.close()