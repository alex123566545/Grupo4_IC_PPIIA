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

from src.config.settings import FEATURES, RANDOM_STATE

TARGET_CLASIFICACION = "target_riesgo_alto"


def evaluar_groupkfold_clasificacion(conn, n_splits=5):
    """
    Igual que evaluar_groupkfold_por_lote(), pero para el problema de
    clasificación (target_riesgo_alto): en cada fold, los lotes de test
    nunca aparecieron en train. Mide si el buen AUC visto con split
    aleatorio (0.74-0.82) se sostiene con un lote nuevo, o si también
    estaba inflado por fuga de lote.
    """

    # ====================================
    # Leer datos
    # ====================================

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    df["semana_sin"] = np.sin(2 * np.pi * df["semana_anio"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_anio"] / 52)

    X = df[FEATURES].copy()
    y = df[TARGET_CLASIFICACION].copy()
    grupos = df["id_lote"]

    n_lotes = grupos.nunique()
    if n_splits > n_lotes:
        print(
            f"⚠️  Pediste {n_splits} folds pero solo hay {n_lotes} lotes. "
            f"Usando n_splits={n_lotes} (1 lote por fold)."
        )
        n_splits = n_lotes

    # ====================================
    # Encoding (igual que en train_model, fit sobre todo X)
    # ====================================

    categorical_columns = X.select_dtypes(include=["object", "category", "str"]).columns
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

        # OJO: sin class_weight="balanced" para no descalibrar
        # predict_proba(), que es lo que usamos para el AUC.
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # si el fold de test no tiene ambas clases, AUC no se puede calcular
        if y_test.nunique() < 2:
            print(
                f"Fold {fold_i} | lotes test={lotes_test} | "
                f"n_test={len(X_test)} (todos clase {y_test.iloc[0]}) | "
                f"AUC no calculable (una sola clase en test)"
            )
            resultados.append({
                "fold": fold_i, "lotes_test": lotes_test,
                "auc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan
            })
            continue

        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

        print(
            f"Fold {fold_i} | lotes test={lotes_test} | "
            f"n_test={len(X_test)} ({int(y_test.sum())} positivos) | "
            f"AUC={auc:.2f} precision={precision:.2f} recall={recall:.2f} f1={f1:.2f} | "
            f"TP={tp} FP={fp} FN={fn} TN={tn}"
        )

        resultados.append({
            "fold": fold_i, "lotes_test": lotes_test,
            "auc": auc, "precision": precision, "recall": recall, "f1": f1
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n" + "=" * 60)
    print("PROMEDIO ENTRE FOLDS (generalización real a lote nuevo):")
    print(f"AUC promedio       : {resultados_df['auc'].mean():.3f}  (std {resultados_df['auc'].std():.3f})")
    print(f"Precision promedio : {resultados_df['precision'].mean():.3f}  (std {resultados_df['precision'].std():.3f})")
    print(f"Recall promedio    : {resultados_df['recall'].mean():.3f}  (std {resultados_df['recall'].std():.3f})")
    print(f"F1 promedio        : {resultados_df['f1'].mean():.3f}  (std {resultados_df['f1'].std():.3f})")
    print("=" * 60)

    return resultados_df


if __name__ == "__main__":
    # Ajusta esta ruta de import a donde tengas definida tu función
    # get_connection() (la que usa psycopg2.connect(...) con Supabase)
    from src.config.database import get_connection  # <-- AJUSTA ESTA RUTA

    conn = get_connection()
    evaluar_groupkfold_clasificacion(conn)
    conn.close()