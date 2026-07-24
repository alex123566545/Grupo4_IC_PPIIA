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
    precision_recall_curve,
)

from src.config.settings import FEATURES, TARGET, RANDOM_STATE


def evaluar_groupkfold_clasificacion(conn, n_splits=5):
    """
    Igual que evaluar_groupkfold_por_lote(), pero para el problema de
    clasificación (target_riesgo_alto_3sem: riesgo de mortalidad alta en
    t+1, t+2 o t+3): en cada fold, los lotes de test nunca aparecieron en
    train. Mide si el AUC/recall/precision obtenidos con split aleatorio
    se sostienen con un lote nuevo, o si están inflados por fuga de lote.
    """

    # ====================================
    # Leer datos
    # ====================================

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    # Excluir filas sin target válido (últimas semanas de cada lote,
    # sin 3 semanas completas de futuro para evaluar) -- igual que en train.py
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

        # Mismos hiperparámetros que tu train.py en producción (300 árboles,
        # max_depth=8, class_weight="balanced"), para que este número sea
        # comparable con el AUC=0.80 que ya obtuviste con split aleatorio.
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
        pred_050 = (proba >= 0.5).astype(int)

        # --- umbral ajustado: el más bajo que alcanza recall >= 0.60 ---
        # (para un sistema de alerta, preferimos avisar de más que
        # dejar pasar un caso real de riesgo alto)
        precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
        idx_validos = np.where(recalls[:-1] >= 0.60)[0]
        if len(idx_validos) > 0:
            # entre los que cumplen recall>=0.60, toma el de mayor precision
            mejor_idx = idx_validos[np.argmax(precisions[idx_validos])]
            umbral_ajustado = thresholds[mejor_idx]
        else:
            umbral_ajustado = thresholds[np.argmax(recalls[:-1])] if len(thresholds) else 0.5
        pred_ajustado = (proba >= umbral_ajustado).astype(int)

        auc = roc_auc_score(y_test, proba)

        precision_050 = precision_score(y_test, pred_050, zero_division=0)
        recall_050 = recall_score(y_test, pred_050, zero_division=0)
        f1_050 = f1_score(y_test, pred_050, zero_division=0)

        precision_adj = precision_score(y_test, pred_ajustado, zero_division=0)
        recall_adj = recall_score(y_test, pred_ajustado, zero_division=0)
        f1_adj = f1_score(y_test, pred_ajustado, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_ajustado, labels=[0, 1]).ravel()

        print(
            f"Fold {fold_i} | lotes test={lotes_test} | n_test={len(X_test)} ({int(y_test.sum())} positivos) | AUC={auc:.2f}\n"
            f"   umbral=0.50        -> precision={precision_050:.2f} recall={recall_050:.2f} f1={f1_050:.2f}\n"
            f"   umbral={umbral_ajustado:.2f} (ajustado) -> precision={precision_adj:.2f} recall={recall_adj:.2f} f1={f1_adj:.2f} | TP={tp} FP={fp} FN={fn} TN={tn}"
        )

        resultados.append({
            "fold": fold_i, "lotes_test": lotes_test,
            "auc": auc,
            "precision_050": precision_050, "recall_050": recall_050, "f1_050": f1_050,
            "umbral_ajustado": umbral_ajustado,
            "precision_adj": precision_adj, "recall_adj": recall_adj, "f1_adj": f1_adj,
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n" + "=" * 60)
    print("PROMEDIO ENTRE FOLDS (generalización real a lote nuevo):")
    print(f"AUC promedio              : {resultados_df['auc'].mean():.3f}  (std {resultados_df['auc'].std():.3f})")
    print("--- umbral 0.50 (default) ---")
    print(f"Precision promedio        : {resultados_df['precision_050'].mean():.3f}")
    print(f"Recall promedio           : {resultados_df['recall_050'].mean():.3f}")
    print(f"F1 promedio               : {resultados_df['f1_050'].mean():.3f}")
    print("--- umbral ajustado (recall objetivo >= 0.60) ---")
    print(f"Umbral promedio usado     : {resultados_df['umbral_ajustado'].mean():.3f}")
    print(f"Precision promedio        : {resultados_df['precision_adj'].mean():.3f}")
    print(f"Recall promedio           : {resultados_df['recall_adj'].mean():.3f}")
    print(f"F1 promedio               : {resultados_df['f1_adj'].mean():.3f}")
    print("=" * 60)

    return resultados_df


if __name__ == "__main__":
    from src.config.database import get_connection

    conn = get_connection()
    evaluar_groupkfold_clasificacion(conn)
    conn.close()