import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import FEATURES, TARGET, RANDOM_STATE


def evaluar_groupkfold_por_lote(conn, n_splits=5):
    """
    Evalúa el modelo con GroupKFold agrupando por id_lote: en cada fold,
    los lotes de test NUNCA aparecieron en train. Esto responde la
    pregunta real: ¿el modelo aprende patrones que generalizan a un
    lote nuevo, o está memorizando el comportamiento histórico de
    cada uno de los 8 lotes que ya conoce?

    Si el R2 aquí es mucho más bajo que con train_test_split aleatorio,
    significa que el split anterior tenía fuga de lote (el mismo
    id_lote presente en train y test) y las métricas previas eran
    optimistas.
    """

    # ====================================
    # Leer datos (se necesita id_lote para agrupar, aunque no sea feature)
    # ====================================

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    df["semana_sin"] = np.sin(2 * np.pi * df["semana_anio"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_anio"] / 52)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
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

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        print(
            f"Fold {fold_i} | lotes test={lotes_test} | "
            f"n_test={len(X_test)} ({(y_test > 0).sum()} con mortalidad>0) | "
            f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.2f}"
        )

        resultados.append({"fold": fold_i, "lotes_test": lotes_test, "mae": mae, "rmse": rmse, "r2": r2})

    resultados_df = pd.DataFrame(resultados)

    print("\n" + "=" * 60)
    print("PROMEDIO ENTRE FOLDS (generalización real a lote nuevo):")
    print(f"MAE promedio  : {resultados_df['mae'].mean():.3f}  (std {resultados_df['mae'].std():.3f})")
    print(f"RMSE promedio : {resultados_df['rmse'].mean():.3f}  (std {resultados_df['rmse'].std():.3f})")
    print(f"R2 promedio   : {resultados_df['r2'].mean():.3f}  (std {resultados_df['r2'].std():.3f})")
    print("=" * 60)

    return resultados_df


if __name__ == "__main__":
    # Ajusta esta ruta de import a donde tengas definida tu función
    # get_connection() (la que usa psycopg2.connect(...) con Supabase)
    from src.config.database import get_connection  # <-- AJUSTA ESTA RUTA

    conn = get_connection()
    evaluar_groupkfold_por_lote(conn)
    conn.close()