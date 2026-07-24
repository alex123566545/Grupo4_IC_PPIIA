import pandas as pd


def diagnosticar_features(model, X_train):
    """
    Imprime la importancia de cada feature del Random Forest entrenado,
    para revisar si las variables de tendencia (lag/media móvil) están
    aportando señal real o si son redundantes con features que ya
    existían (lo cual podría explicar la caída de AUC en GroupKFold).
    """

    importancias = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print("=" * 60)
    print("IMPORTANCIA DE VARIABLES (Random Forest)")
    print("=" * 60)
    print(importancias.to_string())

    print("\n--- Variables de tendencia (agregadas en este último cambio) ---")
    nuevas = [
        "casos_respiratorios_lag1", "casos_diarreicos_lag1",
        "media_movil_3s_temperatura", "media_movil_3s_pastura",
        "media_movil_3s_condicion_corporal", "media_movil_3s_casos_clinicos",
        "interaccion_desparasitacion_animales_nuevos",
    ]
    presentes = [n for n in nuevas if n in importancias.index]
    print(importancias.loc[presentes].sort_values(ascending=False).to_string())
    print(f"\nSuma de importancia de las 7 nuevas: {importancias.loc[presentes].sum():.3f}")
    print(f"(de referencia: si todas las ~41 features aportaran por igual, cada una valdría ~{1/41:.3f})")

    return importancias