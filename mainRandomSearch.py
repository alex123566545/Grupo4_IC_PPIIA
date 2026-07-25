from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import numpy as np

from src.config.database import get_connection
from src.training.train_random_search import train_model


def buscar_mejor_umbral(
        y_real,
        probabilidades,
        precision_minima=0.70
):
    """
    Busca el umbral que mantenga una Precision mínima
    y maximice el Recall.
    """

    mejor_umbral = 0.50
    mejor_recall = -1

    resultados = []

    for umbral in np.arange(0.30, 0.91, 0.01):

        predicciones = (
            probabilidades >= umbral
        ).astype(int)

        precision = precision_score(
            y_real,
            predicciones,
            zero_division=0
        )

        recall = recall_score(
            y_real,
            predicciones,
            zero_division=0
        )

        f1 = f1_score(
            y_real,
            predicciones,
            zero_division=0
        )

        resultados.append({
            "umbral": umbral,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        # Solo considerar umbrales con la precisión mínima requerida
        if precision >= precision_minima:

            if recall > mejor_recall:

                mejor_recall = recall
                mejor_umbral = umbral

    return mejor_umbral, resultados


def main():

    # ====================================
    # Conexión
    # ====================================

    conn = get_connection()

    # ====================================
    # Entrenamiento
    # ====================================

    model, X_test, y_test = train_model(conn)

    # ====================================
    # Probabilidades
    # ====================================

    y_prob = model.predict_proba(X_test)[:, 1]

    # ====================================
    # Buscar mejor umbral
    # ====================================

    mejor_umbral, resultados = buscar_mejor_umbral(

        y_test,

        y_prob,

        precision_minima=0.70

    )

    # ====================================
    # Mostrar todos los resultados
    # ====================================

    print("\n")
    print("=" * 80)
    print("RESULTADOS PARA CADA UMBRAL")
    print("=" * 80)

    print(f"{'Umbral':<10}{'Precision':<12}{'Recall':<12}{'F1'}")

    for r in resultados:

        print(
            f"{r['umbral']:<10.2f}"
            f"{r['precision']:<12.3f}"
            f"{r['recall']:<12.3f}"
            f"{r['f1']:.3f}"
        )

    print("\n")
    print("=" * 80)
    print("MEJOR UMBRAL ENCONTRADO")
    print("=" * 80)

    print(f"Umbral seleccionado: {mejor_umbral:.2f}")
    print(f"Precision mínima requerida: 0.70")

    # ====================================
    # Predicción final
    # ====================================

    y_pred = (

        y_prob >= mejor_umbral

    ).astype(int)

    # ====================================
    # Métricas
    # ====================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    matriz = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n")
    print("=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    print("\nMatriz de confusión")
    print(matriz)

    print("\nReporte de clasificación")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    conn.close()


if __name__ == "__main__":

    main()