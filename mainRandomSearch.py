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
    Busca automáticamente el mejor umbral.

    Criterios:

    1. Precision >= precision_minima

    2. Entre ellos,
       elegir el mayor Recall.

    3. Si hay empate,
       elegir el mayor F1.
    """

    mejor_umbral = None

    mejor_precision = 0

    mejor_recall = -1

    mejor_f1 = -1

    resultados = []

    for umbral in np.arange(
        0.30,
        0.91,
        0.01
    ):

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

        # Solo considerar umbrales
        # con precisión suficiente

        if precision >= precision_minima:

            if (

                recall > mejor_recall

                or

                (

                    recall == mejor_recall

                    and

                    f1 > mejor_f1

                )

            ):

                mejor_umbral = umbral

                mejor_precision = precision

                mejor_recall = recall

                mejor_f1 = f1

    # Si ningún umbral alcanza la
    # precisión solicitada, usar el
    # de mayor Precision disponible.

    if mejor_umbral is None:

        mejor = max(

            resultados,

            key=lambda x: x["precision"]

        )

        mejor_umbral = mejor["umbral"]

        mejor_precision = mejor["precision"]

        mejor_recall = mejor["recall"]

        mejor_f1 = mejor["f1"]

        print("\n⚠ Ningún umbral alcanzó la Precision solicitada.")

        print("Se utilizará el umbral con mayor Precision.\n")

    return (

        mejor_umbral,

        mejor_precision,

        mejor_recall,

        mejor_f1,

        resultados

    )


def main():

    # ====================================
    # Conexión
    # ====================================

    conn = get_connection()

    # ====================================
    # Entrenar modelo
    # ====================================

    model, X_test, y_test = train_model(conn)

    # ====================================
    # Obtener probabilidades
    # ====================================

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

        # ====================================
    # Buscar mejor umbral
    # ====================================

    (
        mejor_umbral,
        mejor_precision,
        mejor_recall,
        mejor_f1,
        resultados
    ) = buscar_mejor_umbral(

        y_test,

        y_prob,

        precision_minima=0.70

    )

    # ====================================
    # Mostrar resultados de todos
    # los umbrales
    # ====================================

    print("\n")
    print("=" * 90)
    print("RESULTADOS PARA CADA UMBRAL")
    print("=" * 90)

    print(
        f"{'Umbral':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1'}"
    )

    for r in resultados:

        print(

            f"{r['umbral']:<10.2f}"

            f"{r['precision']:<12.3f}"

            f"{r['recall']:<12.3f}"

            f"{r['f1']:.3f}"

        )

    # ====================================
    # Mejor umbral encontrado
    # ====================================

    print("\n")
    print("=" * 90)
    print("MEJOR UMBRAL ENCONTRADO")
    print("=" * 90)

    print(f"Umbral seleccionado : {mejor_umbral:.2f}")
    print(f"Precision           : {mejor_precision:.3f}")
    print(f"Recall              : {mejor_recall:.3f}")
    print(f"F1 Score            : {mejor_f1:.3f}")

    print("\nCriterio utilizado:")

    print("1. Precision >= 0.70")

    print("2. Mayor Recall")

    print("3. Mayor F1 (desempate)")

    # ====================================
    # Predicción final
    # ====================================

    y_pred = (

        y_prob >= mejor_umbral

    ).astype(int)

    # ====================================
    # Calcular métricas finales
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

        # ====================================
    # Mostrar resultados finales
    # ====================================

    print("\n")
    print("=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    print("\n")
    print("=" * 60)
    print("MATRIZ DE CONFUSIÓN")
    print("=" * 60)

    print(matriz)

    tn, fp, fn, tp = matriz.ravel()

    print("\nDetalle:")

    print(f"TP (Verdaderos Positivos): {tp}")

    print(f"FP (Falsos Positivos)    : {fp}")

    print(f"FN (Falsos Negativos)    : {fn}")

    print(f"TN (Verdaderos Negativos): {tn}")

    print("\n")
    print("=" * 60)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 60)

    print(

        classification_report(

            y_test,

            y_pred,

            zero_division=0

        )

    )

    # ====================================
    # Resumen ejecutivo
    # ====================================

    print("\n")
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    if precision >= 0.70:

        print("✅ Se alcanzó la Precision objetivo (>= 0.70).")

    else:

        print("❌ No se alcanzó la Precision objetivo.")

    print(f"Umbral seleccionado : {mejor_umbral:.2f}")
    print(f"Precision final     : {precision:.4f}")
    print(f"Recall final        : {recall:.4f}")
    print(f"F1 Score final      : {f1:.4f}")
    print(f"AUC-ROC             : {auc:.4f}")

    print("=" * 60)

    # ====================================
    # Cerrar conexión
    # ====================================

    conn.close()


if __name__ == "__main__":

    main()