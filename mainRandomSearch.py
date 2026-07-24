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
        probabilidades
):

    """
    Busca el umbral que maximiza Precision.
    """

    mejor_precision = 0

    mejor_umbral = 0.5


    resultados = []


    # probar diferentes umbrales

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


        resultados.append(
            {
                "umbral": umbral,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        )



        if precision > mejor_precision:

            mejor_precision = precision

            mejor_umbral = umbral



    return (
        mejor_umbral,
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

    model, X_test, y_test = train_model(
        conn
    )



    # ====================================
    # Probabilidades
    # ====================================

    y_prob = model.predict_proba(
        X_test
    )[:,1]



    # ====================================
    # Buscar mejor umbral
    # ====================================

    mejor_umbral, resultados = buscar_mejor_umbral(

        y_test,

        y_prob

    )



    print("\n")
    print("=" * 60)
    print("MEJOR UMBRAL ENCONTRADO")
    print("=" * 60)


    print(
        f"Umbral: {mejor_umbral:.2f}"
    )



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
        y_pred
    )


    recall = recall_score(
        y_test,
        y_pred
    )


    f1 = f1_score(
        y_test,
        y_pred
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
    print("RESULTADOS RANDOM FOREST OPTIMIZADO")
    print("=" * 60)


    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"AUC-ROC  : {auc:.4f}"
    )


    print("\nMatriz de confusión")

    print(
        matriz
    )


    print("\nReporte de clasificación")


    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    conn.close()




if __name__ == "__main__":

    main()