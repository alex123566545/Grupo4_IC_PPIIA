from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config.database import get_connection
from src.training.decision_tree import train_model


def main():

    conn = get_connection()

    try:

        # Entrenar modelo
        model, X_test, y_test = train_model(conn)

        # Predicciones
        y_pred = model.predict(X_test)

        # ===============================
        # Métricas
        # ===============================

        print("\n========== MÉTRICAS ==========\n")

        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

        # ===============================
        # Matriz de confusión
        # ===============================

        print("\nMatriz de confusión")
        print(confusion_matrix(y_test, y_pred))

        # ===============================
        # Reporte de clasificación
        # ===============================

        print("\nReporte de clasificación")
        print(classification_report(y_test, y_pred))

    finally:
        conn.close()


if __name__ == "__main__":
    main()