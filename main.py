from src.config.database import get_connection

from src.training.train import train_model
from src.training.evaluate import evaluate_model
#from prediction.predict import predict_model


def main():

    conn = get_connection()

    try:

        # Entrenar modelo
        model, X_test, y_test = train_model(conn)

        # Evaluar modelo
        evaluate_model(model, X_test, y_test)


        # Generar predicciones
       # predict_model(conn)

        print("✅ Pipeline ML finalizado correctamente.")

    except Exception as e:

        print("❌ Error:", e)

    finally:

        conn.close()


if __name__ == "__main__":
    main()