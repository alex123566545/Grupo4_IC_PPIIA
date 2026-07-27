# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
# Se importan las métricas necesarias para evaluar el desempeño
# del modelo de clasificación. Estas métricas permiten medir
# distintos aspectos del rendimiento del modelo una vez entrenado.

from sklearn.metrics import (
    accuracy_score,          # Calcula la exactitud (Accuracy)
    precision_score,         # Calcula la precisión (Precision)
    recall_score,            # Calcula el Recall o Sensibilidad
    f1_score,                # Calcula el F1 Score
    roc_auc_score,           # Calcula el área bajo la curva ROC (AUC)
    confusion_matrix,        # Genera la matriz de confusión
    classification_report    # Genera un reporte completo de clasificación
)

# Librería utilizada para operaciones numéricas.
# En este caso únicamente se utiliza para generar
# una secuencia de posibles umbrales.
import numpy as np

# ==========================================================
# IMPORTACIÓN DE MÓDULOS DEL PROYECTO
# ==========================================================

# Obtiene la conexión con la base de datos.
from src.config.database import get_connection

# Entrena el modelo utilizando los hiperparámetros previamente
# seleccionados mediante Random Search.
from src.training.train_random_search import train_model

# Ejecuta una validación mediante GroupKFold para comprobar
# la capacidad de generalización del modelo sobre lotes nunca vistos.
from src.training.evaluate_groupkfold_random_search import (
    evaluar_groupkfold_hiperparametros_fijos
)


# ==========================================================
# FUNCIÓN: buscar_mejor_umbral()
# ==========================================================
def buscar_mejor_umbral(
        y_real,
        probabilidades,
        recall_minimo=0.60
):
    """
    Busca automáticamente el mejor umbral de decisión para convertir
    las probabilidades generadas por el modelo en clases binarias.

    En clasificación binaria, muchos modelos generan probabilidades
    en lugar de responder directamente "Sí" o "No". Por ello es
    necesario establecer un umbral para decidir cuándo una observación
    será considerada positiva.

    Parámetros
    ----------
    y_real : array-like
        Valores reales de la variable objetivo.

    probabilidades : array-like
        Probabilidades estimadas por el modelo para la clase positiva.

    recall_minimo : float, opcional
        Recall mínimo aceptable definido como meta del proyecto.
        Por defecto es 0.60.

    Retorna
    -------
    tuple
        Devuelve:

        - mejor_umbral
        - mejor_precision
        - mejor_recall
        - mejor_f1
        - resultados (lista con las métricas para todos los umbrales)

    ----------------------------------------------------------
    CRITERIOS UTILIZADOS
    ----------------------------------------------------------

    1. Recall >= recall_minimo (meta obligatoria)

    2. Entre todos los umbrales que cumplen ese requisito,
       seleccionar el que tenga la mayor Precision.

    3. Si existen varios candidatos con la misma Precision,
       escoger aquel cuyo F1 Score sea mayor.

    ----------------------------------------------------------
    NOTA IMPORTANTE
    ----------------------------------------------------------

    El proyecto define Recall >= 0.60 como requisito obligatorio.

    Esto significa que el modelo debe identificar al menos
    el 60 % de los casos positivos.

    La Precision es importante y se intenta maximizar,
    pero nunca sacrificando el Recall por debajo de dicha meta.

    Una versión anterior realizaba el proceso inverso,
    priorizando Precision >= 0.70, lo que ocasionaba
    modelos con muy pocos falsos positivos pero demasiados
    falsos negativos.
    """

    # ------------------------------------------------------
    # Variables donde se almacenará el mejor resultado
    # encontrado durante la búsqueda.
    # ------------------------------------------------------

    mejor_umbral = None

    mejor_precision = -1

    mejor_recall = 0

    mejor_f1 = -1

    # Lista donde se almacenan los resultados obtenidos
    # para cada umbral evaluado.
    resultados = []

    # ------------------------------------------------------
    # Se prueban umbrales desde 0.30 hasta 0.90 con un
    # incremento de 0.01.
    # ------------------------------------------------------

    for umbral in np.arange(
        0.30,
        0.91,
        0.01
    ):

        # Convierte las probabilidades en clases binarias.
        #
        # Si la probabilidad es mayor o igual que el umbral,
        # la observación será clasificada como positiva (1).
        # En caso contrario será clasificada como negativa (0).
        predicciones = (
            probabilidades >= umbral
        ).astype(int)

        # --------------------------------------------------
        # Cálculo de métricas para el umbral actual.
        # --------------------------------------------------

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

        # Se almacenan las métricas correspondientes al
        # umbral actual para mostrarlas posteriormente.
        resultados.append({

            "umbral": umbral,

            "precision": precision,

            "recall": recall,

            "f1": f1

        })

        # --------------------------------------------------
        # Solo se consideran los umbrales que cumplen
        # el Recall mínimo establecido por el proyecto.
        # --------------------------------------------------

        if recall >= recall_minimo:

            # Si la Precision es mayor que la mejor encontrada,
            # se actualiza automáticamente el mejor umbral.
            #
            # En caso de empate en Precision,
            # se utiliza el F1 Score como criterio de desempate.
            if (

                precision > mejor_precision

                or

                (

                    precision == mejor_precision

                    and

                    f1 > mejor_f1

                )

            ):

                mejor_umbral = umbral

                mejor_precision = precision

                mejor_recall = recall

                mejor_f1 = f1

    # ------------------------------------------------------
    # Si ningún umbral logra alcanzar el Recall solicitado,
    # se selecciona el que obtiene el Recall más alto
    # disponible.
    #
    # De esta manera el programa informa que el modelo
    # todavía no cumple la meta definida para el proyecto.
    # ------------------------------------------------------

    if mejor_umbral is None:

        mejor = max(

            resultados,

            key=lambda x: x["recall"]

        )

        mejor_umbral = mejor["umbral"]

        mejor_precision = mejor["precision"]

        mejor_recall = mejor["recall"]

        mejor_f1 = mejor["f1"]

        print("\n⚠ Ningún umbral alcanzó el Recall solicitado.")
        print("Se utilizará el umbral con mayor Recall disponible.")
        print("(el modelo puede necesitar ajustes -- este resultado")
        print(" no cumple la meta del proyecto tal como está)\n")

    # ------------------------------------------------------
    # Retorna toda la información necesaria para continuar
    # con la evaluación del modelo.
    # ------------------------------------------------------

    return (

        mejor_umbral,

        mejor_precision,

        mejor_recall,

        mejor_f1,

        resultados

    )
# ==========================================================
# FUNCIÓN PRINCIPAL DEL PROGRAMA
# ==========================================================
def main():

    # ======================================================
    # ESTABLECER CONEXIÓN CON LA BASE DE DATOS
    # ======================================================
    #
    # Se crea una conexión con la base de datos desde la cual
    # se obtendrán los registros utilizados para entrenar y
    # evaluar el modelo de Machine Learning.
    #
    # Esta conexión permanecerá abierta durante toda la
    # ejecución y será cerrada al finalizar el programa.
    #

    conn = get_connection()

    # ======================================================
    # ENTRENAMIENTO DEL MODELO
    # ======================================================
    #
    # Se entrena el modelo utilizando los hiperparámetros
    # previamente optimizados mediante Random Search.
    #
    # La función devuelve:
    #
    # model  -> Modelo ya entrenado.
    # X_test -> Variables predictoras del conjunto de prueba.
    # y_test -> Valores reales del conjunto de prueba.
    #

    model, X_test, y_test = train_model(conn)

    # ======================================================
    # OBTENCIÓN DE LAS PROBABILIDADES
    # ======================================================
    #
    # En lugar de utilizar directamente las clases predichas,
    # el modelo devuelve la probabilidad de pertenecer a cada
    # una de las clases.
    #
    # predict_proba() retorna dos columnas:
    #
    # Columna 0 -> Probabilidad de clase negativa.
    # Columna 1 -> Probabilidad de clase positiva.
    #
    # Para este proyecto únicamente interesa la probabilidad
    # de la clase positiva (índice 1).
    #

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # ======================================================
    # BÚSQUEDA AUTOMÁTICA DEL MEJOR UMBRAL
    # ======================================================
    #
    # Una vez obtenidas las probabilidades se evalúan varios
    # umbrales de decisión con el objetivo de encontrar el
    # que mejor se adapta a la meta del proyecto.
    #
    # El criterio utilizado consiste en:
    #
    # 1. Cumplir Recall >= 0.60.
    # 2. Maximizar la Precision.
    # 3. Desempatar utilizando el F1 Score.
    #

    (
        mejor_umbral,
        mejor_precision,
        mejor_recall,
        mejor_f1,
        resultados
    ) = buscar_mejor_umbral(

        y_test,

        y_prob,

        recall_minimo=0.60

    )

    # ======================================================
    # MOSTRAR EL RENDIMIENTO DE CADA UMBRAL EVALUADO
    # ======================================================
    #
    # Se imprime una tabla con las métricas obtenidas para
    # todos los umbrales evaluados.
    #
    # Esto permite analizar cómo cambia el comportamiento del
    # modelo al aumentar o disminuir el umbral de decisión.
    #

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

    # ======================================================
    # MOSTRAR EL MEJOR UMBRAL ENCONTRADO
    # ======================================================
    #
    # Después de evaluar todos los candidatos se muestran las
    # métricas correspondientes al umbral seleccionado.
    #

    print("\n")
    print("=" * 90)
    print("MEJOR UMBRAL ENCONTRADO")
    print("=" * 90)

    print(f"Umbral seleccionado : {mejor_umbral:.2f}")
    print(f"Precision           : {mejor_precision:.3f}")
    print(f"Recall              : {mejor_recall:.3f}")
    print(f"F1 Score            : {mejor_f1:.3f}")

    print("\nCriterio utilizado:")

    print("1. Recall >= 0.60 (piso, meta del proyecto)")

    print("2. Mayor Precision")

    print("3. Mayor F1 (desempate)")

    # ======================================================
    # GENERACIÓN DE LAS PREDICCIONES FINALES
    # ======================================================
    #
    # Se convierte cada probabilidad en una clase binaria
    # utilizando el mejor umbral encontrado durante la etapa
    # anterior.
    #
    # Probabilidad >= umbral -> Clase positiva.
    # Probabilidad <  umbral -> Clase negativa.
    #

    y_pred = (

        y_prob >= mejor_umbral

    ).astype(int)

    # ======================================================
    # CÁLCULO DE LAS MÉTRICAS FINALES
    # ======================================================
    #
    # Una vez obtenidas las predicciones definitivas se
    # calculan las métricas que permitirán evaluar el
    # rendimiento global del modelo.
    #
    # Accuracy  -> Porcentaje total de aciertos.
    # Precision -> Exactitud de las predicciones positivas.
    # Recall    -> Capacidad para detectar positivos.
    # F1 Score  -> Balance entre Precision y Recall.
    # AUC-ROC   -> Capacidad discriminativa del modelo.
    #

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

    # ======================================================
    # CÁLCULO DEL ÁREA BAJO LA CURVA ROC
    # ======================================================
    #
    # A diferencia de las demás métricas, el AUC se calcula
    # utilizando las probabilidades originales generadas por
    # el modelo y no las clases binarias.
    #

    auc = roc_auc_score(

        y_test,

        y_prob

    )

    # ======================================================
    # MATRIZ DE CONFUSIÓN
    # ======================================================
    #
    # Resume la cantidad de:
    #
    # TP -> Verdaderos Positivos
    # FP -> Falsos Positivos
    # FN -> Falsos Negativos
    # TN -> Verdaderos Negativos
    #
    # Esta matriz constituye la base para el cálculo de
    # prácticamente todas las métricas anteriores.
    #

    matriz = confusion_matrix(

        y_test,

        y_pred

    )

        # ======================================================
    # MOSTRAR RESULTADOS FINALES DEL MODELO
    # ======================================================
    #
    # Una vez calculadas todas las métricas se muestran por
    # pantalla para facilitar la interpretación del desempeño
    # obtenido con el mejor umbral encontrado.
    #

    print("\n")
    print("=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    # ======================================================
    # MOSTRAR MATRIZ DE CONFUSIÓN
    # ======================================================
    #
    # La matriz de confusión permite observar el número de
    # predicciones correctas e incorrectas realizadas por el
    # modelo.
    #
    #                Predicción
    #              0            1
    #
    # Real 0      TN           FP
    # Real 1      FN           TP
    #
    # A partir de esta matriz pueden calcularse la mayoría de
    # métricas de clasificación.
    #

    print("\n")
    print("=" * 60)
    print("MATRIZ DE CONFUSIÓN")
    print("=" * 60)

    print(matriz)

    # ======================================================
    # DESCOMPONER LA MATRIZ DE CONFUSIÓN
    # ======================================================
    #
    # ravel() convierte la matriz de 2x2 en un vector para
    # extraer fácilmente cada uno de sus componentes.
    #

    tn, fp, fn, tp = matriz.ravel()

    print("\nDetalle:")

    print(f"TP (Verdaderos Positivos): {tp}")

    print(f"FP (Falsos Positivos)    : {fp}")

    print(f"FN (Falsos Negativos)    : {fn}")

    print(f"TN (Verdaderos Negativos): {tn}")

    # ======================================================
    # REPORTE DE CLASIFICACIÓN
    # ======================================================
    #
    # classification_report() genera un resumen completo con:
    #
    # - Precision
    # - Recall
    # - F1 Score
    # - Support
    #
    # para cada clase del problema.
    #

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

    # ======================================================
    # RESUMEN EJECUTIVO
    # ======================================================
    #
    # En esta sección se verifica automáticamente si el modelo
    # cumple las metas establecidas para el proyecto.
    #
    # Se consideran dos objetivos:
    #
    # Recall >= 0.60
    # Precision >= 0.70
    #
    # El Recall representa el requisito mínimo obligatorio
    # definido durante la planificación del proyecto.
    #
    # La Precision constituye una meta adicional que se busca
    # alcanzar siempre que no comprometa el Recall.
    #

    print("\n")
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    # ======================================================
    # VERIFICACIÓN DE LAS METAS DEL PROYECTO
    # ======================================================
    #
    # Se evalúan las métricas finales obtenidas para informar
    # automáticamente si el modelo cumple o no los objetivos.
    #

    if recall >= 0.60 and precision >= 0.70:

        print("✅ Se alcanzaron ambas metas: Recall >= 0.60 y Precision >= 0.70.")

    elif recall >= 0.60:

        print("✅ Se alcanzó la meta de Recall (>= 0.60).")
        print("⚠️  Precision quedó por debajo de 0.70.")

    else:

        print("❌ No se alcanzó la meta de Recall del proyecto (>= 0.60).")

    # ======================================================
    # MOSTRAR LAS MÉTRICAS DEFINITIVAS
    # ======================================================
    #
    # Se presentan nuevamente las métricas principales para
    # facilitar la elaboración de informes o documentación del
    # proyecto.
    #

    print(f"Umbral seleccionado : {mejor_umbral:.2f}")
    print(f"Precision final     : {precision:.4f}")
    print(f"Recall final        : {recall:.4f}")
    print(f"F1 Score final      : {f1:.4f}")
    print(f"AUC-ROC             : {auc:.4f}")

    print("=" * 60)

    # ======================================================
    # VALIDACIÓN MEDIANTE GROUPKFOLD
    # ======================================================
    #
    # Aunque las métricas obtenidas sobre el conjunto de prueba
    # sean satisfactorias, todavía es necesario comprobar la
    # capacidad de generalización del modelo.
    #
    # Para ello se ejecuta una validación GroupKFold utilizando
    # los lotes como grupos.
    #
    # En cada iteración:
    #
    # • Un lote completo se reserva para prueba.
    # • Los demás lotes se utilizan para entrenamiento.
    #
    # Esto permite simular el comportamiento del modelo frente
    # a datos completamente nuevos, evitando que existan
    # registros del mismo lote tanto en entrenamiento como en
    # prueba.
    #
    # Esta validación proporciona una estimación mucho más
    # realista del desempeño esperado en producción.
    #

    print("\n")
    print("=" * 60)
    print("VALIDACIÓN GROUPKFOLD (generalización a lote nuevo)")
    print("=" * 60)

    evaluar_groupkfold_hiperparametros_fijos(conn)

    # ======================================================
    # CIERRE DE LA CONEXIÓN
    # ======================================================
    #
    # Una vez finalizado todo el proceso se libera la conexión
    # con la base de datos para evitar mantener recursos
    # abiertos innecesariamente.
    #

    conn.close()


# ==========================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# ==========================================================
#
# Esta condición garantiza que la función main() únicamente
# se ejecute cuando el archivo sea iniciado directamente.
#
# Si el archivo es importado desde otro módulo, el código no
# se ejecutará automáticamente.
#

if __name__ == "__main__":

    main()