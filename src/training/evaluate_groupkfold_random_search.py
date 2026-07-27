# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
#
# Se importan las librerías necesarias para el procesamiento
# de datos, entrenamiento del modelo y evaluación mediante
# validación cruzada agrupada (GroupKFold).
#

# Librerías para manejo de datos y operaciones matemáticas.
import numpy as np
import pandas as pd

# ==========================================================
# IMPORTACIÓN DEL MODELO
# ==========================================================
#
# Se utiliza RandomForestClassifier debido a que fue el modelo
# seleccionado durante la etapa de experimentación por ofrecer
# el mejor equilibrio entre rendimiento y capacidad de
# generalización.
#

from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# VALIDACIÓN CRUZADA
# ==========================================================
#
# GroupKFold garantiza que un mismo grupo (en este proyecto,
# un lote) nunca aparezca simultáneamente en entrenamiento y
# prueba.
#
# Esto permite medir la capacidad real del modelo para
# generalizar sobre lotes completamente nuevos.
#

from sklearn.model_selection import GroupKFold

# ==========================================================
# PREPROCESAMIENTO
# ==========================================================
#
# LabelEncoder convierte variables categóricas en valores
# numéricos para que puedan ser procesadas por el algoritmo
# Random Forest.
#

from sklearn.preprocessing import LabelEncoder

# ==========================================================
# MÉTRICAS DE EVALUACIÓN
# ==========================================================
#
# Se importan las métricas utilizadas para evaluar el modelo
# durante cada fold de la validación.
#

from sklearn.metrics import (

    # Área bajo la curva ROC.
    roc_auc_score,

    # Precisión del modelo.
    precision_score,

    # Sensibilidad o Recall.
    recall_score,

    # Media armónica entre Precision y Recall.
    f1_score,

    # Matriz de confusión.
    confusion_matrix,

)

# ==========================================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================================
#
# Se importan las constantes generales utilizadas durante el
# entrenamiento y evaluación.
#
# FEATURES
#     Lista de variables predictoras.
#
# TARGET
#     Variable objetivo.
#
# RANDOM_STATE
#     Semilla para garantizar reproducibilidad.
#

from src.config.settings import FEATURES, TARGET, RANDOM_STATE

# ==========================================================
# HIPERPARÁMETROS DEL MODELO
# ==========================================================
#
# Estos hiperparámetros corresponden al mejor modelo obtenido
# mediante RandomizedSearchCV durante la etapa de optimización.
#
# En este script NO se vuelve a ejecutar Random Search debido
# a su elevado costo computacional.
#
# El objetivo aquí no es buscar un nuevo modelo, sino validar
# si el modelo previamente seleccionado mantiene su rendimiento
# cuando se enfrenta a lotes completamente nuevos.
#
# De esta manera se obtiene una estimación mucho más realista
# del comportamiento esperado en producción.
#

MEJORES_HIPERPARAMETROS = dict(

    # Número de árboles del bosque.
    n_estimators=815,

    # Profundidad máxima permitida para cada árbol.
    max_depth=15,

    # Función utilizada para medir la calidad de las divisiones.
    criterion="entropy",

    # Número mínimo de muestras para dividir un nodo.
    min_samples_split=6,

    # Número mínimo de muestras permitido en una hoja.
    min_samples_leaf=8,

    # Número de variables consideradas en cada división.
    max_features=None,

    # Utilizar Bootstrap Sampling.
    bootstrap=True,

    # Porcentaje de registros utilizados por árbol.
    max_samples=0.7,

    # Sin ponderación especial de clases.
    class_weight=None,

    # Parámetro de poda del árbol.
    ccp_alpha=0.005,

    # Semilla aleatoria.
    random_state=RANDOM_STATE,

    # Utilizar todos los núcleos disponibles.
    n_jobs=-1,

)


# ==========================================================
# FUNCIÓN PRINCIPAL DE VALIDACIÓN
# ==========================================================
def evaluar_groupkfold_hiperparametros_fijos(
        conn,
        n_splits=5,
        recall_minimo=0.60
):
    """
    Evalúa la capacidad de generalización del modelo mediante
    GroupKFold utilizando los lotes como grupos.

    A diferencia de una validación aleatoria tradicional,
    GroupKFold garantiza que los registros pertenecientes a un
    mismo lote nunca aparezcan simultáneamente en entrenamiento
    y prueba.

    Esto evita la fuga de información (Data Leakage) y permite
    conocer con mayor precisión cómo se comportará el modelo
    frente a datos completamente nuevos.

    Parámetros
    ----------
    conn
        Conexión activa a la base de datos.

    n_splits : int
        Número de folds utilizados durante la validación.

    recall_minimo : float
        Recall mínimo aceptado como meta del proyecto.

    Retorna
    -------
    pandas.DataFrame

        Tabla con las métricas obtenidas en cada fold.

    Durante cada iteración se realiza el siguiente proceso:

    1. Separar un lote completo para prueba.

    2. Entrenar nuevamente el modelo utilizando únicamente
       los lotes restantes.

    3. Obtener probabilidades.

    4. Buscar el mejor umbral de decisión.

    5. Calcular las métricas correspondientes.

    Finalmente se calcula el promedio entre todos los folds,
    proporcionando una estimación mucho más confiable del
    rendimiento esperado del modelo.
    """

    # ======================================================
    # LECTURA DE LOS DATOS
    # ======================================================
    #
    # Se recupera desde la base de datos el conjunto final de
    # variables utilizado para el entrenamiento del modelo.
    #

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    # ======================================================
    # ELIMINACIÓN DE REGISTROS SIN VARIABLE OBJETIVO
    # ======================================================
    #
    # Algunos registros no poseen un valor válido para la
    # variable objetivo debido a que no cuentan con suficiente
    # información futura para construir el target.
    #
    # Dichos registros son eliminados antes de iniciar la
    # validación.
    #

    filas_antes = len(df)

    df = df[df[TARGET].notna()].copy()

    print(f"ℹ️  Filas excluidas por target nulo: {filas_antes - len(df)}")

    # ======================================================
    # TRANSFORMACIÓN CÍCLICA
    # ======================================================
    #
    # La semana del año posee un comportamiento cíclico.
    #
    # La semana 52 y la semana 1 están realmente muy próximas,
    # aunque numéricamente parezcan muy alejadas.
    #
    # Para representar correctamente esta relación se crean
    # dos variables utilizando funciones trigonométricas.
    #

    df["semana_sin"] = np.sin(
        2 * np.pi * df["semana_anio"] / 52
    )

    df["semana_cos"] = np.cos(
        2 * np.pi * df["semana_anio"] / 52
    )

    # ======================================================
    # SEPARACIÓN DE VARIABLES
    # ======================================================
    #
    # X
    #     Variables predictoras.
    #
    # y
    #     Variable objetivo.
    #
    # grupos
    #     Identificador del lote utilizado por GroupKFold.
    #

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int).copy()

    grupos = df["id_lote"]

    # ======================================================
    # VALIDACIÓN DEL NÚMERO DE FOLDS
    # ======================================================
    #
    # GroupKFold requiere que el número de folds sea menor o
    # igual al número de grupos disponibles.
    #
    # Si el usuario solicita más folds que lotes existentes,
    # el programa ajusta automáticamente dicho valor.
    #

    n_lotes = grupos.nunique()

    if n_splits > n_lotes:

        print(
            f"⚠️  Pediste {n_splits} folds pero solo hay {n_lotes} lotes. "
            f"Usando n_splits={n_lotes} (1 lote por fold)."
        )

        n_splits = n_lotes

        # ======================================================
    # CODIFICACIÓN DE VARIABLES CATEGÓRICAS
    # ======================================================
    #
    # Random Forest únicamente puede trabajar con variables
    # numéricas, por lo que todas las variables categóricas
    # deben convertirse previamente en valores enteros.
    #
    # Se utiliza LabelEncoder porque este mismo procedimiento
    # fue aplicado durante el entrenamiento del modelo
    # (train_random_search.py).
    #
    # IMPORTANTE:
    #
    # El encoder se ajusta utilizando TODO el conjunto de
    # datos antes de iniciar la validación.
    #
    # Esto no representa fuga de información porque el encoder
    # únicamente aprende el nombre de las categorías y no
    # utiliza información de la variable objetivo.
    #
    # Ejemplo:
    #
    # "Norte" → 0
    # "Centro" → 1
    # "Sur" → 2
    #

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns

    for col in categorical_columns:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col].astype(str))

    # ======================================================
    # CREACIÓN DEL GROUPKFOLD
    # ======================================================
    #
    # GroupKFold divide los datos respetando completamente los
    # grupos definidos mediante la variable "id_lote".
    #
    # Esto significa que:
    #
    # • Ningún registro perteneciente al lote de prueba puede
    #   aparecer durante el entrenamiento.
    #
    # • Cada lote será utilizado exactamente una vez como
    #   conjunto de prueba.
    #
    # Esta estrategia permite evaluar la verdadera capacidad
    # de generalización del modelo.
    #

    gkf = GroupKFold(n_splits=n_splits)

    # ======================================================
    # LISTA DONDE SE ALMACENARÁN LOS RESULTADOS DE CADA FOLD
    # ======================================================

    resultados = []

    # ======================================================
    # VALIDACIÓN CRUZADA
    # ======================================================
    #
    # En cada iteración ocurre el siguiente proceso:
    #
    # 1. Se seleccionan uno o varios lotes completos para
    #    prueba.
    #
    # 2. El resto de los lotes forman el conjunto de
    #    entrenamiento.
    #
    # 3. Se entrena nuevamente el Random Forest.
    #
    # 4. Se calculan probabilidades.
    #
    # 5. Se busca automáticamente el mejor umbral.
    #
    # 6. Se calculan las métricas correspondientes.
    #

    for fold_i, (train_idx, test_idx) in enumerate(

            gkf.split(
                X,
                y,
                groups=grupos
            ),

            start=1

    ):

        # ==================================================
        # SEPARACIÓN DE LOS DATOS DEL FOLD ACTUAL
        # ==================================================

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Lista de lotes utilizados como prueba.
        lotes_test = sorted(
            grupos.iloc[test_idx].unique()
        )

        # ==================================================
        # CREACIÓN DEL MODELO
        # ==================================================
        #
        # En cada fold se crea un nuevo Random Forest
        # utilizando exactamente los mismos hiperparámetros
        # obtenidos durante Randomized Search.
        #
        # De esta forma la única diferencia entre folds son
        # los datos utilizados para entrenamiento y prueba.
        #

        model = RandomForestClassifier(

            **MEJORES_HIPERPARAMETROS

        )

        # Entrenamiento del modelo.

        model.fit(

            X_train,

            y_train

        )

        # ==================================================
        # VALIDACIÓN DE LAS CLASES DEL CONJUNTO DE PRUEBA
        # ==================================================
        #
        # El cálculo del AUC requiere la presencia de ambas
        # clases.
        #
        # Si el conjunto de prueba contiene únicamente una
        # clase, dicha métrica no puede calcularse.
        #

        if y_test.nunique() < 2:

            print(

                f"Fold {fold_i} | lotes test={lotes_test} | "

                f"n_test={len(X_test)} "

                f"(todos clase {y_test.iloc[0]}) | "

                f"AUC no calculable (una sola clase en test)"

            )

            resultados.append({

                "fold": fold_i,

                "lotes_test": lotes_test,

                "auc": np.nan,

                "umbral": np.nan,

                "precision": np.nan,

                "recall": np.nan,

                "f1": np.nan,

            })

            continue

        # ==================================================
        # OBTENCIÓN DE LAS PROBABILIDADES
        # ==================================================
        #
        # Se utiliza la probabilidad de pertenecer a la clase
        # positiva para calcular posteriormente el mejor
        # umbral de clasificación.
        #

        proba = model.predict_proba(

            X_test

        )[:, 1]

        # ==================================================
        # CÁLCULO DEL AUC
        # ==================================================
        #
        # El AUC mide la capacidad del modelo para diferenciar
        # correctamente ambas clases independientemente del
        # umbral utilizado.
        #

        auc = roc_auc_score(

            y_test,

            proba

        )

        # ==================================================
        # BÚSQUEDA DEL MEJOR UMBRAL
        # ==================================================
        #
        # Se evalúan múltiples umbrales comprendidos entre
        # 0.30 y 0.90.
        #
        # Para cada uno se calculan:
        #
        # • Precision
        # • Recall
        # • F1 Score
        #
        # El criterio utilizado es exactamente el mismo que
        # en buscar_mejor_umbral.py:
        #
        # 1. Cumplir Recall mínimo.
        #
        # 2. Maximizar Precision.
        #
        # 3. Desempatar utilizando F1.
        #

        mejor_umbral = None

        mejor_precision = -1

        mejor_recall = 0

        mejor_f1 = -1

        for umbral in np.arange(

                0.30,

                0.91,

                0.01

        ):

            # Conversión de probabilidades en clases binarias.
            pred = (

                proba >= umbral

            ).astype(int)

            # Cálculo de métricas para el umbral actual.

            p = precision_score(

                y_test,

                pred,

                zero_division=0

            )

            r = recall_score(

                y_test,

                pred,

                zero_division=0

            )

            f1v = f1_score(

                y_test,

                pred,

                zero_division=0

            )

            # Actualización del mejor umbral encontrado.

            if (

                r >= recall_minimo

                and

                (

                    p > mejor_precision

                    or

                    (

                        p == mejor_precision

                        and

                        f1v > mejor_f1

                    )

                )

            ):

                mejor_umbral = umbral

                mejor_precision = p

                mejor_recall = r

                mejor_f1 = f1v