TEST_SIZE = 0.15

RANDOM_STATE = 42

TARGET = "tasa_mortalidad_semana_siguiente_pct"

FEATURES = [
    #"id_lote",
    "distrito",
    "altitud_msnm",
    "categoria_zootecnica",
    "raza_predominante",
    "tamano_lote_cabezas",
    "lote_sensorizado",
    "distancia_centro_veterinario_km",
    "uso_registro_digital",
    "cobertura_vacunacion_pct",
    "dias_desde_desparasitacion",
    "animales_nuevos_30d",
    "visitas_tecnicas_mes",
    "casos_respiratorios",
    "casos_diarreicos",
    "temperatura_min_c",
    "temperatura_media_c",
    "temperatura_max_c",
    "humedad_relativa_pct",
    "precipitacion_semanal_mm",
    "condicion_pastura_indice",
    "indice_ndvi_satelital",
    "consumo_ms_kg_animal_dia",
    "agua_l_animal_dia",
    "actividad_sensor_indice",
    "flag_actividad_sensor_imputado",
    "condicion_corporal_prom",
    "flag_condicion_corporal_imputado",
    "precio_leche_local_s_kg",
    "amplitud_termica",
    "riesgo_climatico",
    "indice_salud_lote",
    "mes",
    "temporada",
    "semana_sin",
    "semana_cos",
    #"semana_anio",                        # reemplazada por codificación cíclica
    #"muertes_semana_siguiente",           # fuga: mismo target en otra forma
    #"riesgo_mortalidad_alta_semana_siguiente",  # fuga: mismo target en otra forma
]

MODEL_PATH = "models/model.pkl"

ENCODERS_PATH = "models/encoders.pkl"