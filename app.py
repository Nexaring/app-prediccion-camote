# =========================================================
# APP DE PREDICCIÓN DE PRODUCCIÓN DE CAMOTE
# STREAMLIT + CATBOOST
# SOLO VARIABLES ORIGINALES DEL DATASET
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="NEXARING - Predicción de Camote",
    page_icon="🌱",
    layout="centered"
)

# =========================================================
# CARGAR MODELO Y ARCHIVOS
# =========================================================
@st.cache_resource
def cargar_modelo():
    modelo = CatBoostRegressor()
    modelo.load_model("modelo_camote.cbm")
    return modelo


@st.cache_data
def cargar_json(nombre_archivo):
    with open(nombre_archivo, "r") as f:
        return json.load(f)


modelo = cargar_modelo()
columnas_modelo = cargar_json("columnas_modelo.json")
rangos = cargar_json("rangos_variables.json")
metricas = cargar_json("metricas_modelo.json")

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def es_variable_ph(col):
    nombre = col.lower()

    return (
        nombre == "ph_suelo" or
        nombre == "phsuelo" or
        "ph" in nombre
    )


def obtener_columnas_riego(columnas):
    columnas_riego = []

    for col in columnas:
        if "riego" in col.lower():
            columnas_riego.append(col)

    return columnas_riego


def aplicar_regla_riego(entrada):
    """
    Si Riego Etapa 1 o Riego Etapa 2 es 0,
    la producción será 0.
    """

    columnas_riego = obtener_columnas_riego(columnas_modelo)

    if len(columnas_riego) < 2:
        st.error("No se encontraron correctamente las columnas de riego.")
        return False

    riego_1 = columnas_riego[0]
    riego_2 = columnas_riego[1]

    if entrada[riego_1] == 0 or entrada[riego_2] == 0:
        return True

    return False


def aplicar_regla_mes(entrada, prediccion):
    """
    Si MesInicioPlantacion es 1, 2, 11 o 12,
    la producción total no puede pasar de 100 quintales.
    """

    mes = int(entrada.get("MesInicioPlantacion", 0))

    if mes in [1, 2, 11, 12]:
        prediccion = min(prediccion, 100)
        return prediccion, True

    return prediccion, False


def clasificar_produccion(prediccion):
    if prediccion >= 300:
        return "Producción alta estimada", "success"
    elif prediccion >= 200:
        return "Producción media estimada", "warning"
    elif prediccion > 0:
        return "Producción baja estimada", "error"
    else:
        return "Producción nula estimada", "error"


# =========================================================
# ENCABEZADO
# =========================================================
st.title("🌱 Sistema de Predicción de Producción de Camote")
st.subheader("NEXARING-Data Science-UTM")

st.markdown(
    """
    Esta aplicación estima la **Producción Total de camote** usando un modelo 
    de inteligencia artificial basado en **CatBoost Regressor**.
    """
)

# =========================================================
# MÉTRICAS DEL MODELO
# =========================================================
with st.expander("📊 Ver métricas del modelo"):
    st.write(f"**R² Score:** {metricas['r2']:.4f}")
    st.write(f"**MAE:** {metricas['mae']:.4f}")
    st.write(f"**MSE:** {metricas['mse']:.4f}")
    st.write(f"**RMSE:** {metricas['rmse']:.4f}")

# =========================================================
# FORMULARIO DE ENTRADA
# =========================================================
st.markdown("## Ingrese los valores")

entrada_usuario = {}

for col in columnas_modelo:
    valor_min = rangos[col]["min"]
    valor_max = rangos[col]["max"]
    valor_mean = rangos[col]["mean"]

    if es_variable_ph(col):
        entrada_usuario[col] = st.slider(
            label=col,
            min_value=float(valor_min),
            max_value=float(valor_max),
            value=float(round(valor_mean, 1)),
            step=0.1
        )
    else:
        entrada_usuario[col] = st.slider(
            label=col,
            min_value=int(round(valor_min)),
            max_value=int(round(valor_max)),
            value=int(round(valor_mean)),
            step=1
        )

# =========================================================
# BOTONES
# =========================================================
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    boton_predecir = st.button(
        "🔍 Predecir producción",
        use_container_width=True
    )

with col2:
    boton_valores_ideales = st.button(
        "⭐ Valores ideales",
        use_container_width=True
    )

if boton_valores_ideales:
    st.info(
        "Los valores iniciales de las barras ya corresponden a los valores promedio del dataset. "
        "Para volver a ellos, actualice la página."
    )

# =========================================================
# PREDICCIÓN
# =========================================================
if boton_predecir:

    entrada_df = pd.DataFrame([entrada_usuario], columns=columnas_modelo)

    # Mostrar el mes seleccionado para comprobar la regla
    if "MesInicioPlantacion" in entrada_usuario:
        st.write("Mes seleccionado:", entrada_usuario["MesInicioPlantacion"])
    else:
        st.error("No se encontró la variable MesInicioPlantacion.")

    # =====================================================
    # REGLA 1: SI FALTA RIEGO EN ETAPA 1 O 2, PRODUCCIÓN = 0
    # =====================================================
    if aplicar_regla_riego(entrada_usuario):
        prediccion = 0.0

        st.error("Producción nula estimada por falta de riego.")
        st.warning("Riego Etapa 1 o Riego Etapa 2 está en 0.")
        st.metric(
            "Predicción de Producción Total",
            f"{prediccion:.2f} quintales"
        )

    else:
        # =================================================
        # PREDICCIÓN DEL MODELO
        # =================================================
        prediccion = modelo.predict(entrada_df)[0]

        if prediccion < 0:
            prediccion = 0

        # =================================================
        # REGLA 2: MESES 1, 2, 11 Y 12 NO PASAN DE 100
        # =================================================
        prediccion, regla_mes_aplicada = aplicar_regla_mes(
            entrada_usuario,
            prediccion
        )

        mensaje, tipo = clasificar_produccion(prediccion)

        if tipo == "success":
            st.success(mensaje)
        elif tipo == "warning":
            st.warning(mensaje)
        else:
            st.error(mensaje)

        if regla_mes_aplicada:
            st.warning(
                "La producción máxima permitida es 100 quintales."
            )

        st.metric(
            "Predicción de Producción Total",
            f"{prediccion:.2f} quintales"
        )

# =========================================================
# INFORMACIÓN FINAL
# =========================================================
st.markdown("---")
st.caption("Desarrollado para predicción agrícola de producción de camote.")
