# =========================================================
# APP DE PREDICCIÓN DE PRODUCCIÓN DE CAMOTE
# STREAMLIT + CATBOOST
# VERSIÓN CORREGIDA PARA MODELO ENTRENADO CON DATOS NORMALIZADOS
# =========================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from streamlit_geolocation import streamlit_geolocation


# =========================================================
# 1. CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Predicción de producción de camote",
    page_icon="🌱",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODELO_FILE = BASE_DIR / "modelo_camote.cbm"
COLUMNAS_FILE = BASE_DIR / "columnas_modelo.json"
RANGOS_FILE = BASE_DIR / "rangos_variables.json"
METRICAS_FILE = BASE_DIR / "metricas_modelo.json"

# IMPORTANTE:
# El modelo fue entrenado con Producción_Total normalizada.
# Estos valores corresponden al rango de la nueva base utilizada.
# Si posteriormente guardas el rango del objetivo en JSON, el programa
# lo utilizará automáticamente.
TARGET_MIN = 0.0
TARGET_MAX = 39228.0


# =========================================================
# 2. CARGA DE ARCHIVOS
# =========================================================

@st.cache_resource
def cargar_modelo():
    if not MODELO_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODELO_FILE.name}"
        )

    modelo = CatBoostRegressor()
    modelo.load_model(str(MODELO_FILE))
    return modelo


@st.cache_data
def cargar_json(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta.name}"
        )

    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    modelo = cargar_modelo()
    columnas_modelo = cargar_json(COLUMNAS_FILE)
    rangos = cargar_json(RANGOS_FILE)

    if METRICAS_FILE.exists():
        metricas = cargar_json(METRICAS_FILE)
    else:
        metricas = {}

except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")
    st.stop()


# =========================================================
# 3. VALIDACIÓN DE VARIABLES
# =========================================================

columnas_esperadas = [
    "MesInicioPlantacion",
    "DuracionPlantacion_dias",
    "Fertilizante",
    "Riego_Etapa1_mm",
    "Riego_Etapa2_mm",
    "Riego_Etapa3_mm",
    "Temperatura_C",
    "Precipitacion_mm",
    "Humedad_porcentaje",
    "Altitud_msnm",
    "TipoSuelo",
    "pH_Suelo"
]

faltantes = [
    col for col in columnas_esperadas
    if col not in columnas_modelo
]

if faltantes:
    st.error(
        "El modelo no contiene las variables esperadas: "
        + ", ".join(faltantes)
    )
    st.stop()

# Se respeta EXACTAMENTE el orden utilizado por el modelo.
columnas_modelo = list(columnas_modelo)


# =========================================================
# 4. NOMBRES AMIGABLES
# =========================================================

nombres_amigables = {
    "MesInicioPlantacion": "Mes de inicio de plantación",
    "DuracionPlantacion_dias": "Duración del cultivo",
    "Fertilizante": "Fertilizante",
    "Riego_Etapa1_mm": "Riego en etapa inicial",
    "Riego_Etapa2_mm": "Riego en etapa intermedia",
    "Riego_Etapa3_mm": "Riego en etapa final",
    "Temperatura_C": "Temperatura media",
    "Precipitacion_mm": "Precipitación acumulada",
    "Humedad_porcentaje": "Humedad relativa",
    "Altitud_msnm": "Altitud del terreno",
    "TipoSuelo": "Tipo de suelo",
    "pH_Suelo": "pH del suelo"
}


def obtener_nombre_amigable(col):
    return nombres_amigables.get(col, col)


# =========================================================
# 5. VALORES AGRONÓMICOS DE REFERENCIA
# =========================================================

valores_ideales = {
    "MesInicioPlantacion": 6,
    "DuracionPlantacion_dias": 90,
    "Temperatura_C": 26,
    "Precipitacion_mm": 564,
    "Humedad_porcentaje": 70,
    "Fertilizante": 80,
    "Riego_Etapa1_mm": 35,
    "Riego_Etapa2_mm": 45,
    "Riego_Etapa3_mm": 55,
    "TipoSuelo": 1,
    "pH_Suelo": 6,
    "Altitud_msnm": 250
}

rangos_ideales = {
    "MesInicioPlantacion": [1, 12],
    "DuracionPlantacion_dias": [60, 120],
    "Temperatura_C": [10, 40],
    "Precipitacion_mm": [200, 1200],
    "Humedad_porcentaje": [20, 100],
    "Fertilizante": [0, 100],
    "Riego_Etapa1_mm": [0, 60],
    "Riego_Etapa2_mm": [0, 70],
    "Riego_Etapa3_mm": [0, 80],
    "TipoSuelo": [1, 3],
    "pH_Suelo": [4, 6],
    "Altitud_msnm": [0, 1500]
}


# =========================================================
# 6. NORMALIZACIÓN
# =========================================================

def normalizar_variable(variable, valor):
    """
    Normaliza una variable mediante Min-Max:
        x_norm = (x - min) / (max - min)

    Debe coincidir con el procedimiento utilizado para generar
    Data_Normalizada.csv.
    """

    if variable not in rangos:
        raise KeyError(
            f"No existe información de rango para '{variable}'."
        )

    minimo = float(rangos[variable]["min"])
    maximo = float(rangos[variable]["max"])

    if maximo == minimo:
        return 0.0

    valor = float(valor)

    valor_normalizado = (valor - minimo) / (maximo - minimo)

    # Evita valores fuera del intervalo de entrenamiento.
    valor_normalizado = max(0.0, min(1.0, valor_normalizado))

    return valor_normalizado


def normalizar_entrada(entrada):
    """
    Convierte los valores introducidos en la aplicación a la misma
    escala utilizada durante el entrenamiento del CatBoost.
    """

    fila = {}

    for variable in columnas_modelo:
        fila[variable] = normalizar_variable(
            variable,
            entrada[variable]
        )

    return pd.DataFrame(
        [fila],
        columns=columnas_modelo
    )


# =========================================================
# 7. DESNORMALIZAR PRODUCCIÓN
# =========================================================

def obtener_rango_objetivo():
    """
    Busca primero el rango de Producción_Total dentro de
    rangos_variables.json. Si no existe, utiliza los valores
    definidos arriba para la nueva base.
    """

    posibles_nombres = [
        "Produccion_Total",
        "Producción_Total",
        "ProduccionTotal",
        "y"
    ]

    for nombre in posibles_nombres:
        if nombre in rangos:
            if (
                isinstance(rangos[nombre], dict)
                and "min" in rangos[nombre]
                and "max" in rangos[nombre]
            ):
                return (
                    float(rangos[nombre]["min"]),
                    float(rangos[nombre]["max"])
                )

    return TARGET_MIN, TARGET_MAX


def desnormalizar_produccion(valor_normalizado):
    """
    Convierte la predicción Min-Max nuevamente a kg/ha.
    """

    minimo, maximo = obtener_rango_objetivo()

    return (
        float(valor_normalizado) * (maximo - minimo)
        + minimo
    )


# =========================================================
# 8. GEOLOCALIZACIÓN + TEMPERATURA + ALTITUD
# =========================================================

@st.cache_data(ttl=600)
def obtener_datos_ambientales(latitud, longitud):
    """
    Obtiene temperatura actual y elevación mediante Open-Meteo
    utilizando las coordenadas proporcionadas por el dispositivo/navegador.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitud,
        "longitude": longitud,
        "current": "temperature_2m",
        "elevation": latitud,
        "timezone": "auto"
    }

    respuesta = requests.get(
        url,
        params=params,
        timeout=15
    )

    respuesta.raise_for_status()

    datos = respuesta.json()

    temperatura = datos["current"]["temperature_2m"]

    elevacion = datos.get("elevation")

    return float(temperatura), float(elevacion)


def detectar_ambiente():
    """
    Solicita la ubicación del dispositivo mediante el navegador.
    """

    ubicacion = streamlit_geolocation()

    if not ubicacion:
        return None

    latitud = ubicacion.get("latitude")
    longitud = ubicacion.get("longitude")

    if latitud is None or longitud is None:
        return None

    try:
        temperatura, altitud = obtener_datos_ambientales(
            float(latitud),
            float(longitud)
        )

        return {
            "latitud": float(latitud),
            "longitud": float(longitud),
            "temperatura": temperatura,
            "altitud": altitud
        }

    except Exception as e:
        st.warning(
            "No fue posible obtener temperatura y altitud automáticamente: "
            f"{e}"
        )
        return None


# =========================================================
# 9. REGLAS AGRONÓMICAS
# =========================================================

def aplicar_regla_riego(entrada):
    """
    Si el riego de la etapa inicial o intermedia es 0,
    la producción debe ser 0.

        R1 = 0 OR R2 = 0
        => Producción = 0
    """

    riego_1 = entrada["Riego_Etapa1_mm"]
    riego_2 = entrada["Riego_Etapa2_mm"]

    return riego_1 == 0 or riego_2 == 0


def clasificar_produccion(produccion):
    if produccion >= 20000:
        return "Producción alta estimada", "success"

    elif produccion >= 10000:
        return "Producción media estimada", "warning"

    elif produccion > 0:
        return "Producción baja estimada", "warning"

    else:
        return "Producción nula estimada", "error"


# =========================================================
# 10. GRÁFICO DINÁMICO
# =========================================================

def normalizar_grafico(variable, valor):
    """
    Escala el valor a 0-100 exclusivamente para visualización.
    No modifica el valor utilizado por el modelo.
    """

    if variable not in rangos:
        return 0

    minimo = float(rangos[variable]["min"])
    maximo = float(rangos[variable]["max"])

    if maximo == minimo:
        return 0

    resultado = (
        (float(valor) - minimo)
        / (maximo - minimo)
    ) * 100

    return max(0, min(100, resultado))


def crear_grafico_dinamico(entrada):

    variables = []
    valores = []
    valores_norm = []
    ideales_norm = []

    for variable in columnas_modelo:

        valor = entrada[variable]

        # Para temperatura y altitud se muestra el valor detectado.
        ideal = valores_ideales.get(
            variable,
            rangos[variable]["mean"]
        )

        variables.append(
            obtener_nombre_amigable(variable)
        )

        valores.append(valor)

        valores_norm.append(
            normalizar_grafico(variable, valor)
        )

        ideales_norm.append(
            normalizar_grafico(variable, ideal)
        )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=variables,
            y=valores_norm,
            name="Valores seleccionados",
            text=[
                f"{v:.1f}" if isinstance(v, float) else str(v)
                for v in valores
            ],
            textposition="outside"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=variables,
            y=ideales_norm,
            name="Valores ideales",
            mode="lines+markers",
            line=dict(width=4),
            marker=dict(size=9)
        )
    )

    fig.update_layout(
        title="Comparación entre valores seleccionados e ideales",
        xaxis_title="Variables del cultivo",
        yaxis_title="Escala normalizada 0 - 100",
        height=600,
        template="plotly_white"
    )

    fig.update_xaxes(tickangle=45)

    return fig


# =========================================================
# 11. INTERFAZ
# =========================================================

st.title(
    "🌱 Prototipo para la estimación predictiva "
    "de producción de camote"
)

st.subheader("NEXARING")

st.markdown(
    """
    Esta aplicación estima la **Producción Total de camote**
    mediante un modelo de inteligencia artificial basado en
    **CatBoost Regressor**.
    """
)


# =========================================================
# 12. MÉTRICAS
# =========================================================

with st.expander("📊 Ver métricas del modelo"):

    if metricas:

        if "r2" in metricas:
            st.write(
                f"**R²:** {metricas['r2']:.4f}"
            )

        if "mae" in metricas:
            st.write(
                f"**MAE:** {metricas['mae']:.4f}"
            )

        if "mse" in metricas:
            st.write(
                f"**MSE:** {metricas['mse']:.4f}"
            )

        if "rmse" in metricas:
            st.write(
                f"**RMSE:** {metricas['rmse']:.4f}"
            )

    else:
        st.info(
            "No se encontró el archivo de métricas."
        )


# =========================================================
# 13. DETECCIÓN AUTOMÁTICA DEL AMBIENTE
# =========================================================

st.markdown("## 🌎 Condiciones ambientales")

st.info(
    "La temperatura y la altitud se obtienen automáticamente "
    "a partir de la ubicación autorizada por el dispositivo/navegador. "
    "Se requiere conexión a Internet."
)

datos_ambientales = detectar_ambiente()

if datos_ambientales:

    st.session_state["temperatura_auto"] = (
        datos_ambientales["temperatura"]
    )

    st.session_state["altitud_auto"] = (
        datos_ambientales["altitud"]
    )

    st.success(
        "Ubicación detectada correctamente."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Temperatura actual",
            f"{datos_ambientales['temperatura']:.1f} °C"
        )

    with col2:
        st.metric(
            "Altitud",
            f"{datos_ambientales['altitud']:.0f} msnm"
        )

    with col3:
        st.metric(
            "Latitud",
            f"{datos_ambientales['latitud']:.4f}"
        )

else:

    st.warning(
        "No se ha autorizado todavía la ubicación. "
        "Permite el acceso a la ubicación en el navegador "
        "para obtener automáticamente temperatura y altitud."
    )

    # Fallback para poder probar la aplicación.
    if "temperatura_auto" not in st.session_state:
        st.session_state["temperatura_auto"] = 26.0

    if "altitud_auto" not in st.session_state:
        st.session_state["altitud_auto"] = 250.0


# =========================================================
# 14. CONTROLES RÁPIDOS
# =========================================================

st.markdown("## ⚙️ Controles rápidos")

col_a, col_b = st.columns(2)

with col_a:
    boton_ideales = st.button(
        "⭐ Valores ideales",
        use_container_width=True
    )

with col_b:
    boton_promedios = st.button(
        "🔄 Promedios del dataset",
        use_container_width=True
    )


# =========================================================
# 15. VALORES DE ENTRADA
# =========================================================

st.markdown("## 🧑‍🌾 Ingrese o ajuste los valores")

entrada_usuario = {}


# ---------------------------------------------------------
# Valores normales editables
# ---------------------------------------------------------

for variable in columnas_modelo:

    # Temperatura y altitud se toman automáticamente.
    if variable == "Temperatura_C":

        entrada_usuario[variable] = (
            st.session_state["temperatura_auto"]
        )

        st.number_input(
            "🌡️ Temperatura media (obtenida automáticamente)",
            value=float(
                st.session_state["temperatura_auto"]
            ),
            disabled=True,
            key="temperatura_visual"
        )

        continue

    if variable == "Altitud_msnm":

        entrada_usuario[variable] = (
            st.session_state["altitud_auto"]
        )

        st.number_input(
            "⛰️ Altitud del terreno (obtenida automáticamente)",
            value=float(
                st.session_state["altitud_auto"]
            ),
            disabled=True,
            key="altitud_visual"
        )

        continue

    minimo = int(
        round(
            rangos[variable]["min"]
        )
    )

    maximo = int(
        round(
            rangos[variable]["max"]
        )
    )

    if variable == "pH_Suelo":

        valor_default = (
            valores_ideales["pH_Suelo"]
        )

        entrada_usuario[variable] = st.slider(
            "pH del suelo",
            min_value=4.0,
            max_value=7.5,
            value=float(valor_default),
            step=0.1
        )

    elif variable == "TipoSuelo":

        entrada_usuario[variable] = st.selectbox(
            "Tipo de suelo",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1 - Franco",
                2: "2 - Arenoso",
                3: "3 - Arcilloso"
            }[x]
        )

    else:

        valor_default = valores_ideales.get(
            variable,
            rangos[variable]["mean"]
        )

        valor_default = max(
            minimo,
            min(
                maximo,
                int(round(valor_default))
            )
        )

        entrada_usuario[variable] = st.slider(
            obtener_nombre_amigable(variable),
            min_value=minimo,
            max_value=maximo,
            value=valor_default,
            step=1
        )


# =========================================================
# 16. TABLA DE VALORES
# =========================================================

with st.expander("🌿 Ver valores seleccionados"):

    tabla = []

    for variable in columnas_modelo:

        tabla.append({
            "Variable": obtener_nombre_amigable(variable),
            "Valor seleccionado": entrada_usuario[variable]
        })

    st.dataframe(
        pd.DataFrame(tabla),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 17. GRÁFICO
# =========================================================

st.markdown("## 📊 Comparación gráfica")

fig = crear_grafico_dinamico(
    entrada_usuario
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "La escala 0-100 se utiliza únicamente para visualizar "
    "variables con diferentes unidades. Los valores mostrados "
    "corresponden a los valores reales ingresados."
)


# =========================================================
# 18. PREDICCIÓN
# =========================================================

st.markdown("---")

boton_predecir = st.button(
    "🔍 Predecir producción",
    use_container_width=True
)


if boton_predecir:

    try:

        # -------------------------------------------------
        # REGLA AGRONÓMICA DEL RIEGO
        # -------------------------------------------------

        if aplicar_regla_riego(entrada_usuario):

            produccion = 0.0

            st.error(
                "Producción nula estimada por déficit crítico "
                "de riego en la etapa inicial o intermedia."
            )

            st.warning(
                "De acuerdo con la regla agronómica: "
                "si Riego_Etapa1_mm = 0 o "
                "Riego_Etapa2_mm = 0, "
                "Producción_Total = 0."
            )

        else:

            # ---------------------------------------------
            # NORMALIZAR ANTES DE ENVIAR AL MODELO
            # ---------------------------------------------

            entrada_normalizada = normalizar_entrada(
                entrada_usuario
            )

            # ---------------------------------------------
            # PREDICCIÓN DEL CATBOOST
            # ---------------------------------------------

            prediccion_normalizada = modelo.predict(
                entrada_normalizada
            )[0]

            # ---------------------------------------------
            # DESNORMALIZAR A KG/HA
            # ---------------------------------------------

            produccion = desnormalizar_produccion(
                prediccion_normalizada
            )

            # Nunca permitir producción negativa.
            produccion = max(
                0.0,
                produccion
            )

        # -------------------------------------------------
        # CLASIFICACIÓN
        # -------------------------------------------------

        mensaje, tipo = clasificar_produccion(
            produccion
        )

        if tipo == "success":
            st.success(mensaje)

        elif tipo == "warning":
            st.warning(mensaje)

        else:
            st.error(mensaje)

        # -------------------------------------------------
        # RESULTADO
        # -------------------------------------------------

        st.metric(
            "Producción estimada",
            f"{produccion:,.2f} kg/ha"
        )

        st.info(
            f"Equivalente aproximado: "
            f"{produccion / 45.36:,.2f} quintales/ha "
            "(1 quintal = 45,36 kg)."
        )

        # -------------------------------------------------
        # DETALLE DEL VECTOR ENTRADA
        # -------------------------------------------------

        with st.expander(
            "🔎 Ver datos utilizados por el modelo"
        ):

            st.write(
                "Valores originales:"
            )

            st.dataframe(
                pd.DataFrame(
                    [entrada_usuario]
                ),
                use_container_width=True
            )

            st.write(
                "Valores normalizados enviados al CatBoost:"
            )

            st.dataframe(
                entrada_normalizada,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "Se produjo un error durante la predicción."
        )

        st.exception(e)


# =========================================================
# 19. INFORMACIÓN FINAL
# =========================================================

st.markdown("---")

st.caption(
    "NEXARING — Sistema de estimación predictiva "
    "de producción agrícola mediante inteligencia artificial."
)
