# =====================================================================
# PROTOTIPO: ESTIMACIÓN DE LA PRODUCCIÓN DE CAMOTE (kg/ha)
# Tesis: "Modelo de simulación para la estimación de producción de camote
#         en Portoviejo a partir de rangos agronómicos definidos"
# Autor: Ing. Néxar Bolívar Lucas Ostaiza — UTM, Facultad de Posgrado
# Modelo: CatBoost (Categorical Boosting), entrenado como en la tesis
# =====================================================================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from streamlit_js_eval import get_geolocation

st.set_page_config(page_title="Producción de camote", page_icon="🍠", layout="wide")

CARPETA = Path(__file__).resolve().parent
# Datos estandarizados (escenario B). Se acepta también "DataGauss_Normalizada (1).csv", etc.
ARCHIVO_DATOS = next(iter(sorted(CARPETA.glob("DataGauss_Normalizada*.csv"))), CARPETA / "DataGauss_Normalizada.csv")
ARCHIVO_ESCALA = CARPETA / "escala_datos.json"           # media y desviación del dataset original

PORTOVIEJO = {"lat": -1.0546, "lon": -80.4545, "nombre": "Portoviejo, Manabí (predeterminado)"}
DIAS_PRECIPITACION = 90  # duración ideal del cultivo: la lluvia se acumula en los últimos 90 días

# ---------------------------------------------------------------------
# 1. VARIABLES (Tabla 1 de la tesis): mínimo, ideal y máximo
# ---------------------------------------------------------------------
VARIABLES = [
    # clave,                   nombre,                         unidad,  mín,  ideal,  máx,  paso
    ("MesInicioPlantacion",     "Mes de inicio de plantación",  "mes",     1,    6,     12,   1),
    ("DuracionPlantacion_dias", "Duración del cultivo",         "días",   60,   90,    120,   1),
    ("Fertilizante",            "Fertilizante",                 "kg/ha",   0,   80,    100,   1),
    ("Riego_Etapa1_mm",         "Riego en etapa inicial",       "mm",      0,   35,     60,   1),
    ("Riego_Etapa2_mm",         "Riego en etapa intermedia",    "mm",      0,   45,     70,   1),
    ("Riego_Etapa3_mm",         "Riego en etapa final",         "mm",      0,   55,     80,   1),
    ("pH_Suelo",                "pH del suelo",                 "",        4,  6.5,      8,   0.1),
    ("TipoSuelo",               "Tipo de suelo",                "",        1,    1,      3,   1),
    ("Temperatura_C",           "Temperatura",                  "°C",     10,   26,     40,   0.1),
    ("Precipitacion_mm",        "Precipitación acumulada",      "mm",    200,  564,   1200,   1),
    ("Humedad_porcentaje",      "Humedad relativa",             "%",      20,   70,    100,   1),
    ("Altitud_msnm",            "Altitud del terreno",          "msnm",    0,  250,   1500,   1),
]
VAR = {v[0]: dict(zip(["clave", "nombre", "unidad", "min", "ideal", "max", "paso"], v)) for v in VARIABLES}

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
SUELOS = {1: "Franco", 2: "Arenoso", 3: "Arcilloso"}
CLIMATICAS = ["Temperatura_C", "Humedad_porcentaje", "Precipitacion_mm", "Altitud_msnm"]

# Función de cálculo de la tesis (Capítulo II): pesos y factores
PESOS = {"DuracionPlantacion_dias": 0.09, "Temperatura_C": 0.12, "Precipitacion_mm": 0.06,
         "Humedad_porcentaje": 0.05, "Fertilizante": 0.17, "Riego_Etapa1_mm": 0.14,
         "Riego_Etapa2_mm": 0.16, "Riego_Etapa3_mm": 0.09, "pH_Suelo": 0.08, "Altitud_msnm": 0.04}
F_MES = {1: 0.40, 2: 0.45, 3: 0.50, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0,
         10: 0.50, 11: 0.45, 12: 0.40}
F_SUELO = {1: 1.00, 2: 0.85, 3: 0.75}
UMBRAL_IDEAL = 0.90
KG_POR_QUINTAL = 45.36  # puntaje ≥ 0.90 → la variable se considera en su valor ideal (verde)


def valor_inicial(clave):
    """Valor ideal con el tipo correcto (decimal o entero) para su barra."""
    v = VAR[clave]
    return float(v["ideal"]) if v["paso"] < 1 else int(v["ideal"])


def numero(valor):
    """Número con separador de miles en punto y decimales con coma (formato Ecuador)."""
    texto = f"{valor:,.1f}" if not float(valor).is_integer() else f"{valor:,.0f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def acotar(valor, clave):
    """Mantiene un valor dentro del rango [mínimo, máximo] de la variable."""
    v = VAR[clave]
    return min(max(valor, v["min"]), v["max"])


def puntaje(clave, valor):
    """Puntuación de la tesis: 1 en el valor ideal y 0 en los extremos del rango."""
    if clave == "MesInicioPlantacion":
        return F_MES[int(valor)]
    if clave == "TipoSuelo":
        return F_SUELO[int(valor)]
    if clave == "pH_Suelo":
        if valor < 6:
            return max(0.0, (valor - 4) / 2)
        if valor <= 7:
            return 1.0
        return max(0.0, 8 - valor)
    v = VAR[clave]
    if valor <= v["ideal"]:
        return max(0.0, (valor - v["min"]) / (v["ideal"] - v["min"]))
    return max(0.0, (v["max"] - valor) / (v["max"] - v["ideal"]))


def produccion_funcion_tesis(valores):
    """Producción teórica según la función de cálculo de la tesis (sin ruido)."""
    if valores["Riego_Etapa1_mm"] == 0 or valores["Riego_Etapa2_mm"] == 0:
        return 0.0
    s = sum(peso * puntaje(clave, valores[clave]) for clave, peso in PESOS.items())
    return 40000 * s * F_MES[int(valores["MesInicioPlantacion"])] * F_SUELO[int(valores["TipoSuelo"])]


def es_ideal(clave, valor):
    return puntaje(clave, valor) >= (1.0 if clave in ("MesInicioPlantacion", "TipoSuelo") else UMBRAL_IDEAL)


def texto_valor(clave, valor):
    if clave == "MesInicioPlantacion":
        return MESES[int(valor) - 1]
    if clave == "TipoSuelo":
        return SUELOS[int(valor)]
    return f"{numero(valor)} {VAR[clave]['unidad']}".strip()


def texto_limite(clave, valor):
    """Valor sin unidad, para las columnas de la tabla."""
    if clave in ("MesInicioPlantacion", "TipoSuelo"):
        return texto_valor(clave, valor)
    return numero(valor)


# ---------------------------------------------------------------------
# 2. MODELO CATBOOST Y ESTANDARIZACIÓN
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando el modelo CatBoost…")
def cargar_modelo():
    """Entrena CatBoost exactamente como en el cuaderno de la tesis:
    mismas 12 entradas, partición 80/20 (random_state=42) y mismos hiperparámetros."""
    datos = pd.read_csv(ARCHIVO_DATOS)
    X = datos.iloc[:, :12]
    y = datos.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                               loss_function="RMSE", random_seed=42, verbose=0)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    metricas = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred),
                "mse": mse, "rmse": float(np.sqrt(mse))}
    with open(ARCHIVO_ESCALA, encoding="utf-8") as f:
        escala = json.load(f)
    escala["orden_entradas"] = list(X.columns)
    escala["objetivo"] = datos.columns[-1]
    escala["metricas"] = metricas
    return modelo, escala


def estimar_produccion(modelo, escala, valores):
    """Estandariza las entradas con la media y desviación del dataset original,
    ejecuta CatBoost y devuelve la producción en kg/ha (transformación inversa)."""
    fila = {c: (valores[c] - escala["media"][c]) / escala["desv"][c] for c in escala["orden_entradas"]}
    entrada = pd.DataFrame([fila], columns=escala["orden_entradas"])
    z = float(modelo.predict(entrada)[0])
    objetivo = escala["objetivo"]
    return max(0.0, z * escala["desv"][objetivo] + escala["media"][objetivo]), entrada


# ---------------------------------------------------------------------
# 3. DATOS DE LA WEB: ubicación, clima actual, lluvia acumulada y altitud
# ---------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def obtener_clima(lat, lon):
    """Open-Meteo (gratuito, sin clave): temperatura y humedad actuales,
    precipitación de los últimos 90 días y altitud del lugar."""
    respuesta = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m",
                "daily": "precipitation_sum", "past_days": 92, "forecast_days": 1,
                "timezone": "auto"},
        timeout=15)
    respuesta.raise_for_status()
    d = respuesta.json()
    lluvia_diaria = [x or 0 for x in d["daily"]["precipitation_sum"][:-1]]  # sin el día de hoy
    return {
        "Temperatura_C": float(d["current"]["temperature_2m"]),
        "Humedad_porcentaje": float(d["current"]["relative_humidity_2m"]),
        "Precipitacion_mm": float(sum(lluvia_diaria[-DIAS_PRECIPITACION:])),
        "Altitud_msnm": float(d.get("elevation", 0) or 0),
        "hora": d["current"].get("time", ""),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def nombre_del_lugar(lat, lon):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": lat, "lon": lon, "format": "json", "zoom": 10,
                                 "accept-language": "es"},
                         headers={"User-Agent": "app-prediccion-camote (tesis UTM)"}, timeout=10)
        a = r.json().get("address", {})
        partes = [a.get("city") or a.get("town") or a.get("village") or a.get("county"), a.get("state")]
        return ", ".join(p for p in partes if p) or f"{lat:.3f}, {lon:.3f}"
    except Exception:
        return f"{lat:.3f}, {lon:.3f}"


# ---------------------------------------------------------------------
# 4. ESTADO INICIAL DE LAS BARRAS (valores ideales por defecto)
# ---------------------------------------------------------------------
for clave in VAR:
    st.session_state.setdefault(clave, valor_inicial(clave))
st.session_state.setdefault("clima_aplicado_en", None)


def poner_valores_ideales():
    for clave in VAR:
        st.session_state[clave] = valor_inicial(clave)


def poner_datos_clima():
    clima = st.session_state.get("clima_actual")
    if clima:
        for clave in CLIMATICAS:
            paso = VAR[clave]["paso"]
            valor = acotar(clima[clave], clave)
            st.session_state[clave] = float(round(valor, 1)) if paso < 1 else int(round(valor))


# ---------------------------------------------------------------------
# 5. ENCABEZADO
# ---------------------------------------------------------------------
st.title("🍠 Estimación de la producción de camote de pulpa anaranjada")
st.markdown(
    "Prototipo del **modelo de simulación** desarrollado en la tesis de Maestría en Ciencia de Datos "
    "y Máquinas de Aprendizaje (UTM). Estima la producción en **kg/ha** con el modelo **CatBoost**, "
    "a partir de los rangos agronómicos definidos con el INIAP.")

try:
    modelo, escalador = cargar_modelo()
except FileNotFoundError as error:
    st.error(f"Falta un archivo en el repositorio: **{Path(error.filename).name}**. Súbalo junto a app.py.")
    st.stop()
except Exception as error:
    st.error("No se pudo preparar el modelo. Envíe este mensaje a soporte:")
    st.exception(error)
    st.stop()

# ---------------------------------------------------------------------
# 6. UBICACIÓN Y CLIMA
# ---------------------------------------------------------------------
st.subheader("🌎 Condiciones del lugar")
ubicacion = get_geolocation()
if ubicacion and "coords" in ubicacion:
    lat, lon = ubicacion["coords"]["latitude"], ubicacion["coords"]["longitude"]
    lugar = nombre_del_lugar(round(lat, 3), round(lon, 3))
    origen = "ubicación del dispositivo"
else:
    lat, lon, lugar = PORTOVIEJO["lat"], PORTOVIEJO["lon"], PORTOVIEJO["nombre"]
    origen = "ubicación predeterminada"
    if ubicacion and "error" in ubicacion:
        st.info("No se autorizó la ubicación del navegador; se usan los datos de Portoviejo. "
                "Para usar su ubicación, permita el acceso en el navegador y recargue la página.")

try:
    clima = obtener_clima(round(lat, 3), round(lon, 3))
    st.session_state["clima_actual"] = clima
    marca = f"{lat:.3f},{lon:.3f}"
    if st.session_state["clima_aplicado_en"] != marca:   # se cargan una sola vez por lugar
        poner_datos_clima()
        st.session_state["clima_aplicado_en"] = marca
except Exception:
    clima = None
    st.warning("No fue posible consultar el clima en la web. Las barras muestran los valores ideales "
               "y pueden ajustarse manualmente.")

if clima:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📍 Lugar", lugar.split(",")[0], help=f"{lugar} ({origen})")
    c2.metric("🌡️ Temperatura actual", f"{clima['Temperatura_C']:.1f} °C")
    c3.metric("💧 Humedad relativa", f"{clima['Humedad_porcentaje']:.0f} %")
    c4.metric(f"🌧️ Lluvia últimos {DIAS_PRECIPITACION} días", f"{clima['Precipitacion_mm']:.0f} mm")
    c5.metric("⛰️ Altitud", f"{clima['Altitud_msnm']:.0f} msnm")
    avisos = [f"{VAR[c]['nombre']}: {clima[c]:.0f} {VAR[c]['unidad']} está fuera del rango del modelo "
              f"({VAR[c]['min']}–{VAR[c]['max']}); se usó el límite más cercano."
              for c in CLIMATICAS if clima[c] != acotar(clima[c], c)]
    for a in avisos:
        st.caption("⚠️ " + a)
    st.caption(f"Fuente: Open-Meteo · consulta {clima['hora'].replace('T', ' ')} hora local. "
               "Estos valores se cargan en las barras, pero puede modificarlos manualmente.")

b1, b2, _ = st.columns([1, 1, 2])
b1.button("⭐ Valores ideales", on_click=poner_valores_ideales, use_container_width=True)
b2.button("🌦️ Usar datos del clima", on_click=poner_datos_clima, use_container_width=True,
          disabled=clima is None)

# ---------------------------------------------------------------------
# 7. BARRAS DE DESPLAZAMIENTO (entre el mínimo y el máximo de cada variable)
# ---------------------------------------------------------------------
st.subheader("🎚️ Variables independientes")
grupos = {
    "🌱 Variables agronómicas": ["MesInicioPlantacion", "DuracionPlantacion_dias", "Fertilizante",
                                "Riego_Etapa1_mm", "Riego_Etapa2_mm", "Riego_Etapa3_mm"],
    "🟤 Variables edáficas (suelo)": ["pH_Suelo", "TipoSuelo"],
    "🌦️ Variables ambientales": ["Temperatura_C", "Precipitacion_mm", "Humedad_porcentaje", "Altitud_msnm"],
}
valores = {}
for grupo, claves in grupos.items():           # una barra debajo de otra
    with st.container(border=True):
        st.markdown(f"#### {grupo}")
        if "ambientales" in grupo:
            st.caption("Se cargan con los datos del clima de su ubicación; puede cambiarlos manualmente.")
        for clave in claves:
            v = VAR[clave]
            if clave == "MesInicioPlantacion":
                valores[clave] = st.select_slider(v["nombre"], options=list(range(1, 13)),
                                                  format_func=lambda m: MESES[m - 1], key=clave)
            elif clave == "TipoSuelo":
                valores[clave] = st.select_slider(v["nombre"], options=[1, 2, 3],
                                                  format_func=lambda s: SUELOS[s], key=clave)
            else:
                decimal = v["paso"] < 1
                etiqueta = f"{v['nombre']} ({v['unidad']})" if v["unidad"] else v["nombre"]
                valores[clave] = st.slider(
                    etiqueta,
                    min_value=float(v["min"]) if decimal else int(v["min"]),
                    max_value=float(v["max"]) if decimal else int(v["max"]),
                    step=v["paso"], key=clave,
                    help=f"Mínimo {v['min']} · ideal {v['ideal']} · máximo {v['max']}")

# ---------------------------------------------------------------------
# 8. TABLA DE VARIABLES: mínimo, ideal, máximo y valor actual
# ---------------------------------------------------------------------
st.subheader("📋 Rangos de las variables")
def estado(clave, valor):
    """Ideal, o en qué dirección se aleja del valor ideal."""
    if es_ideal(clave, valor):
        return "🟢 Ideal"
    if clave == "MesInicioPlantacion":
        return "🔴 Antes de la época ideal" if valor < 4 else "🔴 Después de la época ideal"
    if clave == "TipoSuelo":
        return f"🔴 Suelo no ideal ({SUELOS[int(valor)].lower()})"
    referencia = 6 if (clave == "pH_Suelo" and valor < 6) else (7 if clave == "pH_Suelo" else VAR[clave]["ideal"])
    return "🔴 Por debajo del valor ideal" if valor < referencia else "🔴 Por encima del valor ideal"


def rango(clave):
    if clave == "TipoSuelo":
        return "Franco, Arenoso, Arcilloso"
    return f"{texto_limite(clave, VAR[clave]['min'])} – {texto_limite(clave, VAR[clave]['max'])}"


tabla = pd.DataFrame([{
    "Variable": f"{VAR[c]['nombre']} ({VAR[c]['unidad']})" if VAR[c]["unidad"] and c != "MesInicioPlantacion"
                else VAR[c]["nombre"],
    "Rango": rango(c),
    "Ideal": {"pH_Suelo": "6 – 7", "MesInicioPlantacion": "Abril – Septiembre"}.get(
        c, texto_limite(c, VAR[c]["ideal"])),
    "Valor actual": texto_limite(c, valores[c]),
    "Estado": estado(c, valores[c]),
} for c in VAR])
st.dataframe(tabla, hide_index=True, use_container_width=True, height=36 * (len(tabla) + 1) + 3)

# ---------------------------------------------------------------------
# 9. GRÁFICO EN TIEMPO REAL: verde = ideal, rojo = alejado del ideal
# ---------------------------------------------------------------------
st.subheader("📊 Comparación con los valores ideales")


def posicion(clave, valor):
    v = VAR[clave]
    return (valor - v["min"]) / (v["max"] - v["min"]) * 100


nombres = [VAR[c]["nombre"] for c in VAR]
pos_actual = [posicion(c, valores[c]) for c in VAR]
pos_ideal = [posicion(c, VAR[c]["ideal"]) for c in VAR]
colores = ["#2e9e4f" if es_ideal(c, valores[c]) else "#d64545" for c in VAR]

grafico = go.Figure()
grafico.add_bar(x=nombres, y=pos_actual, marker_color=colores, name="Valor ingresado",
                text=[texto_valor(c, valores[c]) for c in VAR], textposition="outside",
                hovertemplate="%{x}<br>%{text}<extra></extra>")
grafico.add_scatter(x=nombres, y=pos_ideal, mode="lines+markers", name="Valor ideal",
                    line=dict(color="#1b7f3b", width=3), marker=dict(size=9),
                    text=[texto_valor(c, VAR[c]["ideal"]) for c in VAR],
                    hovertemplate="%{x}<br>Ideal: %{text}<extra></extra>")
grafico.add_scatter(x=[None], y=[None], mode="markers", name="Alejado del ideal",
                    marker=dict(color="#d64545", size=12, symbol="square"))
grafico.update_layout(height=480, yaxis=dict(title="Posición dentro del rango (%)", range=[0, 118]),
                      xaxis=dict(tickangle=-35), legend=dict(orientation="h", y=1.12),
                      margin=dict(t=40, b=10), template="plotly_white")
st.plotly_chart(grafico, use_container_width=True)
st.caption("Cada barra muestra dónde se ubica el valor entre el mínimo (0 %) y el máximo (100 %) de su rango. "
           "La línea verde marca el valor ideal. La barra es verde cuando la variable está en su valor ideal "
           f"(puntuación de la tesis ≥ {UMBRAL_IDEAL:.2f}; meses de abril a septiembre; suelo franco) "
           "y roja cuando se aleja de él.")

# ---------------------------------------------------------------------
# 10. ESTIMACIÓN DE LA PRODUCCIÓN CON CATBOOST
# ---------------------------------------------------------------------
st.subheader("🌱 Producción estimada")
if st.button("🔍 Predecir producción", type="primary", use_container_width=True):
    if valores["Riego_Etapa1_mm"] == 0 or valores["Riego_Etapa2_mm"] == 0:
        produccion = 0.0
        st.error("Sin riego en la etapa inicial o intermedia el cultivo se pierde "
                 "(regla agronómica de la tesis): la producción es 0 kg/ha.")
        entrada = None
    else:
        produccion, entrada = estimar_produccion(modelo, escalador, valores)

    m = escalador["metricas"]
    desv_kg = escalador["desv"][escalador["objetivo"]]       # desviación de la producción (kg/ha)
    rmse_kg = m["rmse"] * desv_kg
    r1, r2, r3 = st.columns(3)
    r1.metric("Producción estimada", f"{numero(round(produccion))} kg/ha")
    r2.metric("En quintales", f"{numero(round(produccion / KG_POR_QUINTAL))} qq/ha")
    if produccion > 0:
        bajo, alto = max(0, produccion - rmse_kg), produccion + rmse_kg
        r3.metric("Rango probable", f"{numero(round(bajo))} – {numero(round(alto))} kg/ha",
                  help="Producción estimada ± el error típico del modelo (RMSE).")
        st.info(f"🧑‍🌾 **Cómo leer este resultado:** con estas condiciones se esperan unos "
                f"**{numero(round(produccion))} kg por hectárea** "
                f"(≈ {numero(round(produccion / KG_POR_QUINTAL))} quintales). Lo más probable es que la "
                f"producción esté entre **{numero(round(bajo))} y {numero(round(alto))} kg/ha**, porque el "
                f"modelo suele equivocarse en ± {numero(round(rmse_kg))} kg/ha "
                f"(≈ {numero(round(rmse_kg / produccion * 100, 1))} % de esta estimación).")

    alejadas = [VAR[c]["nombre"] for c in VAR if not es_ideal(c, valores[c])]
    if alejadas:
        st.info("Variables alejadas de su valor ideal: " + ", ".join(alejadas) + ".")
    else:
        st.success("Todas las variables están en su valor ideal.")

    with st.expander("Ver detalle del cálculo"):
        st.write("Producción según la función de cálculo de la tesis (sin ruido): "
                 f"**{numero(round(produccion_funcion_tesis(valores)))} kg/ha**")
        if entrada is not None:
            st.write("Valores enviados al modelo (estandarizados con la media y desviación del dataset):")
            st.dataframe(entrada, hide_index=True, use_container_width=True)
        m = escalador.get("metricas", {})
        if m:
            st.write(f"Desempeño del modelo en el 20 % de prueba (escala estandarizada): "
                     f"R² = {m['r2']:.4f} · MAE = {m['mae']:.4f} · MSE = {m['mse']:.4f} · RMSE = {m['rmse']:.4f}")

# ---------------------------------------------------------------------
# 11. ¿QUÉ TAN CONFIABLE ES LA ESTIMACIÓN? (métricas traducidas al campo)
# ---------------------------------------------------------------------
st.subheader("📏 ¿Qué tan confiable es la estimación?")
m = escalador["metricas"]
desv_kg = escalador["desv"][escalador["objetivo"]]
mae_kg, rmse_kg = m["mae"] * desv_kg, m["rmse"] * desv_kg
tabla_metricas = pd.DataFrame([
    {"Métrica": "R² (coeficiente de determinación)",
     "Equivalente en el campo": f"Explica el {numero(round(m['r2'] * 100, 1))} % de las diferencias de producción "
                                "entre una condición y otra"},
    {"Métrica": "MAE (error absoluto medio)",
     "Equivalente en el campo": f"± {numero(round(mae_kg))} kg/ha (≈ {numero(round(mae_kg / KG_POR_QUINTAL, 1))} "
                                f"qq/ha) de error promedio"},
    {"Métrica": "RMSE (raíz del error cuadrático medio)",
     "Equivalente en el campo": f"± {numero(round(rmse_kg))} kg/ha (≈ {numero(round(rmse_kg / KG_POR_QUINTAL, 1))} "
                                f"qq/ha) de error típico; es el margen del \"rango probable\""},
])
st.dataframe(tabla_metricas, hide_index=True, use_container_width=True)
st.caption(f"Las métricas se calculan en escala estandarizada sobre el 20 % de los datos reservados para prueba; "
           f"se convierten a kg/ha multiplicándolas por la desviación estándar de la producción "
           f"({numero(round(desv_kg))} kg/ha). 1 quintal = {numero(KG_POR_QUINTAL)} kg. Estos errores miden cuánto "
           "se aleja el modelo de la simulación basada en los criterios del INIAP, no de cosechas reales medidas en campo.")

st.divider()
st.caption("⚠️ Prototipo con carácter demostrativo: el modelo se entrenó con datos sintéticos generados a partir "
           "de rangos agronómicos del INIAP y aún no ha sido validado con datos de campo. "
           f"© {datetime.now().year} · Néxar Lucas Ostaiza · UTM")
