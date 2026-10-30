# -*- coding: utf-8 -*-
"""
CardioRiesgo - ACA 1 Aprendizaje Automático (CUN), Grupo 13.
Aplicación Streamlit sobre el dataset UCI Heart Disease (Cleveland), 303 pacientes.

Siete pestañas que cuentan el proyecto en orden: el problema, los datos, cómo aprende
un modelo (entrenamiento en vivo), los resultados, la evaluación de un paciente, la de
varios pacientes con un CSV y la ficha del proyecto. En la barra lateral se elige el
modelo activo: la regresión logística del notebook (por defecto) u otros cuatro que se
entrenan al abrir la app con los mismos pacientes e hiperparámetros.

Diseño: tema claro por defecto (como un ECG impreso) y oscuro (como un monitor de
cabecera). Cian para el modelo, verde riesgo bajo, rojo riesgo alto, ámbar umbral;
en los gráficos, azul sin enfermedad y naranja con enfermedad.

Ejecución local:  streamlit run app.py
"""
import base64
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo" / "modelo_cardio.joblib"
RUTA_META = BASE / "modelo" / "metadata.json"
RUTA_DATOS = BASE / "data" / "heart_clean.csv"
RUTA_PLANTILLA = BASE / "data" / "plantilla_pacientes.csv"
RUTA_LOGO = BASE / "assets" / "cun.png"

# ---------------------------------------------------------------------------
# Paleta y tipografía. Hay dos temas con los mismos significados de color:
# claro (por defecto, como un trazo de ECG impreso) y oscuro (como la pantalla de
# un monitor). El CSS usa light-dark(), así que cada pieza propia toma el valor
# del tema que Streamlit tenga activo. Contraste verificado: texto >= 4,5:1.
# ---------------------------------------------------------------------------
TOKENS = {  # nombre: (tema claro, tema oscuro)
    "pantalla": ("#f6f7f9", "#060b16"),
    "panel": ("#ffffff", "#0c1424"),
    "panel-2": ("#f1f4f8", "#111b30"),
    "borde": ("#e2e6ee", "#1e2b47"),
    "borde-2": ("#cdd4e0", "#2a3a5c"),
    "tinta": ("#0f172a", "#eaf0fa"),
    "tinta-2": ("#475066", "#a9b6cc"),
    "tinta-3": ("#5d6880", "#8391a8"),
    "cian": ("#0b7493", "#2fd3f0"),    # canal del modelo y acento de la interfaz
    "verde": ("#0b7f47", "#2fe08f"),   # riesgo bajo (siempre con ícono y texto)
    "rojo": ("#c4223a", "#ff5a6e"),    # riesgo alto (siempre con ícono y texto)
    "ambar": ("#9a5a00", "#ffb648"),   # umbral de decisión
    "traza": ("#111827", "#2fe08f"),   # trazo del ECG: tinta en papel / fósforo en pantalla
    "hero": ("#fffbfb", "#0c1424"),    # papel milimetrado / vidrio del monitor
    "cuadricula": ("rgba(214, 69, 69, .2)", "rgba(122, 146, 190, .07)"),
    "cuadricula-fina": ("rgba(214, 69, 69, .09)", "rgba(0, 0, 0, 0)"),
    "brillo-traza": ("rgba(0, 0, 0, 0)", "rgba(47, 224, 143, .55)"),
    "brillo-umbral": ("rgba(0, 0, 0, 0)", "rgba(255, 182, 72, .6)"),
}
OSCURO = {nombre: valores[1] for nombre, valores in TOKENS.items()}

# Colores con que se dibujan los gráficos de Plotly. Los de datos sirven en los dos
# temas (contraste >= 3:1 sobre blanco y sobre el panel oscuro); los de texto, rejilla
# y ejes se escriben con el valor oscuro y el CSS los cambia en el tema claro.
PANEL, PANEL_2, BORDE, BORDE_2 = OSCURO["panel"], OSCURO["panel-2"], OSCURO["borde"], OSCURO["borde-2"]
TINTA, TINTA_2, TINTA_3 = OSCURO["tinta"], OSCURO["tinta-2"], OSCURO["tinta-3"]
AMBAR = OSCURO["ambar"]
AZUL = "#4292f4"       # sin enfermedad / baja el riesgo
NARANJA = "#e7672b"    # con enfermedad / sube el riesgo
REJILLA = "#18233b"
EJE = BORDE_2
GRIS_BARRA = "#33415e"
TEXTO_CELDA = "#f8fafc"  # texto sobre celdas oscuras de los mapas de calor (igual en los dos temas)
COLOR_DX = {"Sin enfermedad": AZUL, "Con enfermedad": NARANJA}
GRAFICO_CLARO = {  # color oscuro con que se dibuja -> color en el tema claro
    TINTA: "#0f172a", TINTA_2: "#475066", TINTA_3: "#5d6880", REJILLA: "#e6e9ef", EJE: "#c9d0dc",
    GRIS_BARRA: "#b9c1ce", AMBAR: "#9a5a00", PANEL: "#ffffff", PANEL_2: "#ffffff", BORDE: "#e2e6ee",
}
FUENTE_CIFRAS = "Saira, 'Atkinson Hyperlegible Next', system-ui, sans-serif"

st.set_page_config(
    page_title="CardioRiesgo · Grupo 13",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="auto",
)

def _rgb(color_hex):
    return "rgb({}, {}, {})".format(*(int(color_hex[i:i + 2], 16) for i in (1, 3, 5)))


RAIZ = (":root {\n" + "\n".join(f"  --cr-{n}: light-dark({c}, {o});" for n, (c, o) in TOKENS.items())
        + f"\n  --cr-cifras: {FUENTE_CIFRAS};\n  --cr-ease-out: cubic-bezier(0.23, 1, 0.32, 1);\n}}\n")
# Plotly escribe los colores en línea (style="fill: rgb(...)"); estas reglas los cambian según el tema
CSS_GRAFICOS = "\n".join(
    f'.js-plotly-plot [style*="{prop}: {_rgb(oscuro)}"] {{{prop}: light-dark({claro}, {oscuro}) !important;}}'
    for oscuro, claro in GRAFICO_CLARO.items() for prop in ("fill", "stroke"))

CSS = """
/* Escala fluida (mobile-first): hasta 1600 px de ancho la base es 16 px; en monitores grandes
   y televisores crece hasta 30 px, y el ancho del contenido (en rem) crece con ella */
html {font-size: clamp(16px, calc(16px + (100vw - 1600px) * 0.0075), 30px) !important;}
.block-container, [data-testid="stMainBlockContainer"] {max-width: min(100%, 100rem);
  padding: 4.2rem clamp(1rem, 3.5vw, 4rem) 3rem;}
[data-testid="stHeading"] h3 {font-weight: 600; font-size: 1.45rem;}

/* Encabezado: marca, trazo de ECG y canales del monitor */
section.cr-hero {position: relative; padding: 1.375rem 1.75rem 1.25rem; border: 1px solid var(--cr-borde);
  border-radius: 18px; background: var(--cr-hero); overflow: hidden; margin: 0;}
section.cr-hero::before {content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(var(--cr-cuadricula-fina) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula-fina) 1px, transparent 1px);
  background-size: 30px 30px, 30px 30px, 6px 6px, 6px 6px;}
section.cr-hero > * {position: relative;}
section.cr-hero div.cr-hero-top {display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem 1rem;
  flex-wrap: wrap; margin: 0 0 0.75rem;}
section.cr-hero ul.cr-chips {display: flex; flex-wrap: wrap; gap: 0.5rem; list-style: none; margin: 0; padding: 0;}
/* El logo va sobre una placa blanca para que sus verdes se vean igual en los dos temas */
section.cr-hero img.cr-logo-cun {height: 2.75rem; width: auto; padding: 0.375rem 0.625rem; background: #ffffff;
  border-radius: 10px; border: 1px solid var(--cr-borde); flex: none;}
section.cr-hero ul.cr-chips li {margin: 0; padding: 0.3125rem 0.6875rem; font-size: .84rem; line-height: 1.2; color: var(--cr-tinta-2);
  border: 1px solid var(--cr-borde-2); border-radius: 999px; background: var(--cr-panel-2);}
section.cr-hero ul.cr-chips li::marker {content: none;}
section.cr-hero div.cr-marca {font-family: var(--cr-cifras); font-weight: 700; font-stretch: 125%;
  font-variation-settings: "wdth" 125; font-size: clamp(2.5rem, 3.2vw + 1.2rem, 4rem); line-height: 1;
  letter-spacing: -0.01em; color: var(--cr-tinta); margin: 0; padding: 0;}
section.cr-hero svg.cr-ecg {display: block; width: 100%; height: auto; margin: 0.375rem 0 0.25rem; overflow: visible;}
section.cr-hero .cr-traza {fill: none; stroke: var(--cr-traza); stroke-width: 2.6; stroke-linecap: round;
  stroke-linejoin: round; filter: drop-shadow(0 0 5px var(--cr-brillo-traza));}
section.cr-hero .cr-punto {fill: var(--cr-traza);}
section.cr-hero .cr-onda {fill: none; stroke: var(--cr-traza); stroke-width: 1.6; transform-box: fill-box;
  transform-origin: center; animation: cr-latido 1.1s var(--cr-ease-out) infinite;}
section.cr-hero p.cr-lema {color: var(--cr-tinta-2); font-size: 1.04rem; line-height: 1.55; max-width: 68ch; margin: 0;}

div.cr-fila {container-type: inline-size;}
div.cr-canales {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.75rem; margin: 0.875rem 0 0.625rem;}
div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: 1 / -1;}
@container (min-width: 38.75rem) {
  div.cr-canales.cr-n4 {grid-template-columns: repeat(4, minmax(0, 1fr));}
  div.cr-canales.cr-n6 {grid-template-columns: repeat(3, minmax(0, 1fr));}
}
@container (min-width: 47.5rem) {
  div.cr-canales.cr-n5 {grid-template-columns: repeat(5, minmax(0, 1fr));}
  div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: auto;}
}
@container (min-width: 60rem) {
  div.cr-canales.cr-n6 {grid-template-columns: repeat(6, minmax(0, 1fr));}
  div.cr-canales.cr-n6 .cr-canal-valor {font-size: 1.8rem;}
}
div.cr-canales .cr-canal {position: relative; display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 0.25rem;
  align-content: start; padding: 0.8125rem 0.9375rem 0.75rem; background: var(--cr-panel); border: 1px solid var(--cr-borde);
  border-radius: 12px; overflow: hidden; min-width: 0;}
div.cr-canales .cr-canal::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--canal);}
div.cr-canales .cr-canal-nombre {font-size: .88rem; line-height: 1.3; color: var(--cr-tinta-2); align-self: start;}
div.cr-canales .cr-canal-valor {font-family: var(--cr-cifras); font-weight: 600; font-size: 2.15rem; line-height: 1.1;
  color: var(--canal); font-variant-numeric: tabular-nums; white-space: nowrap; align-self: end;
  text-shadow: 0 0 18px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--canal) 35%, transparent));}
div.cr-canales .cr-canal-nota {font-size: .8rem; line-height: 1.35; color: var(--cr-tinta-3);}
div.cr-canales .cr-c-datos {--canal: var(--cr-tinta);}
div.cr-canales .cr-c-modelo {--canal: var(--cr-cian);}
div.cr-canales .cr-c-decision {--canal: var(--cr-verde);}

/* Animación de entrada: solo la primera vez que se abre la app */
div.cr-anim .cr-traza {stroke-dasharray: 1; stroke-dashoffset: 1; animation: cr-barrido 1.7s linear .15s forwards;}
div.cr-anim .cr-cabeza {opacity: 0; animation: cr-aparece 180ms ease-out 1.8s forwards;}
div.cr-anim .cr-canal {opacity: 0; transform: translateY(6px); animation: cr-entra 360ms var(--cr-ease-out) forwards;}
div.cr-anim .cr-canal:nth-child(1) {animation-delay: .55s;}
div.cr-anim .cr-canal:nth-child(2) {animation-delay: .61s;}
div.cr-anim .cr-canal:nth-child(3) {animation-delay: .67s;}
div.cr-anim .cr-canal:nth-child(4) {animation-delay: .73s;}
div.cr-anim .cr-canal:nth-child(5) {animation-delay: .79s;}
@keyframes cr-barrido {to {stroke-dashoffset: 0;}}
@keyframes cr-aparece {to {opacity: 1;}}
@keyframes cr-entra {to {opacity: 1; transform: none;}}
@keyframes cr-latido {from {transform: scale(1); opacity: .75;} to {transform: scale(3.2); opacity: 0;}}

/* Lectura del riesgo */
div.cr-lectura {--estado: var(--cr-verde); position: relative; padding: 1rem 1.25rem 0.875rem; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); border-radius: 16px; overflow: hidden;}
div.cr-lectura.cr-alto {--estado: var(--cr-rojo);}
div.cr-lectura::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--estado);}
div.cr-lectura .cr-lectura-cab {display: flex; justify-content: space-between; align-items: center; gap: 0.5rem 0.875rem;
  flex-wrap: wrap; font-size: .92rem; color: var(--cr-tinta-2);}
div.cr-lectura .cr-criterio {font-size: .82rem; color: var(--cr-tinta); padding: 0.25rem 0.625rem; border-radius: 999px;
  border: 1px solid var(--cr-borde-2); background: var(--cr-panel-2); white-space: nowrap;}
div.cr-lectura .cr-lectura-valor {font-family: var(--cr-cifras); font-weight: 600; font-size: clamp(4rem, 5vw + 1.4rem, 5.6rem);
  line-height: 1; margin: 0.875rem 0 0.75rem; color: var(--estado); font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
  text-shadow: 0 0 28px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--estado) 40%, transparent));}
div.cr-lectura .cr-unidad {font-size: .4em; font-weight: 500; margin-left: .14em; color: var(--cr-tinta-2); text-shadow: none;}
div.cr-lectura .cr-estado {display: flex; align-items: center; gap: 0.625rem; padding: 0.625rem 0.75rem; border-radius: 10px;
  font-size: .98rem; line-height: 1.4; color: var(--cr-tinta); background: color-mix(in srgb, var(--estado) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--estado) 38%, transparent);}
div.cr-lectura .cr-estado svg {flex: none; color: var(--estado);}
div.cr-lectura .cr-barra {position: relative; display: flex; gap: 0.1875rem; margin: 1.25rem 0 0.375rem;}
div.cr-lectura .cr-seg {flex: 1; height: 0.875rem; border-radius: 3px; background: var(--cr-panel-2);
  box-shadow: inset 0 0 0 1px var(--cr-borde);}
div.cr-lectura .cr-seg.on {background: var(--estado);
  box-shadow: 0 0 8px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--estado) 45%, transparent));}
div.cr-lectura .cr-umbral {position: absolute; top: -6px; bottom: -6px; width: 2px; margin-left: -0.0625rem; border-radius: 2px;
  background: var(--cr-ambar); box-shadow: 0 0 6px var(--cr-brillo-umbral);}
div.cr-lectura .cr-escala {position: relative; height: 20px; font-size: .8rem; color: var(--cr-tinta-3);}
div.cr-lectura .cr-escala span {position: absolute; top: 0; white-space: nowrap;}
div.cr-lectura .cr-escala .cr-e-umbral {color: var(--cr-ambar); transform: translateX(-50%);}
div.cr-lectura .cr-escala .cr-e-fin {right: 0;}
p.cr-nota {color: var(--cr-tinta-2); font-size: .94rem; line-height: 1.55; margin: 0.75rem 0.125rem 0;}
p.cr-nota b {color: var(--cr-tinta);}
@media (min-width: 992px) {
  [data-testid="stColumn"]:has(.cr-lectura) {position: sticky; top: 4.25rem; align-self: flex-start;}
}

/* Rótulos de grupo, avisos, fichas y pasos */
p.cr-rotulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.02rem; color: var(--cr-tinta); margin: 0 0 0.125rem;}
div.cr-aviso {display: flex; gap: 0.625rem; align-items: flex-start; padding: 0.75rem 0.875rem; border-radius: 12px;
  background: var(--cr-panel); border: 1px solid var(--cr-borde); color: var(--cr-tinta-2); font-size: .9rem; line-height: 1.5;}
div.cr-aviso svg {flex: none; margin-top: 0.0625rem; color: var(--cr-ambar);}
div.cr-fichas {display: grid; grid-template-columns: repeat(auto-fit, minmax(15.5rem, 1fr)); gap: 0.875rem; margin: 0.25rem 0 0.5rem;}
div.cr-fichas .cr-ficha {padding: 1.125rem 1.25rem; background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px;}
div.cr-fichas .cr-ficha-titulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.15rem; color: var(--cr-tinta);
  margin: 0 0 0.5rem; padding: 0;}
div.cr-fichas p {color: var(--cr-tinta-2); font-size: .96rem; line-height: 1.55; margin: 0;}
div.cr-fichas p b {color: var(--cr-tinta);}
dl.cr-dic {margin: 0; padding: 0;}
dl.cr-dic > div {padding: 0.5rem 0; border-bottom: 1px solid var(--cr-borde); margin: 0;}
dl.cr-dic > div:last-child {border-bottom: 0; padding-bottom: 0;}
dl.cr-dic dt {font-weight: 700; color: var(--cr-tinta); font-size: .95rem; margin: 0;}
dl.cr-dic dd {margin: 0.125rem 0 0; color: var(--cr-tinta-2); font-size: .9rem; line-height: 1.45;}
dl.cr-dic .cr-dic-val {display: block; margin-top: 0.125rem; font-family: var(--cr-cifras); font-size: .84rem; color: var(--cr-cian);}
ol.cr-pasos {list-style: none; counter-reset: paso; display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: 0.625rem; padding: 0; margin: 0.375rem 0 1.125rem;}
ol.cr-pasos li {counter-increment: paso; margin: 0; padding: 0.875rem 0.875rem 0.8125rem; border: 1px solid var(--cr-borde);
  border-radius: 12px; background: var(--cr-panel);}
ol.cr-pasos li::before {content: counter(paso); display: block; font-family: var(--cr-cifras); font-weight: 700;
  font-size: 1.5rem; line-height: 1; color: var(--cr-cian); margin-bottom: 0.5rem;}
ol.cr-pasos.cr-seis {grid-template-columns: repeat(2, 1fr);}
@media (min-width: 900px) {ol.cr-pasos.cr-seis {grid-template-columns: repeat(3, 1fr);}}
@media (min-width: 1300px) {ol.cr-pasos.cr-seis {grid-template-columns: repeat(6, 1fr);}}
ol.cr-pasos strong {display: block; color: var(--cr-tinta); font-size: 1rem; margin-bottom: 0.25rem;}
ol.cr-pasos span {display: block; color: var(--cr-tinta-2); font-size: .88rem; line-height: 1.45;}

/* Barra lateral */
div.cr-side .cr-side-marca {display: flex; align-items: center; gap: 0.625rem; margin: 0 0 0.375rem; color: var(--cr-tinta);
  font-family: var(--cr-cifras); font-weight: 700; font-size: 1.4rem; font-variation-settings: "wdth" 115;}
div.cr-side .cr-side-marca svg {color: var(--cr-verde); flex: none;}
div.cr-side p {font-size: .9rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0 0 0.25rem;}
div.cr-side .cr-side-titulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1rem; color: var(--cr-tinta);
  margin: 1.375rem 0 0.375rem; padding: 0;}
div.cr-side dl {margin: 0; padding: 0;}
div.cr-side dl > div {display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; padding: 0.4375rem 0;
  margin: 0; border-bottom: 1px solid var(--cr-borde);}
div.cr-side dt {font-size: .86rem; color: var(--cr-tinta-2); margin: 0; font-weight: 400;}
div.cr-side dd {margin: 0; padding: 0; font-family: var(--cr-cifras); font-size: .95rem; color: var(--cr-tinta); text-align: right; font-variant-numeric: tabular-nums;}
div.cr-side ul {margin: 0; padding: 0 0 0 1.05rem;}
div.cr-side li {font-size: .9rem; color: var(--cr-tinta); margin: 0.1875rem 0; padding: 0;}
div.cr-side p.cr-side-aviso {margin-top: 1.25rem; padding: 0.625rem 0.75rem; border-radius: 10px; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); font-size: .84rem;}

/* En pantallas grandes la barra lateral crece con la escala para que su texto no se parta */
@media (min-width: 1700px) {
  [data-testid="stSidebar"][data-collapsed="false"] {width: 19rem !important; min-width: 19rem !important;
    max-width: 19rem !important;}
}

.st-key-escala_pantalla {display: none !important;}

/* Gráficos dentro de una tarjeta */
[data-testid="stPlotlyChart"] {background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px; overflow: hidden;}

/* El selector de tema del menú se oculta para que el interruptor sea el único control
   (la app lo usa por dentro para cambiar de tema) */
[data-testid="stThemeSwitcher"],
[data-testid="stThemeSwitcher"] + [data-testid="stMainMenuDivider"] {display: none !important;}

/* Pestañas */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 0.25rem;}
[data-testid="stTabs"] button[data-baseweb="tab"] p {font-size: 1rem; font-weight: 600;}

/* Respuesta al presionar (botones) */
[data-testid^="stBaseButton"] {transition: transform 160ms var(--cr-ease-out), background-color 160ms ease,
  border-color 160ms ease, color 160ms ease;}
[data-testid^="stBaseButton"]:active {transform: scale(0.97);}

/* Entrada de cada pestaña: contexto en lenguaje sencillo y una pregunta guía */
div.cr-entrada {max-width: 62rem; margin: 0.25rem 0 1rem;}
div.cr-entrada p {font-size: 1.04rem; line-height: 1.6; color: var(--cr-tinta-2); margin: 0 0 0.5rem;}
p.cr-guia {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.12rem; color: var(--cr-tinta);
  border-left: 3px solid var(--cr-cian); padding: 0.125rem 0 0.125rem 0.75rem; margin: 0.625rem 0 0;}

/* El problema contado como una escena */
div.cr-relato {display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 0.875rem; margin: 0.25rem 0 1rem;}
div.cr-relato .cr-escena {position: relative; padding: 1.125rem 1.25rem 1.125rem 1.375rem; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); border-radius: 14px; overflow: hidden;}
div.cr-relato .cr-escena::before {content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--cr-cian);}
div.cr-relato .cr-e2::before {background: var(--cr-ambar);}
div.cr-relato .cr-e3::before {background: var(--cr-rojo);}
div.cr-relato p {font-size: 1rem; line-height: 1.6; color: var(--cr-tinta-2); margin: 0;}
div.cr-relato p b {color: var(--cr-tinta);}
div.cr-relato p.cr-guia {margin-top: 0.75rem;}

/* Flujo de la consulta a la decisión */
ol.cr-flujo {list-style: none; display: grid; grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr)); gap: 1.25rem;
  padding: 0; margin: 0.375rem 0 1.25rem;}
ol.cr-flujo li {position: relative; margin: 0; padding: 0.9375rem 0.9375rem 0.875rem; border-radius: 12px;
  background: var(--cr-panel); border: 1px solid var(--cr-borde); border-top: 3px solid var(--cr-cian);}
ol.cr-flujo li:nth-child(4) {border-top-color: var(--cr-ambar);}
ol.cr-flujo li:nth-child(5) {border-top-color: var(--cr-verde);}
ol.cr-flujo li:not(:last-child)::after {content: "→"; position: absolute; right: -1.05rem; top: 50%; transform: translateY(-50%);
  font-family: var(--cr-cifras); font-weight: 700; color: var(--cr-tinta-3); font-size: 1.1rem;}
ol.cr-flujo strong {display: block; color: var(--cr-tinta); font-family: var(--cr-cifras); font-size: 1.05rem; margin-bottom: 0.25rem;}
ol.cr-flujo span {display: block; color: var(--cr-tinta-2); font-size: .88rem; line-height: 1.45;}

/* Comparaciones con barras HTML (pistas y votos de los modelos) */
div.cr-pistas {display: grid; align-items: start; grid-template-columns: repeat(auto-fit, minmax(min(24rem, 100%), 1fr)); gap: 0.875rem; margin: 0.5rem 0 1rem;}
div.cr-pista {margin-top: 0.75rem; padding: 1rem 1.125rem; background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px;}
div.cr-pista .cr-pista-tit {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.05rem; color: var(--cr-tinta); margin: 0 0 0.625rem;}
div.cr-pista .cr-pista-nota {font-size: .86rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0.625rem 0 0;}
div.cr-comp {display: grid; grid-template-columns: minmax(6rem, 38%) 1fr 3.25rem; gap: 0.625rem; align-items: center; margin: 0.375rem 0;}
div.cr-comp .cr-comp-et {font-size: .9rem; color: var(--cr-tinta-2); line-height: 1.3;}
div.cr-comp .cr-comp-pista {height: 0.875rem; border-radius: 999px; background: var(--cr-panel-2); box-shadow: inset 0 0 0 1px var(--cr-borde); overflow: hidden;}
div.cr-comp .cr-comp-barra {display: block; height: 100%; border-radius: 999px; background: #4292f4;
  transition: width 400ms var(--cr-ease-out);}
div.cr-comp .cr-comp-barra.cr-alta {background: #e7672b;}
div.cr-comp .cr-comp-val {font-family: var(--cr-cifras); font-weight: 600; font-size: 1rem; color: var(--cr-tinta);
  text-align: right; font-variant-numeric: tabular-nums;}

/* Los 4 resultados posibles */
div.cr-resultados {display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 0.75rem; margin: 0.5rem 0 0.875rem;}
div.cr-res {display: grid; align-content: start; gap: 0.25rem; padding: 1rem 1.125rem; background: var(--cr-panel); border: 1px solid var(--cr-borde);
  border-radius: 14px; border-top: 4px solid var(--cr-c);}
div.cr-res-vp {--cr-c: var(--cr-verde);} div.cr-res-vn {--cr-c: var(--cr-cian);}
div.cr-res-fp {--cr-c: var(--cr-ambar);} div.cr-res-fn {--cr-c: var(--cr-rojo);}
div.cr-res .cr-res-num {font-family: var(--cr-cifras); font-weight: 700; font-size: 2.6rem; line-height: 1; color: var(--cr-c);
  font-variant-numeric: tabular-nums;}
div.cr-res .cr-res-tit {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.08rem; color: var(--cr-tinta);}
div.cr-res .cr-res-txt {font-size: .88rem; line-height: 1.45; color: var(--cr-tinta-2);}
div.cr-metricas {display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 0.75rem; margin: 0 0 0.875rem;}
div.cr-metrica {padding: 0.875rem 1rem; border-radius: 12px; background: var(--cr-panel-2); border: 1px solid var(--cr-borde);}
div.cr-metrica p {font-size: .9rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0;}
div.cr-metrica .cr-metrica-tit {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.05rem; color: var(--cr-tinta); margin-bottom: 0.25rem;}
div.cr-metrica .cr-metrica-nota {font-size: .84rem; margin-top: 0.375rem;}
span.cr-formula {display: block; margin-top: 0.25rem; font-family: var(--cr-cifras); font-weight: 600; color: var(--cr-cian);
  font-variant-numeric: tabular-nums;}

/* Bitácora del entrenamiento en vivo */
div.cr-bitacora {padding: 0.875rem 1rem; background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px;}
div.cr-bitacora ol {list-style: none; margin: 0.375rem 0 0; padding: 0;}
div.cr-bitacora li {display: flex; gap: 0.625rem; align-items: baseline; font-size: .9rem; line-height: 1.45; color: var(--cr-tinta-2);
  padding: 0.375rem 0; margin: 0; border-bottom: 1px solid var(--cr-borde);}
div.cr-bitacora li:first-child {color: var(--cr-tinta);}
div.cr-bitacora li:last-child {border-bottom: 0;}
span.cr-bit-paso {flex: none; min-width: 1.9rem; font-family: var(--cr-cifras); font-weight: 700; color: var(--cr-cian);
  font-variant-numeric: tabular-nums;}

/* Fichas con borde de color según lo que está en juego */
div.cr-fichas .cr-borde-rojo {border-top: 4px solid var(--cr-rojo);}
div.cr-fichas .cr-borde-ambar {border-top: 4px solid var(--cr-ambar);}
div.cr-fichas .cr-borde-verde {border-top: 4px solid var(--cr-verde);}
div.cr-side p.cr-side-como {margin: 0.25rem 0 0.75rem; font-size: .88rem;}

@media (max-width: 640px) {
  ol.cr-flujo {gap: 0.625rem;}
  ol.cr-flujo li:not(:last-child)::after {content: none;}
  div.cr-comp {grid-template-columns: 6.5rem 1fr 3rem;}
}
@media (prefers-reduced-motion: reduce) {
  div.cr-comp .cr-comp-barra {transition: none;}
}

@media (max-width: 640px) {
  section.cr-hero {padding: 1.125rem 1.125rem 1rem;}
  section.cr-hero .cr-traza {stroke-width: 6;}
  section.cr-hero .cr-onda {stroke-width: 3;}
  div.cr-canales .cr-canal-valor {font-size: 1.9rem;}
}
@media (prefers-reduced-motion: reduce) {
  div.cr-anim .cr-traza {animation: none; stroke-dashoffset: 0;}
  div.cr-anim .cr-cabeza {animation: none; opacity: 1;}
  div.cr-anim .cr-canal {transform: none; animation: cr-aparece 200ms ease forwards;}
  section.cr-hero .cr-onda {animation: none; opacity: 0;}
  [data-testid^="stBaseButton"]:active {transform: none;}
}
"""

st.html(f"<style>{RAIZ}{CSS}{CSS_GRAFICOS}</style>")

# Íconos (SVG propios, trazo de 2 px; decorativos junto a texto visible)
ICONO_ALERTA = ('<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3.6 2.6 20h18.8L12 3.6z"/>'
                '<path d="M12 10v4.4"/><path d="M12 17.3h.01"/></svg>')
ICONO_OK = ('<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/>'
            '<path d="m8 12.4 2.8 2.8L16.2 9.8"/></svg>')
ICONO_INFO = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/>'
              '<path d="M12 11v5.5"/><path d="M12 7.6h.01"/></svg>')
ICONO_MARCA = ('<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.8" '
               'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
               '<path d="M12 20.4s-7.6-4.5-9.3-9.3C1.5 7.7 3.5 4.6 6.9 4.6c2.1 0 3.7 1.2 5.1 3 1.4-1.8 3-3 5.1-3 '
               '3.4 0 5.4 3.1 4.2 6.5-1.7 4.8-9.3 9.3-9.3 9.3z"/><path d="M3.2 12.2h4.6l1.6-3 2.4 6.2 1.9-4.4 1 1.2h6.1"/></svg>')


# ---------------------------------------------------------------------------
# Utilidades de formato y piezas de interfaz
# ---------------------------------------------------------------------------
def num(x, dec=1):
    """Número con coma decimal (formato usado en Colombia)."""
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, dec=1):
    return f"{num(100 * x, dec)}\u00a0%"


def rgba(color_hex, alfa):
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alfa})"


def canal(nombre, valor, nota="", tipo="datos"):
    """Recuadro estilo monitor: nombre del canal, cifra grande y una nota."""
    nota_html = f'<span class="cr-canal-nota">{nota}</span>' if nota else ""
    return (f'<div class="cr-canal cr-c-{tipo}"><span class="cr-canal-nombre">{nombre}</span>'
            f'<span class="cr-canal-valor">{valor}</span>{nota_html}</div>')


def bloque(html):
    """Inserta HTML propio. Se usa st.markdown porque conserva los SVG en línea (st.html los elimina);
    se quitan sangrías y líneas vacías para que Markdown no lo convierta en bloques de código."""
    st.markdown("\n".join(linea.strip() for linea in html.strip().splitlines() if linea.strip()),
                unsafe_allow_html=True)


def fila_canales(canales):
    bloque(f'<div class="cr-fila"><div class="cr-canales cr-n{len(canales)}">{"".join(canales)}</div></div>')


def rotulo(texto):
    bloque(f'<p class="cr-rotulo">{texto}</p>')


def aviso(texto):
    bloque(f'<div class="cr-aviso">{ICONO_INFO}<span>{texto}</span></div>')


def trazo_ecg():
    """Trazo de ECG estilizado (complejos P-QRS-T) para el encabezado."""
    y = 56
    d = [f"M0 {y}"]
    for x in (40, 280, 520, 760, 1000):
        d += [f"L{x} {y}", f"Q{x + 12} {y - 14} {x + 24} {y}", f"L{x + 36} {y}", f"L{x + 40} {y + 5}",
              f"L{x + 47} {y - 48}", f"L{x + 54} {y + 15}", f"L{x + 59} {y}", f"L{x + 74} {y}",
              f"Q{x + 90} {y - 22} {x + 106} {y}"]
    d.append(f"L1172 {y}")
    return " ".join(d)


def estilo(fig, alto=340, leyenda=True, titulo=None):
    fig.update_layout(
        height=alto,
        margin=dict(l=14, r=18, t=(64 if leyenda else 54) if titulo else 30, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        separators=",.",
        font=dict(family=FUENTE_CIFRAS, color=TINTA_2, size=14),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None,
                    font=dict(color=TINTA_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDE_2, font=dict(family=FUENTE_CIFRAS, size=13, color=TINTA)),
        bargap=0.45,
    )
    if titulo:
        fig.update_layout(title=dict(text=titulo, x=0, xref="container", xanchor="left", y=0.98, yanchor="top",
                                     pad=dict(l=16, t=8), font=dict(family=FUENTE_CIFRAS, size=16, color=TINTA, weight=600)))
    fig.update_xaxes(automargin=True, showgrid=False, zeroline=False, linecolor=EJE, tickcolor=EJE, ticks="outside",
                     title_font=dict(color=TINTA_2, size=12), tickfont=dict(color=TINTA_3), title_standoff=10)
    fig.update_yaxes(automargin=True, gridcolor=REJILLA, gridwidth=1, zeroline=False, linecolor=EJE,
                     title_font=dict(color=TINTA_2, size=12), tickfont=dict(color=TINTA_3), title_standoff=10)
    return fig


def escalar_figura(fig, s):
    """Agranda letras, márgenes y alto de una figura por el factor de escala de la pantalla."""
    def r(v):
        return round(v * s, 1)
    L = fig.layout
    fig.update_layout(height=round((L.height or 340) * s), font_size=r(L.font.size or 14))
    if L.title.font.size:
        fig.update_layout(title_font_size=r(L.title.font.size))
    if L.hoverlabel.font.size:
        fig.update_layout(hoverlabel_font_size=r(L.hoverlabel.font.size))
    for lado in ("l", "r", "t", "b"):
        if getattr(L.margin, lado) is not None:
            setattr(L.margin, lado, round(getattr(L.margin, lado) * s))
    for nota in L.annotations:
        if nota.font.size:
            nota.font.size = r(nota.font.size)
    for eje in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        if eje.tickfont.size:
            eje.tickfont.size = r(eje.tickfont.size)
        if eje.title.font.size:
            eje.title.font.size = r(eje.title.font.size)
    for traza in fig.data:
        if getattr(traza, "textfont", None) is not None and traza.textfont.size:
            traza.textfont.size = r(traza.textfont.size)
    return fig


def mostrar(fig):
    if ESCALA > 1:
        escalar_figura(fig, ESCALA)
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})



# ---------------------------------------------------------------------------
# Carga de datos, metadatos y modelos
# ---------------------------------------------------------------------------
SEMILLA = 42
NOMBRE_ELEGIDO = "Regresión logística"
NOMBRES_MODELOS = ["Regresión logística", "SVM (RBF)", "Random Forest", "KNN", "Árbol de decisión"]
NOMBRE_CORTO = {"Regresión logística": "Logística", "SVM (RBF)": "SVM", "Random Forest": "Bosque",
                "KNN": "KNN", "Árbol de decisión": "Árbol"}
COMO_DECIDE = {
    "Regresión logística": "Le da un peso a cada dato del paciente, los suma y convierte el total en una "
                           "probabilidad. Es el modelo que elegimos: rinde igual o mejor que los demás y se puede explicar.",
    "SVM (RBF)": "Busca la frontera que deja el mayor margen posible entre pacientes sanos y enfermos, "
                 "y puede curvarla.",
    "Random Forest": "Reúne 300 árboles de decisión, cada uno entrenado con una parte distinta de los datos, "
                     "y promedia lo que vota cada árbol.",
    "KNN": "No aprende reglas: busca a los 5 pacientes del entrenamiento más parecidos y mira cuántos "
           "estaban enfermos.",
    "Árbol de decisión": "Hace preguntas de sí o no sobre los datos del paciente, una tras otra, hasta llegar "
                         "a una respuesta. Es fácil de leer, pero con pocos pacientes tiende a memorizar. Como casi siempre "
                         "responde 0 o 100 %, mover el umbral casi no cambia sus decisiones.",
}


@st.cache_data
def cargar_meta():
    return json.loads(RUTA_META.read_text(encoding="utf-8"))


@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


def crear_preprocesador(meta):
    """Mismo preprocesamiento del notebook: imputación, escalado y one-hot dentro del pipeline."""
    g = meta["grupos_variables"]
    return ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("escalar", StandardScaler())]), g["numericas"]),
        ("bin", SimpleImputer(strategy="most_frequent"), g["binarias"]),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), g["categoricas"]),
    ])


def crear_estimador(nombre, meta):
    """Hiperparámetros del notebook: ajustados para los tres mejores modelos y por defecto para los demás."""
    ajustados = {a["modelo"]: a["hiperparametros"] for a in meta["ajuste"]}
    if nombre == "Regresión logística":
        return LogisticRegression(**meta["modelo"]["hiperparametros"])
    if nombre == "SVM (RBF)":
        with warnings.catch_warnings():  # scikit-learn 1.9 anuncia el retiro de probability=True en 1.11
            warnings.simplefilter("ignore", FutureWarning)
            return SVC(probability=True, random_state=SEMILLA, **ajustados.get(nombre, {}))
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=SEMILLA,
                                      **ajustados.get(nombre, {}))
    if nombre == "KNN":
        return KNeighborsClassifier()
    return DecisionTreeClassifier(random_state=SEMILLA)


def curva_umbral(y, p):
    filas = []
    for u in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = p >= u
        filas.append({"umbral": float(u), "sensibilidad": float((pred & (y == 1)).sum() / max(1, (y == 1).sum())),
                      "especificidad": float((~pred & (y == 0)).sum() / max(1, (y == 0).sum()))})
    return pd.DataFrame(filas)


@st.cache_resource(show_spinner="Preparando los modelos…")
def cargar_modelos():
    """Entrena (o carga) los cinco modelos con los mismos 242 pacientes de entrenamiento del notebook.
    Para cada uno calcula sus probabilidades en prueba, su umbral de tamizaje con validación cruzada
    (el mayor umbral que detecta al menos al 90 % de los enfermos) y la importancia de cada dato."""
    import sklearn

    meta, datos = cargar_meta(), cargar_datos()
    V = meta["orden_variables"]
    tr, te = datos[datos["conjunto"] == "entrenamiento"], datos[datos["conjunto"] == "prueba"]
    x_tr, y_tr, x_te, y_te = tr[V], tr["enfermedad"].to_numpy(), te[V], te["enfermedad"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    auc_cv = {c["modelo"]: (c["roc_auc"], c["roc_auc_sd"]) for c in meta["comparacion_cv"]}
    auc_cv.update({a["modelo"]: (a["auc"], a["desv"]) for a in meta["ajuste"]})
    origen = f"modelo entrenado en el notebook (scikit-learn {meta['versiones']['scikit-learn']})"
    salida = {}
    for nombre in NOMBRES_MODELOS:
        nuevo = Pipeline([("prep", crear_preprocesador(meta)), ("modelo", crear_estimador(nombre, meta))])
        if nombre == NOMBRE_ELEGIDO:
            try:
                if meta["versiones"]["scikit-learn"] != sklearn.__version__:
                    raise RuntimeError("versión de scikit-learn distinta")
                modelo = joblib.load(RUTA_MODELO)
            except Exception:
                modelo = nuevo.fit(x_tr, y_tr)
                origen = f"modelo reconstruido con scikit-learn {sklearn.__version__} (mismos datos e hiperparámetros)"
            curva = pd.DataFrame(meta["umbral_cv"])
            u = meta["umbrales"]
            umbrales = {"tamizaje": u["tamizaje"]["valor"], "balanceado": u["balanceado"]["valor"]}
            cv_sens = {k: u[k]["sensibilidad_cv"] for k in u}
            cv_esp = {k: u[k]["especificidad_cv"] for k in u}
        else:
            p_oof = cross_val_predict(clone(nuevo), x_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
            modelo = nuevo.fit(x_tr, y_tr)
            curva = curva_umbral(y_tr, p_oof)
            # Si el modelo no llega al 90 % (el árbol solo da unas pocas probabilidades distintas),
            # se toma la mayor sensibilidad que sí alcanza.
            meta_sens = min(0.90, float(curva["sensibilidad"].max()))
            validos = curva[curva["sensibilidad"] >= meta_sens - 1e-9]
            umbrales = {"tamizaje": float(validos["umbral"].max()), "balanceado": 0.5}
            fila = {k: curva.iloc[(curva["umbral"] - v).abs().argmin()] for k, v in umbrales.items()}
            cv_sens = {k: float(f["sensibilidad"]) for k, f in fila.items()}
            cv_esp = {k: float(f["especificidad"]) for k, f in fila.items()}
        p_te = modelo.predict_proba(x_te)[:, 1]
        imp = permutation_importance(modelo, x_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEMILLA)
        salida[nombre] = {
            "modelo": modelo, "p_prueba": p_te, "auc_prueba": float(roc_auc_score(y_te, p_te)),
            "auc_cv": auc_cv.get(nombre, (np.nan, np.nan)), "curva": curva, "umbrales": umbrales,
            "sens_cv": cv_sens, "esp_cv": cv_esp,
            "importancia": pd.Series(imp.importances_mean, index=V).sort_values(ascending=False),
        }
    return salida, y_te, origen


meta = cargar_meta()
datos = cargar_datos()
VARS = meta["orden_variables"]
INFO = meta["variables"]
TIPOS = datos[VARS].dtypes.to_dict()
PRUEBA = meta["prueba"]
PROYECTO = meta["proyecto"]
ENTRENAMIENTO = datos[datos["conjunto"] == "entrenamiento"]
MODELOS, Y_PRUEBA, origen_modelo = cargar_modelos()
NOMBRE_CRITERIO = {"tamizaje": "Tamizaje", "balanceado": "Estándar"}
REF_DANE = ("Departamento Administrativo Nacional de Estadística. (2026, 20 de marzo). *En 2025pr disminuyen los "
            "nacimientos en Colombia en 4,5 % y aumentan las defunciones en 2,8 %* [Comunicado de prensa]. "
            "https://www.dane.gov.co/files/operaciones/EEVV/2026/20-mar-2026/cp-EEVV-Defunciones-IIsem2025pr.pdf")


def etiqueta_valor(var, valor):
    """Texto legible para el valor de una variable (códigos UCI -> español)."""
    info = INFO[var]
    if pd.isna(valor):
        return "Sin dato"
    if info["tipo"] in ("binaria", "categorica"):
        return info["opciones"].get(str(int(valor)), str(valor))
    unidad = f" {info['unidad']}" if info.get("unidad") else ""
    dec = 1 if var == "oldpeak" else 0
    return f"{num(float(valor), dec)}{unidad}"


@st.cache_data
def referencia_sana():
    """Paciente sano típico: mediana (numéricas) o valor más frecuente (categóricas) de los pacientes
    de entrenamiento sin enfermedad."""
    sanos = ENTRENAMIENTO[ENTRENAMIENTO["enfermedad"] == 0]
    return {v: (sanos[v].median() if INFO[v]["tipo"] == "numerica" else sanos[v].mode().iloc[0]) for v in VARS}


def predecir(df_pacientes, nombre=None):
    x = df_pacientes[VARS].astype(TIPOS)
    return MODELOS[nombre or MODELO_ACTIVO]["modelo"].predict_proba(x)[:, 1]


def aportes(paciente, nombre=None):
    """Cuánto cambia la probabilidad si cada dato del paciente fuera el de un paciente sano típico.
    Funciona igual para los cinco modelos. Positivo = ese dato sube el riesgo."""
    ref = referencia_sana()
    variantes = pd.concat([paciente] * len(VARS), ignore_index=True)
    for i, v in enumerate(VARS):
        variantes.loc[i, v] = ref[v]
    p0 = float(predecir(paciente, nombre)[0])
    return pd.Series(p0 - predecir(variantes, nombre), index=VARS)


def lectura(prob, umbral, criterio):
    """Lectura principal del riesgo, como el número grande de un monitor."""
    alto = prob >= umbral
    estado = (f"{ICONO_ALERTA}<span><b>Riesgo alto.</b> Se sugiere estudio cardiológico.</span>" if alto
              else f"{ICONO_OK}<span><b>Riesgo bajo</b> según el modelo.</span>")
    encendidos = max(1, int(round(prob * 20))) if prob > 0 else 0
    segmentos = "".join(f'<span class="cr-seg{" on" if i < encendidos else ""}"></span>' for i in range(20))
    return f"""
<div class="cr-lectura {'cr-alto' if alto else 'cr-bajo'}" role="status" aria-live="polite">
  <div class="cr-lectura-cab"><span>Probabilidad estimada</span>
    <span class="cr-criterio">{NOMBRE_CRITERIO[criterio]}, umbral {pct(umbral, 0)}</span></div>
  <div class="cr-lectura-valor">{num(100 * prob, 0)}<span class="cr-unidad">%</span></div>
  <div class="cr-estado">{estado}</div>
  <div class="cr-barra" aria-hidden="true">{segmentos}<span class="cr-umbral" style="left:{100 * umbral:.1f}%"></span></div>
  <div class="cr-escala" aria-hidden="true"><span>0 %</span>
    <span class="cr-e-umbral" style="left:{100 * umbral:.1f}%">umbral {pct(umbral, 0)}</span>
    <span class="cr-e-fin">100 %</span></div>
</div>"""


def intro(texto, pregunta=None):
    """Párrafo de entrada de cada pestaña, en lenguaje sencillo, con una pregunta guía."""
    guia = f'<p class="cr-guia">{pregunta}</p>' if pregunta else ""
    bloque(f'<div class="cr-entrada"><p>{texto}</p>{guia}</div>')


def seccion(titulo):
    st.subheader(titulo)


def tasa(var, valor, tabla=None):
    t = datos if tabla is None else tabla
    sub = t[t[var] == valor]
    return float(sub["enfermedad"].mean()), len(sub)


# ---------------------------------------------------------------------------
# Barra lateral: modelo activo y datos del grupo
# ---------------------------------------------------------------------------
integrantes = "".join(f"<li>{n}</li>" for n in PROYECTO["integrantes"])
with st.sidebar:
    bloque(f"""
<div class="cr-side">
  <div class="cr-side-marca">{ICONO_MARCA}<span>CardioRiesgo</span></div>
  <p>Riesgo de enfermedad coronaria con machine learning clásico.</p>
</div>""")
    MODELO_ACTIVO = st.selectbox(
        "Modelo activo", NOMBRES_MODELOS, key="modelo", bind="query-params",
        help="Cambia el modelo que usan las cifras del encabezado, los resultados y la evaluación de pacientes.")
    M = MODELOS[MODELO_ACTIVO]
    auc_cv, auc_sd = M["auc_cv"]
    bloque(f"""
<div class="cr-side">
  <p class="cr-side-como">{COMO_DECIDE[MODELO_ACTIVO]}</p>
  <dl>
    <div><dt>AUC validación cruzada</dt><dd>{num(auc_cv, 3)}</dd></div>
    <div><dt>AUC prueba</dt><dd>{num(M['auc_prueba'], 3)}</dd></div>
    <div><dt>Umbral de tamizaje</dt><dd>{num(M['umbrales']['tamizaje'], 2)}</dd></div>
    <div><dt>Pacientes</dt><dd>242 entrenan · 61 prueban</dd></div>
  </dl>
  <p class="cr-side-titulo">Grupo 13</p>
  <ul>{integrantes}</ul>
  <p style="margin-top:10px">Aprendizaje Automático, CUN<br>Docente: {PROYECTO['docente']}</p>
  <p class="cr-side-aviso">Herramienta académica. No reemplaza el criterio médico ni un examen diagnóstico.</p>
</div>""")
    st.caption(f"Regresión logística: {origen_modelo}. Los otros cuatro modelos se entrenan al abrir la app "
               "con los mismos pacientes e hiperparámetros del notebook.")

# ---------------------------------------------------------------------------
# Encabezado: marca, trazo de ECG y canales del monitor
# ---------------------------------------------------------------------------
# Tema: claro por defecto. El interruptor queda enlazado a la URL (?oscuro=true), así que
# el modo elegido se mantiene al recargar o al compartir el enlace.
with st.container(horizontal_alignment="right"):
    MODO_OSCURO = st.toggle(":material/dark_mode: Modo oscuro", key="oscuro", bind="query-params",
                            help="Cambia entre el tema claro (por defecto) y el oscuro.")
    # Escala de la pantalla (base tipográfica / 16 px), la escribe el navegador; ver ESCALA_PANTALLA
    st.text_input("Escala de pantalla", key="escala_pantalla", label_visibility="collapsed")

try:
    ESCALA = min(2.0, max(1.0, float(st.session_state.get("escala_pantalla") or 1)))
except ValueError:
    ESCALA = 1.0
try:
    LOGO_HTML = ('<img class="cr-logo-cun" alt="Corporación Unificada Nacional de Educación Superior, CUN" '
                 f'src="data:image/png;base64,{base64.b64encode(RUTA_LOGO.read_bytes()).decode()}">')
except OSError:
    LOGO_HTML = ""
primera_vez = not st.session_state.get("_intro_vista", False)
st.session_state["_intro_vista"] = True
p_act = M["p_prueba"]
u_tam = M["umbrales"]["tamizaje"]
vp_t = int(((p_act >= u_tam) & (Y_PRUEBA == 1)).sum())
canales_inicio = [
    canal("Pacientes", meta["dataset"]["n_pacientes"], "de la Cleveland Clinic, con angiografía"),
    canal("Datos por paciente", len(VARS), "consulta, laboratorio y prueba de esfuerzo"),
    canal("Modelo activo", NOMBRE_CORTO[MODELO_ACTIVO],
          ("Random Forest" if MODELO_ACTIVO == "Random Forest" else "cámbielo en la barra lateral"), "modelo"),
    canal("AUC (validación cruzada)", num(auc_cv, 3), "1 es perfecto; 0,5 es azar", "modelo"),
    canal("Enfermos detectados", f"{vp_t} de {int(Y_PRUEBA.sum())}",
          f"en los 61 pacientes de prueba, umbral de tamizaje {num(u_tam, 2)}", "decision"),
]
bloque(f"""
<div class="cr-intro cr-fila{' cr-anim' if primera_vez else ''}">
<section class="cr-hero" aria-label="Presentación">
  <div class="cr-hero-top">
  <ul class="cr-chips" aria-label="Contexto del proyecto">
    <li>ACA 1</li><li>Grupo 13</li><li>Aprendizaje Automático</li><li>Especialización en IA, CUN</li>
  </ul>{LOGO_HTML}
  </div>
  <div class="cr-marca" role="heading" aria-level="1">CardioRiesgo</div>
  <svg class="cr-ecg" viewBox="0 0 1200 80" aria-hidden="true" focusable="false">
    <path class="cr-traza" pathLength="1" d="{trazo_ecg()}"/>
    <g class="cr-cabeza"><circle class="cr-onda" cx="1172" cy="56" r="4.5"/><circle class="cr-punto" cx="1172" cy="56" r="4.5"/></g>
  </svg>
  <p class="cr-lema">Con los datos de una consulta y una prueba de esfuerzo, estimamos qué tan probable es que un
  paciente tenga una arteria del corazón obstruida, antes de llegar a un examen invasivo.</p>
</section>
<div class="cr-canales cr-n5" aria-label="Resumen del estudio">{''.join(canales_inicio)}</div>
</div>""")

(tab_problema, tab_datos, tab_aprende, tab_resultados, tab_pac, tab_lote, tab_acerca) = st.tabs([
    ":material/favorite: El problema",
    ":material/dataset: Los datos",
    ":material/model_training: Cómo aprende",
    ":material/leaderboard: Resultados",
    ":material/monitor_heart: Evaluar paciente",
    ":material/table_view: Varios pacientes",
    ":material/menu_book: El proyecto",
])

# ---------------------------------------------------------------------------
# Pestaña 1: el problema, en lenguaje sencillo
# ---------------------------------------------------------------------------
with tab_problema:
    seccion("Una consulta, una pregunta")
    bloque("""
<div class="cr-relato">
  <div class="cr-escena cr-e1"><p>Un hombre de 58 años llega a consulta porque siente presión en el pecho cuando sube
  escaleras. El médico le toma la presión, pide exámenes de sangre, un electrocardiograma y una prueba de esfuerzo
  en caminadora. Con eso tiene pistas, pero todavía no tiene la respuesta.</p></div>
  <div class="cr-escena cr-e2"><p>La respuesta definitiva la da una <b>angiografía coronaria</b>: un catéter llega
  hasta el corazón y un medio de contraste muestra si alguna arteria está estrechada en más de la mitad. Es un
  examen invasivo, costoso y que no está disponible en todos los hospitales.</p></div>
  <div class="cr-escena cr-e3"><p>En Colombia, las enfermedades isquémicas del corazón son la primera causa de
  muerte: 92,7 por cada 100.000 habitantes en 2025, según cifras preliminares del DANE. Decidir a tiempo quién
  necesita ese examen importa.</p>
  <p class="cr-guia">¿Pueden los datos que ya se tomaron en la consulta anticipar si la angiografía saldrá positiva?</p></div>
</div>""")

    seccion("De la consulta a la decisión")
    bloque("""
<ol class="cr-flujo" aria-label="Cómo funciona CardioRiesgo">
  <li><strong>13 datos del paciente</strong><span>edad, síntomas, presión, colesterol, electrocardiograma y prueba de esfuerzo</span></li>
  <li><strong>Modelo de machine learning</strong><span>aprendió de 242 pacientes cuyo resultado ya se conocía</span></li>
  <li><strong>Probabilidad</strong><span>de 0 a 100 %, qué tan probable es una arteria obstruida</span></li>
  <li><strong>Umbral</strong><span>desde qué probabilidad conviene estudiar más al paciente</span></li>
  <li><strong>Decisión</strong><span>riesgo alto: estudio cardiológico; riesgo bajo: control</span></li>
</ol>""")

    seccion("Las pistas que más pesan")
    st.markdown("Estas son las señales que más separan a los pacientes sanos de los enfermos en los 303 casos. "
                "Cada barra muestra qué parte de los pacientes con esa característica tenía la enfermedad.")

    def comparacion_pista(titulo, filas, nota):
        barras = "".join(
            f'<div class="cr-comp"><span class="cr-comp-et">{et}</span><span class="cr-comp-pista">'
            f'<span class="cr-comp-barra{" cr-alta" if t >= 0.5 else ""}" style="width:{100 * t:.0f}%"></span></span>'
            f'<span class="cr-comp-val">{pct(t, 0)}</span></div>' for et, t in filas)
        return f'<div class="cr-pista"><p class="cr-pista-tit">{titulo}</p>{barras}<p class="cr-pista-nota">{nota}</p></div>'

    pistas = [
        comparacion_pista("Prueba de talio (imagen del corazón en esfuerzo)",
                          [("Normal", tasa("thal", 3)[0]), ("Defecto reversible", tasa("thal", 7)[0])],
                          "Si una zona del corazón recibe menos sangre al hacer esfuerzo, la enfermedad es mucho más frecuente."),
        comparacion_pista("Arterias principales que se ven en la fluoroscopia",
                          [("Ninguna", tasa("ca", 0)[0]), ("Tres", tasa("ca", 3)[0])],
                          "Cuantas más arterias se marcan en la imagen, más probable es la obstrucción."),
        comparacion_pista("Tipo de dolor en el pecho",
                          [("Angina típica", tasa("cp", 1)[0]), ("Sin dolor (asintomático)", tasa("cp", 4)[0])],
                          "Parece contradictorio: muchos pacientes enfermos no sienten el dolor clásico. Por eso no basta con preguntar por el dolor."),
        comparacion_pista("Dolor en el pecho al hacer ejercicio",
                          [("No", tasa("exang", 0)[0]), ("Sí", tasa("exang", 1)[0])],
                          "Si el pecho duele en la caminadora, el corazón no está recibiendo la sangre que necesita."),
    ]
    bloque(f'<div class="cr-pistas">{"".join(pistas)}</div>')

    g1, g2 = st.columns([1.3, 1], gap="large")
    with g1:
        # Mayores de 60 en un solo grupo: de 71 a 77 años solo hay 6 pacientes
        grupos = pd.cut(datos["age"], [28, 40, 50, 60, 80], labels=["29-40", "41-50", "51-60", "61-77"])
        t = (datos.assign(grupo=grupos).groupby(["grupo", "sex"], observed=True)["enfermedad"]
             .agg(enfermedad="mean", n="count").reset_index())
        fig = go.Figure()
        for sexo, nombre, color in [(0, "Mujeres", AZUL), (1, "Hombres", NARANJA)]:
            sub = t[t["sex"] == sexo]
            fig.add_trace(go.Scatter(x=sub["grupo"].astype(str), y=100 * sub["enfermedad"], name=nombre, mode="lines+markers",
                                     line=dict(color=color, width=3, shape="spline", smoothing=0.6),
                                     marker=dict(size=10, color=color, line=dict(color=PANEL, width=2)),
                                     customdata=sub["n"],
                                     hovertemplate="%{x} años<br>" + nombre + ": %{y:.0f} % con enfermedad"
                                                   "<br>%{customdata} pacientes<extra></extra>"))
        estilo(fig, alto=340, titulo="Con la edad sube el riesgo, y más en los hombres")
        fig.update_xaxes(title="Edad (años)")
        fig.update_yaxes(title="% con enfermedad", range=[0, 100], ticksuffix=" %")
        mostrar(fig)
    with g2:
        n_enf = int(datos["enfermedad"].sum())
        fig = go.Figure(go.Pie(labels=["Con enfermedad", "Sin enfermedad"], values=[n_enf, len(datos) - n_enf],
                               hole=0.62, sort=False, marker=dict(colors=[NARANJA, AZUL], line=dict(color=PANEL, width=3)),
                               textinfo="percent", textposition="inside", textfont=dict(size=15, color=TEXTO_CELDA),
                               insidetextorientation="horizontal",
                               hovertemplate="%{label}: %{value} pacientes<extra></extra>"))
        fig.add_annotation(text=f"<b>{len(datos)}</b><br>pacientes", showarrow=False, font=dict(size=18, color=TINTA))
        estilo(fig, alto=340, titulo="Casi la mitad tenía la enfermedad")
        fig.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", yanchor="top"))
        mostrar(fig)

    seccion("Lo que está en juego")
    bloque("""
<div class="cr-fichas">
  <div class="cr-ficha cr-borde-rojo"><p class="cr-ficha-titulo">Si el modelo falla con un enfermo</p><p>El paciente
  se va a casa sin el estudio que necesitaba. Es el error más grave: un <b>falso negativo</b>.</p></div>
  <div class="cr-ficha cr-borde-ambar"><p class="cr-ficha-titulo">Si el modelo falla con un sano</p><p>Se le pide un
  examen de más. Cuesta tiempo y dinero, pero el médico lo descarta después: es un <b>falso positivo</b>.</p></div>
  <div class="cr-ficha cr-borde-verde"><p class="cr-ficha-titulo">Por eso usamos un umbral de tamizaje</p><p>Preferimos
  equivocarnos por precaución: el modelo marca riesgo alto aunque la probabilidad no sea tan alta, para no dejar
  pasar a los enfermos.</p></div>
</div>""")

# ---------------------------------------------------------------------------
# Pestaña 2: los datos
# ---------------------------------------------------------------------------
QUE_ES = {
    "age": ("Datos generales", "Edad del paciente."),
    "sex": ("Datos generales", "Sexo biológico."),
    "cp": ("Síntomas", "Cómo describe el dolor en el pecho: típico de angina, atípico, otro o ninguno."),
    "exang": ("Síntomas", "Si aparece dolor en el pecho durante la prueba de esfuerzo."),
    "trestbps": ("Signos y laboratorio", "Presión arterial sistólica en reposo, al llegar."),
    "chol": ("Signos y laboratorio", "Colesterol total en sangre."),
    "fbs": ("Signos y laboratorio", "Si el azúcar en ayunas pasa de 120 mg/dl."),
    "restecg": ("Electrocardiograma y esfuerzo", "Resultado del electrocardiograma en reposo."),
    "thalach": ("Electrocardiograma y esfuerzo", "Pulso más alto que alcanza en la caminadora."),
    "oldpeak": ("Electrocardiograma y esfuerzo", "Cuánto baja una parte del trazo del ECG (segmento ST) con el esfuerzo."),
    "slope": ("Electrocardiograma y esfuerzo", "Si ese segmento sube, queda plano o baja en el pico del esfuerzo."),
    "ca": ("Imágenes", "Arterias principales que se ven marcadas en la fluoroscopia (0 a 3)."),
    "thal": ("Imágenes", "Gammagrafía con talio: si el corazón recibe sangre normal en reposo y en esfuerzo."),
}

with tab_datos:
    intro("Los datos vienen de 303 pacientes que llegaron a la Cleveland Clinic (Estados Unidos) por sospecha de "
          "enfermedad coronaria y a quienes se les hizo una angiografía. El conjunto se publicó en 1988 y hoy está en el "
          "repositorio de la Universidad de California en Irvine (UCI). Cada paciente tiene 13 datos y un resultado: "
          "si la angiografía mostró una arteria estrechada en más del 50 %.",
          "¿Qué se sabía de cada paciente antes del examen invasivo?")
    n_falt = int(datos[VARS].isna().sum().sum())
    n_h = int((datos["sex"] == 1).sum())
    fila_canales([
        canal("Con enfermedad", int(datos["enfermedad"].sum()), pct(datos["enfermedad"].mean()) + " del total", "decision"),
        canal("Sin enfermedad", int((datos["enfermedad"] == 0).sum()), "angiografía normal", "modelo"),
        canal("Hombres · mujeres", f"{n_h} · {len(datos) - n_h}", f"edades de {int(datos['age'].min())} a {int(datos['age'].max())} años"),
        canal("Datos faltantes", n_falt, "4 en arterias y 2 en talio; se completan con el valor típico"),
    ])

    seccion("Cómo se obtuvo cada dato")
    bloque("""
<ol class="cr-pasos cr-seis">
  <li><strong>Consulta</strong><span>Edad, sexo y tipo de dolor en el pecho.</span></li>
  <li><strong>Signos y laboratorio</strong><span>Presión en reposo, colesterol y azúcar en ayunas.</span></li>
  <li><strong>Electrocardiograma</strong><span>El trazo del corazón en reposo.</span></li>
  <li><strong>Prueba de esfuerzo</strong><span>Pulso máximo, dolor al ejercicio y cambios del trazo.</span></li>
  <li><strong>Imágenes</strong><span>Fluoroscopia de las arterias y gammagrafía con talio.</span></li>
  <li><strong>Angiografía</strong><span>El resultado que el modelo aprende a anticipar.</span></li>
</ol>""")

    seccion("Los 13 datos, en palabras sencillas")

    def valores_de(v):
        if INFO[v]["tipo"] != "numerica":
            return " · ".join(INFO[v]["opciones"].values())
        dec = 1 if v == "oldpeak" else 0
        return f"{num(datos[v].min(), dec)} a {num(datos[v].max(), dec)} {INFO[v].get('unidad', '')}".strip()

    fichas_dic = []
    for grupo in dict.fromkeys(g for g, _ in QUE_ES.values()):
        items = "".join(f'<div><dt>{INFO[v]["etiqueta"]}</dt><dd>{QUE_ES[v][1]}<span class="cr-dic-val">{valores_de(v)}</span></dd></div>'
                        for v in QUE_ES if QUE_ES[v][0] == grupo)
        fichas_dic.append(f'<div class="cr-ficha"><p class="cr-ficha-titulo">{grupo}</p><dl class="cr-dic">{items}</dl></div>')
    bloque(f'<div class="cr-fichas">{"".join(fichas_dic)}</div>')

    seccion("Explorar los pacientes")
    with st.container(border=True):
        f1, f2, f3 = st.columns([1, 1.3, 1.3])
        sexo_sel = f1.radio("Sexo", ["Todos", "Mujer", "Hombre"], horizontal=True)
        edad_sel = f2.slider("Rango de edad", int(datos["age"].min()), int(datos["age"].max()),
                             (int(datos["age"].min()), int(datos["age"].max())))
        conj_sel = f3.radio("Conjunto", ["Todos", "entrenamiento", "prueba"], horizontal=True,
                            format_func=lambda c: c.capitalize())
    vista = datos.copy()
    if sexo_sel != "Todos":
        vista = vista[vista["sex"] == (1 if sexo_sel == "Hombre" else 0)]
    vista = vista[vista["age"].between(*edad_sel)]
    if conj_sel != "Todos":
        vista = vista[vista["conjunto"] == conj_sel]

    if vista.empty:
        st.warning("No hay pacientes con esa combinación de filtros. Amplíe el rango de edad o cambie el sexo.")
    else:
        fila_canales([
            canal("Pacientes", f"{len(vista):,}".replace(",", ".")),
            canal("Con enfermedad", pct(vista["enfermedad"].mean()), f"{int(vista['enfermedad'].sum())} pacientes"),
            canal("Edad mediana", f"{num(vista['age'].median(), 0)} años"),
            canal("Pulso máximo mediano", f"{num(vista['thalach'].median(), 0)} lpm", "en la prueba de esfuerzo"),
        ])

        NUMERICAS = [v for v in VARS if INFO[v]["tipo"] == "numerica" and v != "ca"]
        var_v = st.selectbox("Comparar un dato entre sanos y enfermos", NUMERICAS, index=NUMERICAS.index("thalach"),
                             format_func=lambda v: INFO[v]["etiqueta"])
        c1, c2 = st.columns([1.15, 1], gap="large")
        with c1:
            fig = go.Figure()
            for dx, val in [("Sin enfermedad", 0), ("Con enfermedad", 1)]:
                sub = vista[vista["enfermedad"] == val]
                fig.add_trace(go.Violin(y=sub[var_v], name=dx, line_color=COLOR_DX[dx], fillcolor=rgba(COLOR_DX[dx], 0.28),
                                        box=dict(visible=True, fillcolor=rgba(COLOR_DX[dx], 0.5), line=dict(color=COLOR_DX[dx])),
                                        meanline=dict(visible=False), points="all", pointpos=0, jitter=0.35,
                                        marker=dict(size=5, color=COLOR_DX[dx], opacity=0.55), spanmode="hard",
                                        hoveron="points", hovertemplate=f"{INFO[var_v]['etiqueta_corta']}: %{{y}}<extra>{dx}</extra>"))
            unidad = f" ({INFO[var_v]['unidad']})" if INFO[var_v].get("unidad") else ""
            estilo(fig, alto=380, leyenda=False, titulo=f"{INFO[var_v]['etiqueta']}: sanos frente a enfermos")
            fig.update_yaxes(title=INFO[var_v]["etiqueta_corta"] + unidad)
            mostrar(fig)
        with c2:
            fig = go.Figure()
            for dx, val in [("Sin enfermedad", 0), ("Con enfermedad", 1)]:
                sub = vista[vista["enfermedad"] == val]
                fig.add_trace(go.Scatter(
                    x=sub["age"], y=sub["thalach"], mode="markers", name=dx,
                    marker=dict(color=COLOR_DX[dx], size=9, line=dict(color=PANEL, width=1.5), opacity=0.9),
                    hovertemplate="Edad %{x} · pulso máx. %{y} lpm<extra>" + dx + "</extra>"))
            estilo(fig, alto=380, titulo="Los enfermos alcanzan un pulso máximo más bajo")
            fig.update_xaxes(title="Edad (años)")
            fig.update_yaxes(title="Pulso máximo en esfuerzo (lpm)")
            mostrar(fig)

        def barra_tasa(var, titulo):
            t = vista.groupby(var, dropna=True)["enfermedad"].agg(["mean", "count"]).reset_index()
            t["Categoría"] = [etiqueta_valor(var, v) for v in t[var]]
            if var == "ca":
                t["Categoría"] = [f"{c} arteria" if c == "1" else f"{c} arterias" for c in t["Categoría"]]
            fig = go.Figure(go.Bar(
                y=t["Categoría"], x=100 * t["mean"], orientation="h", width=0.6,
                marker=dict(color=[NARANJA if m >= 0.5 else AZUL for m in t["mean"]], cornerradius=6),
                text=[pct(v, 0) for v in t["mean"]], textposition="outside", textfont=dict(color=TINTA_2),
                customdata=t["count"], cliponaxis=False,
                hovertemplate="%{y}<br>Con enfermedad: %{x:.1f} %<br>Pacientes: %{customdata}<extra></extra>"))
            estilo(fig, alto=300, leyenda=False, titulo=titulo)
            fig.update_xaxes(title="% con enfermedad", range=[0, 118], showgrid=True, gridcolor=REJILLA,
                             tickvals=[0, 25, 50, 75, 100])
            fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
            return fig

        g3, g4, g5 = st.columns(3)
        with g3:
            mostrar(barra_tasa("cp", "Según el tipo de dolor"))
        with g4:
            mostrar(barra_tasa("thal", "Según la prueba de talio"))
        with g5:
            mostrar(barra_tasa("ca", "Según las arterias marcadas"))
        st.caption("Barras naranjas: la mitad o más de esos pacientes tenía la enfermedad.")

        with st.expander("Detalle técnico: correlaciones y tabla de pacientes"):
            cols_corr = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "enfermedad"]
            corr = vista[cols_corr].corr()
            nombres = [INFO[c]["etiqueta_corta"] if c in INFO else "Enfermedad" for c in cols_corr]
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=nombres, y=nombres, zmin=-1, zmax=1,
                colorscale=[[0, "#2a62ad"], [0.5, "#5f6b7f"], [1, "#b04d1b"]],
                text=[[num(round(v, 2) + 0.0, 2) for v in fila] for fila in corr.values], texttemplate="%{text}",
                textfont=dict(size=11, color=TEXTO_CELDA), xgap=2, ygap=2,
                hovertemplate="%{y} · %{x}<br>r = %{z:.2f}<extra></extra>",
                colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=TINTA_3))))
            estilo(fig, alto=400, leyenda=False, titulo="Correlación entre variables numéricas")
            fig.update_xaxes(tickangle=-30, showgrid=False, linecolor="rgba(0,0,0,0)", ticks="")
            fig.update_yaxes(autorange="reversed", showgrid=False, linecolor="rgba(0,0,0,0)")
            mostrar(fig)
            tabla = vista[VARS + ["enfermedad", "conjunto"]].copy()
            tabla["diagnóstico"] = tabla["enfermedad"].map({0: "Sin enfermedad", 1: "Con enfermedad"})
            st.dataframe(tabla.drop(columns="enfermedad"), hide_index=True, height=320)
            st.download_button("Descargar selección (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                               "pacientes_filtrados.csv", "text/csv", icon=":material/download:")

# ---------------------------------------------------------------------------
# Pestaña 3: cómo aprende un modelo (entrenamiento en vivo)
# ---------------------------------------------------------------------------
ALGORITMOS_LAB = {
    "Regresión logística": ("época", "Ajusta un peso por cada dato con descenso de gradiente: en cada época mira cuánto "
                            "se equivoca y mueve los pesos para equivocarse menos. Su frontera siempre es una línea recta."),
    "Red neuronal": ("época", "Dos capas de 12 neuronas. En cada época ajusta sus conexiones; por eso su frontera "
                     "puede curvarse."),
    "Árbol de decisión": ("pregunta", "Cada paso le permite hacer una pregunta más de sí o no. Con muchas preguntas "
                          "termina memorizando a los pacientes del entrenamiento."),
    "Random Forest": ("árbol", "Cada paso añade un árbol al bosque. La frontera es el voto promedio de todos los árboles, "
                      "así que se vuelve más estable a medida que crece."),
    "KNN": ("vecinos", "KNN no aprende reglas: guarda a los pacientes y, para decidir, mira a los más parecidos. Cada paso "
            "consulta a más vecinos, y la frontera pasa de nerviosa a suave."),
    "SVM (RBF)": ("rigidez", "Cada paso le da más libertad (parámetro C) para curvar la frontera alrededor de los "
                  "pacientes difíciles."),
}
VARS_LAB = ["thalach", "oldpeak", "age", "trestbps", "chol"]


def _sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


@st.cache_data(show_spinner="Entrenando paso a paso…")
def entrenar_en_vivo(algoritmo, vx, vy, n_grilla=60):
    """Entrena el algoritmo con dos datos del paciente y guarda cada paso: probabilidades sobre una grilla
    (para dibujar la frontera), acierto en entrenamiento y validación, pérdida y una línea de bitácora."""
    base = ENTRENAMIENTO[[vx, vy, "enfermedad"]].dropna()
    x, y = base[[vx, vy]].to_numpy(float), base["enfermedad"].to_numpy()
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEMILLA)
    media, desv = x_tr.mean(0), x_tr.std(0)
    z_tr, z_va = (x_tr - media) / desv, (x_va - media) / desv
    lo, hi = np.vstack([z_tr, z_va]).min(0) - 0.4, np.vstack([z_tr, z_va]).max(0) + 0.4
    gx, gy = np.linspace(lo[0], hi[0], n_grilla), np.linspace(lo[1], hi[1], n_grilla)
    malla = np.c_[np.repeat(gx[None, :], n_grilla, 0).ravel(), np.repeat(gy[:, None], n_grilla, 1).ravel()]
    prueba = datos[datos["conjunto"] == "prueba"][[vx, vy, "enfermedad"]].dropna()
    z_te, y_te = (prueba[[vx, vy]].to_numpy(float) - media) / desv, prueba["enfermedad"].to_numpy()

    pasos = []
    if algoritmo == "Regresión logística":
        # Arranca con los pesos al revés de la solución, a propósito, para que se vea cómo los corrige
        guia = LogisticRegression().fit(z_tr, y_tr).coef_[0]
        w, b, tasa_ap = -1.5 * guia / np.linalg.norm(guia), 0.0, 0.15
        etapas = list(range(1, 61))

        def avanzar(_):
            nonlocal w, b
            p = _sigmoide(z_tr @ w + b)
            w = w - tasa_ap * (z_tr.T @ (p - y_tr)) / len(y_tr)
            b = b - tasa_ap * float(np.mean(p - y_tr))
            return lambda m: _sigmoide(m @ w + b)
    elif algoritmo == "Red neuronal":
        red = MLPClassifier(hidden_layer_sizes=(12, 12), learning_rate_init=0.02, random_state=SEMILLA)
        etapas = list(range(1, 61))

        def avanzar(_):
            red.partial_fit(z_tr, y_tr, classes=[0, 1])
            return lambda m: red.predict_proba(m)[:, 1]
    elif algoritmo == "Árbol de decisión":
        etapas = list(range(1, 11))

        def avanzar(k):
            arbol = DecisionTreeClassifier(max_depth=k, random_state=SEMILLA).fit(z_tr, y_tr)
            return lambda m: arbol.predict_proba(m)[:, 1]
    elif algoritmo == "Random Forest":
        bosque = RandomForestClassifier(n_estimators=1, warm_start=True, min_samples_leaf=3, random_state=SEMILLA)
        etapas = list(range(1, 41))

        def avanzar(k):
            bosque.set_params(n_estimators=k)
            bosque.fit(z_tr, y_tr)
            return lambda m: bosque.predict_proba(m)[:, 1]
    elif algoritmo == "KNN":
        etapas = list(range(1, 52, 2))

        def avanzar(k):
            knn = KNeighborsClassifier(n_neighbors=k).fit(z_tr, y_tr)
            return lambda m: knn.predict_proba(m)[:, 1]
    else:  # SVM (RBF)
        etapas = [round(float(c), 3) for c in np.logspace(-2, 2, 21)]

        def avanzar(c):
            svm = SVC(C=c, gamma=1.0, random_state=SEMILLA).fit(z_tr, y_tr)
            return lambda m: _sigmoide(2 * svm.decision_function(m))

    unidad = ALGORITMOS_LAB[algoritmo][0]
    previo = None
    for i, k in enumerate(etapas, start=1):
        f = avanzar(k)
        p_tr, p_va, p_te = f(z_tr), f(z_va), f(z_te)
        ac_tr, ac_va = float(np.mean((p_tr >= 0.5) == y_tr)), float(np.mean((p_va >= 0.5) == y_va))
        perdida = float(log_loss(y_tr, np.clip(p_tr, 0.01, 0.99)))
        mal = (p_tr >= 0.5) != y_tr
        if algoritmo in ("Regresión logística", "Red neuronal"):
            texto = f"Época {k}: pérdida {num(perdida, 3)}"
            if previo is not None:
                cambio = perdida - previo["perdida"]
                texto += (" (cambia muy poco: está cerca de su mejor punto)" if abs(cambio) < 0.0005
                          else f" ({'bajó' if cambio < 0 else 'subió'} {num(abs(cambio), 3)})")
            texto += f"; acierta {pct(ac_tr, 0)} en entrenamiento y {pct(ac_va, 0)} en validación."
            if k == 1:
                texto = ("Época 1: arranca con los pesos al revés, a propósito, y da un primer paso para corregirlos. "
                         f"Acierta {pct(ac_tr, 0)} en entrenamiento." if algoritmo == "Regresión logística" else
                         f"Época 1: las conexiones empiezan al azar. Acierta {pct(ac_tr, 0)} en entrenamiento.")
        elif algoritmo == "Árbol de decisión":
            texto = (f"Con {k} {'pregunta' if k == 1 else 'preguntas seguidas'} acierta {pct(ac_tr, 0)} en entrenamiento "
                     f"y {pct(ac_va, 0)} en validación.")
            if ac_tr - ac_va > 0.12:
                texto += " Ojo: memoriza más de lo que generaliza (sobreajuste)."
        elif algoritmo == "Random Forest":
            texto = (f"Árbol {k}: el bosque vota con {k} {'árbol' if k == 1 else 'árboles'}; acierta {pct(ac_tr, 0)} "
                     f"en entrenamiento y {pct(ac_va, 0)} en validación.")
        elif algoritmo == "KNN":
            texto = f"Consulta {k} {'vecino' if k == 1 else 'vecinos'}: acierta {pct(ac_tr, 0)} en entrenamiento y {pct(ac_va, 0)} en validación."
            if k == 1:
                texto += " Con un solo vecino se aprende de memoria a cada paciente, ruido incluido."
        else:
            texto = f"Rigidez C = {num(k, 2)}: acierta {pct(ac_tr, 0)} en entrenamiento y {pct(ac_va, 0)} en validación."
        paso = {"i": i, "k": k, "grilla": f(malla).reshape(n_grilla, n_grilla).astype(np.float32),
                "ac_tr": ac_tr, "ac_va": ac_va, "ac_te": float(np.mean((p_te >= 0.5) == y_te)),
                "perdida": perdida, "mal": mal, "texto": texto}
        pasos.append(paso)
        previo = paso
    return {"pasos": pasos, "unidad": unidad, "gx": gx * desv[0] + media[0], "gy": gy * desv[1] + media[1],
            "x_tr": x_tr, "y_tr": y_tr, "x_va": x_va, "y_va": y_va, "n_te": len(y_te)}


def figura_frontera(res, paso, vx, vy):
    p = res["pasos"][paso - 1]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=res["gx"], y=res["gy"], z=p["grilla"], zmin=0, zmax=1, zsmooth="best", showscale=False,
                             colorscale=[[0, rgba(AZUL, 0.55)], [0.5, rgba("#9aa4b5", 0.10)], [1, rgba(NARANJA, 0.55)]],
                             hovertemplate=f"{INFO[vx]['etiqueta_corta']}: %{{x:.0f}}<br>{INFO[vy]['etiqueta_corta']}: "
                                           "%{y:.1f}<br>Probabilidad: %{z:.0%}<extra></extra>"))
    fig.add_trace(go.Contour(x=res["gx"], y=res["gy"], z=p["grilla"], showscale=False, hoverinfo="skip",
                             contours=dict(start=0.5, end=0.5, size=0.5, coloring="lines"),
                             line=dict(color=TINTA, width=2.5), colorscale=[[0, TINTA], [1, TINTA]]))
    for dx, val in [("Sin enfermedad", 0), ("Con enfermedad", 1)]:
        m = res["y_tr"] == val
        fig.add_trace(go.Scatter(x=res["x_tr"][m, 0], y=res["x_tr"][m, 1], mode="markers", name=dx,
                                 marker=dict(color=COLOR_DX[dx], size=8, line=dict(color=PANEL, width=1.2)),
                                 hovertemplate="%{x} · %{y}<extra>" + dx + "</extra>"))
    mal = p["mal"]
    fig.add_trace(go.Scatter(x=res["x_tr"][mal, 0], y=res["x_tr"][mal, 1], mode="markers",
                             name=f"Mal clasificados ({int(mal.sum())})", hoverinfo="skip",
                             marker=dict(symbol="circle-open", size=15, color=AMBAR, line=dict(width=2.2))))
    estilo(fig, alto=560, titulo=f"Frontera de decisión · {res['unidad']} {p['k'] if res['unidad'] != 'rigidez' else num(p['k'], 2)}")
    fig.update_layout(legend=dict(y=-0.2, yanchor="top", x=0, xanchor="left"))
    fig.update_xaxes(title=INFO[vx]["etiqueta"] + (f" ({INFO[vx]['unidad']})" if INFO[vx].get("unidad") else ""),
                     range=[res["gx"][0], res["gx"][-1]])
    fig.update_yaxes(title=INFO[vy]["etiqueta"] + (f" ({INFO[vy]['unidad']})" if INFO[vy].get("unidad") else ""),
                     range=[res["gy"][0], res["gy"][-1]], showgrid=False)
    return fig


def figura_curvas(res, paso):
    pasos = res["pasos"][:paso]
    x = [p["i"] for p in pasos]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[100 * p["ac_tr"] for p in pasos], name="Entrenamiento", mode="lines+markers", marker=dict(size=4),
                             line=dict(color=AZUL, width=3, shape="spline"), fill="tozeroy", fillcolor=rgba(AZUL, 0.08)))
    fig.add_trace(go.Scatter(x=x, y=[100 * p["ac_va"] for p in pasos], name="Validación", mode="lines+markers", marker=dict(size=4),
                             line=dict(color=NARANJA, width=3, shape="spline")))
    estilo(fig, alto=260, titulo="Acierto en cada paso")
    fig.update_xaxes(title="Paso", range=[0.5, len(res["pasos"]) + 0.5])
    fig.update_yaxes(title="Acierto", range=[0, 102], ticksuffix=" %")
    return fig


def figura_perdida(res, paso):
    pasos = res["pasos"][:paso]
    fig = go.Figure(go.Scatter(x=[p["i"] for p in pasos], y=[p["perdida"] for p in pasos], mode="lines", name="Pérdida",
                               line=dict(color=AMBAR, width=3, shape="spline"), fill="tozeroy", fillcolor=rgba(AMBAR, 0.10),
                               hovertemplate="Paso %{x}: pérdida %{y:.3f}<extra></extra>"))
    estilo(fig, alto=200, leyenda=False, titulo="Pérdida (qué tanto se equivoca)")
    fig.update_xaxes(range=[0.5, len(res["pasos"]) + 0.5])
    fig.update_yaxes(rangemode="tozero")
    return fig


@st.fragment
def laboratorio():
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    algoritmo = c1.selectbox("Algoritmo", list(ALGORITMOS_LAB), key="lab_alg")
    vx = c2.selectbox("Dato del eje horizontal", VARS_LAB, key="lab_x", format_func=lambda v: INFO[v]["etiqueta_corta"])
    opciones_y = [v for v in VARS_LAB if v != vx]
    vy = c3.selectbox("Dato del eje vertical", opciones_y, key="lab_y", format_func=lambda v: INFO[v]["etiqueta_corta"])
    velocidad = c4.segmented_control("Velocidad", ["Lenta", "Normal", "Rápida"], default="Normal", key="lab_vel")
    bloque(f'<div class="cr-aviso">{ICONO_INFO}<span><b>{algoritmo}.</b> {ALGORITMOS_LAB[algoritmo][1]}</span></div>')

    res = entrenar_en_vivo(algoritmo, vx, vy)
    total = len(res["pasos"])
    config = (algoritmo, vx, vy)
    if st.session_state.get("lab_config") != config:
        st.session_state["lab_config"] = config
        st.session_state["lab_paso"] = 1

    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    entrenar = b1.button("Entrenar", icon=":material/play_arrow:", type="primary", width="stretch", key="lab_entrenar")
    siguiente = b2.button("Un paso", icon=":material/skip_next:", width="stretch", key="lab_paso_btn")
    if b3.button("Reiniciar", icon=":material/restart_alt:", width="stretch", key="lab_reiniciar"):
        st.session_state["lab_paso"] = 1
    if siguiente:
        st.session_state["lab_paso"] = min(total, st.session_state["lab_paso"] + 1)

    zona_canales = st.empty()
    progreso = st.empty()
    m1, m2 = st.columns([1.35, 1], gap="large")
    zona_mapa = m1.empty()
    with m2:
        zona_curvas, zona_perdida, zona_bitacora = st.empty(), st.empty(), st.empty()

    def dibujar(paso):
        p = res["pasos"][paso - 1]
        with zona_canales.container():
            fila_canales([
                canal("Paso", f"{paso} de {total}", f"{res['unidad']} {p['k'] if res['unidad'] != 'rigidez' else num(p['k'], 2)}", "modelo"),
                canal("Acierto entrenamiento", pct(p["ac_tr"], 0), f"{len(res['y_tr'])} pacientes que ve al aprender"),
                canal("Acierto validación", pct(p["ac_va"], 0), f"{len(res['y_va'])} pacientes que no ve", "decision"),
                canal("Pérdida", num(p["perdida"], 3), "más baja es mejor"),
            ])
        progreso.progress(paso / total)
        zona_mapa.plotly_chart(escalar(figura_frontera(res, paso, vx, vy)), theme=None, key=f"lab_mapa_{paso}",
                               config={"displaylogo": False})
        zona_curvas.plotly_chart(escalar(figura_curvas(res, paso)), theme=None, key=f"lab_curvas_{paso}",
                                 config={"displaylogo": False})
        zona_perdida.plotly_chart(escalar(figura_perdida(res, paso)), theme=None, key=f"lab_perdida_{paso}",
                                  config={"displaylogo": False})
        lineas = "".join(f'<li><span class="cr-bit-paso">{q["i"]}</span>{q["texto"]}</li>'
                         for q in reversed(res["pasos"][max(0, paso - 3):paso]))
        with zona_bitacora.container():
            bloque(f'<div class="cr-bitacora"><p class="cr-rotulo">Bitácora</p><ol>{lineas}</ol></div>')

    if entrenar:
        pausa = {"Lenta": 0.35, "Normal": 0.12, "Rápida": 0.02}.get(velocidad or "Normal", 0.12)
        inicio = 1 if st.session_state["lab_paso"] >= total else st.session_state["lab_paso"]
        for paso in range(inicio, total + 1):
            dibujar(paso)
            time.sleep(pausa)
        st.session_state["lab_paso"] = total
    else:
        dibujar(st.session_state["lab_paso"])

    if st.session_state["lab_paso"] == total:
        final = res["pasos"][-1]
        completo = MODELOS[MODELO_ACTIVO]
        ac_completo = float(np.mean((completo["p_prueba"] >= 0.5) == Y_PRUEBA))
        st.success(f"Examen final con los {res['n_te']} pacientes de prueba: {algoritmo}, con solo 2 datos, "
                   f"acierta {pct(final['ac_te'], 0)}. El modelo activo ({MODELO_ACTIVO}), con los 13 datos, "
                   f"acierta {pct(ac_completo, 0)}.", icon=":material/school:")


def escalar(fig):
    return escalar_figura(fig, ESCALA) if ESCALA > 1 else fig


with tab_aprende:
    intro("Aquí un modelo aprende delante de usted. Elija un algoritmo y dos datos clínicos: el modelo intenta separar "
          "a los pacientes sanos (azul) de los enfermos (naranja) usando solo esos dos datos, y cada paso muestra cómo "
          "se mueve su frontera, cuánto acierta y qué hizo, contado en palabras. Con dos datos el modelo es más simple que "
          "el real, que usa trece; sirve para ver cómo aprende.",
          "¿Qué cambia dentro del modelo en cada paso?")
    laboratorio()
    with st.expander("¿Por qué separar entrenamiento y validación?"):
        st.markdown("El modelo aprende con los pacientes de **entrenamiento**. Los de **validación** no los ve: sirven "
                    "para saber si lo aprendido funciona con pacientes nuevos. Cuando el acierto en entrenamiento sigue "
                    "subiendo pero el de validación se estanca o baja, el modelo está memorizando en vez de aprender "
                    "(sobreajuste). Los 61 pacientes de **prueba** se guardan hasta el final, como un examen.")

# ---------------------------------------------------------------------------
# Pestaña 4: resultados de los modelos
# ---------------------------------------------------------------------------
with tab_resultados:
    intro("Probamos siete modelos con el mismo procedimiento: cada uno se entrenó 25 veces con distintas particiones de "
          "los 242 pacientes de entrenamiento (validación cruzada) y después se examinó una sola vez con 61 pacientes "
          "que nunca había visto. Abajo puede comparar los modelos y ver, con números reales, en qué acierta y en qué se "
          "equivoca el modelo activo.",
          "¿Qué tan bien funciona cada modelo, y cuál conviene usar?")

    seccion("¿Qué modelo separa mejor a sanos de enfermos?")
    cv = pd.DataFrame(meta["comparacion_cv"]).sort_values("roc_auc")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        colores = [AZUL if n == NOMBRE_ELEGIDO else (NARANJA if n == MODELO_ACTIVO else GRIS_BARRA) for n in cv["modelo"]]
        fig = go.Figure()
        for fila, color in zip(cv.itertuples(), colores):
            fig.add_shape(type="line", x0=0.5, x1=fila.roc_auc, y0=fila.modelo, y1=fila.modelo,
                          line=dict(color=color, width=4))
        fig.add_trace(go.Scatter(x=cv["roc_auc"], y=cv["modelo"], mode="markers+text", text=[num(v, 3) for v in cv["roc_auc"]],
                                 textposition="middle right", textfont=dict(color=TINTA_2, size=13),
                                 marker=dict(size=18, color=colores, line=dict(color=PANEL, width=2)),
                                 customdata=cv["roc_auc_sd"],
                                 hovertemplate="%{y}<br>AUC promedio: %{x:.3f} ± %{customdata:.3f}<extra></extra>"))
        estilo(fig, alto=380, leyenda=False, titulo="AUC en validación cruzada, 7 modelos")
        fig.update_xaxes(title="AUC (1 = perfecto, 0,5 = azar)", range=[0.5, 1.04], showgrid=True, gridcolor=REJILLA)
        fig.update_yaxes(showgrid=False)
        mostrar(fig)
        st.caption("Azul: el modelo que elegimos. Naranja: el modelo activo, si es otro.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS_BARRA, width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
        for nombre in NOMBRES_MODELOS:
            fpr, tpr, _ = roc_curve(Y_PRUEBA, MODELOS[nombre]["p_prueba"])
            activo = nombre == MODELO_ACTIVO
            elegido = nombre == NOMBRE_ELEGIDO
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nombre} ({num(MODELOS[nombre]['auc_prueba'], 3)})",
                                     line=dict(color=AZUL if elegido else (NARANJA if activo else GRIS_BARRA),
                                               width=3.5 if (activo or elegido) else 1.8, shape="hv"),
                                     opacity=1 if activo or nombre == NOMBRE_ELEGIDO else 0.8))
        estilo(fig, alto=380, titulo="Curvas ROC en los 61 pacientes de prueba")
        fig.update_layout(legend=dict(orientation="v", x=0.98, xanchor="right", y=0.04, yanchor="bottom",
                                      bgcolor=rgba(PANEL_2, 0.9), bordercolor=BORDE, borderwidth=1, font=dict(size=11)))
        fig.update_xaxes(title="Falsas alarmas (1 - especificidad)", range=[-0.02, 1.02])
        fig.update_yaxes(title="Enfermos detectados (sensibilidad)", range=[-0.02, 1.05])
        mostrar(fig)

    criterio_res = st.segmented_control("Criterio para comparar", ["tamizaje", "balanceado"], default="tamizaje",
                                        format_func=lambda c: NOMBRE_CRITERIO[c], key="criterio_res") or "tamizaje"
    filas = []
    for nombre in NOMBRES_MODELOS:
        u = MODELOS[nombre]["umbrales"][criterio_res]
        pred = MODELOS[nombre]["p_prueba"] >= u
        filas.append({"modelo": nombre, "Sensibilidad": (pred & (Y_PRUEBA == 1)).sum() / (Y_PRUEBA == 1).sum(),
                      "Especificidad": (~pred & (Y_PRUEBA == 0)).sum() / (Y_PRUEBA == 0).sum(),
                      "Exactitud": (pred == Y_PRUEBA).mean()})
    res_df = pd.DataFrame(filas)
    fig = go.Figure()
    for metrica, color in [("Sensibilidad", NARANJA), ("Especificidad", AZUL), ("Exactitud", GRIS_BARRA)]:
        fig.add_trace(go.Bar(x=res_df["modelo"], y=100 * res_df[metrica], name=metrica,
                             marker=dict(color=color, cornerradius=5),
                             text=[pct(v, 0) for v in res_df[metrica]], textposition="outside", textfont=dict(size=11, color=TINTA_2),
                             cliponaxis=False, hovertemplate="%{x}<br>" + metrica + ": %{y:.1f} %<extra></extra>"))
    estilo(fig, alto=380, titulo=f"En prueba, con el criterio {NOMBRE_CRITERIO[criterio_res].lower()} de cada modelo")
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 112], ticksuffix=" %", title="")
    mostrar(fig)
    st.markdown(f"La **regresión logística** quedó primera en validación cruzada (AUC {num(cv['roc_auc'].max(), 3)} con los "
                f"hiperparámetros por defecto y {num(MODELOS[NOMBRE_ELEGIDO]['auc_cv'][0], 3)} ya ajustada) y en prueba no la "
                "supera ningún modelo más complejo. Con 303 pacientes, un modelo simple y "
                "explicable es la mejor opción; por eso es el modelo por defecto.")

    seccion(f"Los 4 resultados posibles · {MODELO_ACTIVO}")
    u_def = float(M["umbrales"]["tamizaje"])
    umbral_sel = st.slider("Umbral de decisión", 0.05, 0.95, u_def, 0.01, key=f"umbral_{MODELO_ACTIVO}",
                           help="Desde qué probabilidad el modelo marca riesgo alto. Empieza en el umbral de tamizaje del modelo activo.")
    pred = M["p_prueba"] >= umbral_sel
    vp = int((pred & (Y_PRUEBA == 1)).sum()); vn = int((~pred & (Y_PRUEBA == 0)).sum())
    fp = int((pred & (Y_PRUEBA == 0)).sum()); fn = int((~pred & (Y_PRUEBA == 1)).sum())
    bloque(f"""
<div class="cr-resultados">
  <div class="cr-res cr-res-vp"><span class="cr-res-num">{vp}</span><span class="cr-res-tit">Enfermos detectados</span>
    <span class="cr-res-txt">Verdaderos positivos. El modelo dijo riesgo alto y la angiografía lo confirmó.</span></div>
  <div class="cr-res cr-res-vn"><span class="cr-res-num">{vn}</span><span class="cr-res-tit">Sanos descartados</span>
    <span class="cr-res-txt">Verdaderos negativos. Dijo riesgo bajo y la angiografía salió normal.</span></div>
  <div class="cr-res cr-res-fp"><span class="cr-res-num">{fp}</span><span class="cr-res-tit">Falsas alarmas</span>
    <span class="cr-res-txt">Falsos positivos. Dijo riesgo alto, pero el paciente estaba sano: un examen de más.</span></div>
  <div class="cr-res cr-res-fn"><span class="cr-res-num">{fn}</span><span class="cr-res-tit">Enfermos que se escapan</span>
    <span class="cr-res-txt">Falsos negativos. Dijo riesgo bajo y el paciente sí estaba enfermo: el error más grave.</span></div>
</div>""")
    sens = vp / (vp + fn) if vp + fn else 0
    espec = vn / (vn + fp) if vn + fp else 0
    prec = vp / (vp + fp) if vp + fp else 0
    exac = (vp + vn) / len(Y_PRUEBA)
    bloque(f"""
<div class="cr-metricas">
  <div class="cr-metrica"><p class="cr-metrica-tit">Sensibilidad · {pct(sens, 0)}</p>
    <p>De los enfermos, ¿a cuántos detectó? <span class="cr-formula">{vp} / ({vp} + {fn})</span></p>
    <p class="cr-metrica-nota">La más importante en un tamizaje.</p></div>
  <div class="cr-metrica"><p class="cr-metrica-tit">Especificidad · {pct(espec, 0)}</p>
    <p>De los sanos, ¿a cuántos descartó? <span class="cr-formula">{vn} / ({vn} + {fp})</span></p>
    <p class="cr-metrica-nota">Mide cuántos exámenes de más se evitan.</p></div>
  <div class="cr-metrica"><p class="cr-metrica-tit">Precisión · {pct(prec, 0)}</p>
    <p>De las alertas, ¿cuántas eran enfermos? <span class="cr-formula">{vp} / ({vp} + {fp})</span></p>
    <p class="cr-metrica-nota">Si es baja, el modelo grita mucho.</p></div>
  <div class="cr-metrica"><p class="cr-metrica-tit">Exactitud · {pct(exac, 0)}</p>
    <p>¿Cuántos acertó en total? <span class="cr-formula">({vp} + {vn}) / {len(Y_PRUEBA)}</span></p>
    <p class="cr-metrica-nota">Engaña si casi todos los pacientes son de una sola clase.</p></div>
</div>
<div class="cr-aviso">{ICONO_INFO}<span><b>Una red de pesca.</b> La sensibilidad es la parte de los peces del lago que
quedó en la red. La precisión es la parte de lo que sacó la red que de verdad eran peces y no botas viejas. Si abre más
la malla (baja el umbral) atrapa más peces, pero también más botas.</span></div>""")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        curva = M["curva"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curva["umbral"], y=curva["sensibilidad"], name="Sensibilidad",
                                 line=dict(color=NARANJA, width=3, shape="spline")))
        fig.add_trace(go.Scatter(x=curva["umbral"], y=curva["especificidad"], name="Especificidad",
                                 line=dict(color=AZUL, width=3, shape="spline")))
        for c in ("tamizaje", "balanceado"):
            fig.add_vline(x=M["umbrales"][c], line_color=AMBAR, line_width=1.5, line_dash="dot")
            fig.add_annotation(x=M["umbrales"][c], y=0.04, xanchor="left", xshift=5, showarrow=False,
                               text=f"{NOMBRE_CRITERIO[c]} {num(M['umbrales'][c], 2)}", font=dict(size=11, color=TINTA_2))
        fig.add_vline(x=umbral_sel, line_color=TINTA, line_width=1)
        estilo(fig, alto=360, titulo="Lo que se gana y se pierde al mover el umbral")
        fig.update_xaxes(title="Umbral de decisión (validación cruzada)", range=[0.05, 0.95])
        fig.update_yaxes(title="Proporción", range=[0, 1.05])
        mostrar(fig)
    with c4:
        imp = M["importancia"].head(8)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[INFO[v]["etiqueta_corta"] for v in imp.index], orientation="h",
                               marker=dict(color=[NARANJA if i == imp.index[-1] else AZUL for i in imp.index], cornerradius=5),
                               width=0.6, hovertemplate="%{y}: el AUC baja %{x:.3f} si se desordena<extra></extra>"))
        estilo(fig, alto=360, leyenda=False, titulo="Qué datos mira más este modelo")
        fig.update_xaxes(title="Pérdida de AUC al desordenar el dato", showgrid=True, gridcolor=REJILLA)
        fig.update_yaxes(showgrid=False)
        mostrar(fig)

    with st.expander("Detalle técnico: tabla de validación cruzada, odds ratios y curva de aprendizaje"):
        tabla = pd.DataFrame(meta["comparacion_cv"]).rename(columns={
            "modelo": "Modelo", "roc_auc": "AUC", "roc_auc_sd": "AUC (desv.)", "accuracy": "Exactitud",
            "recall": "Sensibilidad", "precision": "Precisión", "f1": "F1"})
        st.dataframe(tabla.style.format({c: "{:.3f}" for c in tabla.columns if c != "Modelo"}, decimal=","),
                     hide_index=True, width="stretch")
        st.markdown(f"Con solo {len(Y_PRUEBA)} pacientes de prueba la cifra varía bastante: el intervalo de confianza del 95 % "
                    f"del AUC de la regresión logística va de {num(PRUEBA['ic95']['auc'][0], 3)} a {num(PRUEBA['ic95']['auc'][1], 3)}.")
        t1, t2 = st.columns(2, gap="large")
        with t1:
            ors = pd.DataFrame(meta["odds_ratios"]).head(10)[::-1]
            marcas_or = [0.5, 0.67, 1, 1.5, 2, 3]
            fig = go.Figure(go.Bar(
                x=ors["log_or"], y=ors["etiqueta_corta"], orientation="h", width=0.6,
                marker=dict(color=[NARANJA if v > 0 else AZUL for v in ors["log_or"]], cornerradius=4),
                customdata=np.stack([ors["odds_ratio"], ors["comparacion"]], axis=-1),
                hovertemplate="%{customdata[1]}<br>Odds ratio: %{customdata[0]:.2f}<extra></extra>"))
            fig.add_vline(x=0, line_color=EJE, line_width=1)
            estilo(fig, alto=420, leyenda=False, titulo="Odds ratios de la regresión logística")
            fig.update_xaxes(title="Odds ratio (escala logarítmica)", tickvals=list(np.log(marcas_or)),
                             ticktext=[num(v, 2).rstrip("0").rstrip(",") for v in marcas_or],
                             range=[np.log(0.45), np.log(3.8)], showgrid=True, gridcolor=REJILLA)
            fig.update_yaxes(showgrid=False)
            mostrar(fig)
        with t2:
            ca = meta["curva_aprendizaje"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ca["pacientes"], y=ca["auc_entrenamiento"], name="Entrenamiento",
                                     line=dict(color=AZUL, width=3, shape="spline"), mode="lines+markers"))
            fig.add_trace(go.Scatter(x=ca["pacientes"], y=ca["auc_validacion"], name="Validación",
                                     line=dict(color=NARANJA, width=3, shape="spline"), mode="lines+markers"))
            estilo(fig, alto=420, titulo="Curva de aprendizaje de la regresión logística")
            fig.update_xaxes(title="Pacientes de entrenamiento")
            fig.update_yaxes(title="AUC")
            mostrar(fig)

# ---------------------------------------------------------------------------
# Pestaña 5: evaluar un paciente
# ---------------------------------------------------------------------------
EJEMPLOS = meta["ejemplos"]


def cargar_ejemplo(nombre):
    for var, valor in EJEMPLOS[nombre]["valores"].items():
        st.session_state[f"in_{var}"] = valor


if "in_age" not in st.session_state:
    cargar_ejemplo("bajo")


def entrada(var, contenedor):
    info = INFO[var]
    clave = f"in_{var}"
    ayuda = QUE_ES[var][1]
    etiqueta = info["etiqueta"] + (f" ({info['unidad']})" if info.get("unidad") else "")
    if info["tipo"] in ("binaria", "categorica"):
        codigos = [int(c) for c in info["opciones"].keys()]
        return contenedor.selectbox(etiqueta, codigos, key=clave, help=ayuda,
                                    format_func=lambda c, i=info: i["opciones"][str(c)])
    minimo, maximo = info["rango_entrada"]
    if var == "oldpeak":
        return contenedor.number_input(etiqueta, min_value=float(minimo), max_value=float(maximo),
                                       step=0.1, key=clave, help=ayuda, format="%.1f")
    return contenedor.number_input(etiqueta, min_value=int(minimo), max_value=int(maximo),
                                   step=1, key=clave, help=ayuda)


with tab_pac:
    intro(f"Ingrese los datos de un paciente o cargue uno de los ejemplos. El modelo activo ({MODELO_ACTIVO}) "
          "calcula la probabilidad y explica qué datos la subieron o la bajaron.")
    col_form, col_res = st.columns([1.12, 1], gap="large")

    with col_form:
        st.subheader("Datos del paciente")
        b1, b2 = st.columns(2)
        b1.button("Cargar ejemplo de riesgo bajo", on_click=cargar_ejemplo, args=("bajo",), width="stretch",
                  key="ejemplo_bajo")
        b2.button("Cargar ejemplo de riesgo alto", on_click=cargar_ejemplo, args=("alto",), width="stretch",
                  key="ejemplo_alto")
        st.caption("Los ejemplos son pacientes reales del conjunto de prueba; el modelo no los vio al entrenar.")

        with st.container(border=True):
            rotulo("Datos generales y síntomas")
            c1, c2 = st.columns(2)
            entrada("age", c1)
            entrada("sex", c2)
            entrada("cp", c1)
            entrada("exang", c2)

        with st.container(border=True):
            rotulo("Signos vitales y laboratorio")
            c1, c2, c3 = st.columns(3)
            entrada("trestbps", c1)
            entrada("chol", c2)
            entrada("fbs", c3)

        with st.container(border=True):
            rotulo("Pruebas diagnósticas")
            c1, c2 = st.columns(2)
            entrada("restecg", c1)
            entrada("thalach", c2)
            entrada("oldpeak", c1)
            entrada("slope", c2)
            entrada("ca", c1)
            entrada("thal", c2)

    paciente = pd.DataFrame([{v: st.session_state[f"in_{v}"] for v in VARS}])
    prob = float(predecir(paciente)[0])

    with col_res:
        st.subheader("Resultado")
        criterio = st.radio(
            "Criterio de decisión",
            ["tamizaje", "balanceado"],
            format_func=lambda c: f"{NOMBRE_CRITERIO[c]} (umbral {pct(M['umbrales'][c], 0)})",
            horizontal=True,
            help="Tamizaje prioriza no dejar pasar enfermos (más sensibilidad). "
                 "Estándar trata igual los dos errores y genera menos falsas alarmas (más especificidad).",
        )
        umbral = M["umbrales"][criterio]
        bloque(lectura(prob, umbral, criterio))
        bloque(
            f'<p class="cr-nota">Con este criterio, en validación cruzada el modelo {MODELO_ACTIVO} detectó al '
            f"<b>{pct(M['sens_cv'][criterio], 0)}</b> de los pacientes enfermos y descartó correctamente "
            f"al <b>{pct(M['esp_cv'][criterio], 0)}</b> de los sanos.</p>"
        )
        otros = {n: float(predecir(paciente, n)[0]) for n in NOMBRES_MODELOS}
        filas_html = "".join(
            f'<div class="cr-comp"><span class="cr-comp-et">{n}</span><span class="cr-comp-pista"><span class="cr-comp-barra'
            f'{" cr-alta" if p >= MODELOS[n]["umbrales"][criterio] else ""}" style="width:{100 * p:.0f}%"></span></span>'
            f'<span class="cr-comp-val">{pct(p, 0)}</span></div>' for n, p in otros.items())
        bloque(f'<div class="cr-pista"><p class="cr-pista-tit">Qué dicen los cinco modelos</p>{filas_html}'
               f'<p class="cr-pista-nota">Naranja: ese modelo marca riesgo alto con el criterio {NOMBRE_CRITERIO[criterio].lower()}.</p></div>')

    st.write("")
    c_graf, c_guia = st.columns([1.45, 1], gap="large")
    with c_graf:
        serie = aportes(paciente)
        top = serie.reindex(serie.abs().sort_values(ascending=False).index).head(8)[::-1]
        etiquetas = [f"{INFO[v]['etiqueta_corta']}: {etiqueta_valor(v, paciente.iloc[0][v])}" for v in top.index]
        fig = go.Figure(go.Bar(
            x=100 * top.values, y=etiquetas, orientation="h", width=0.6,
            marker=dict(color=[NARANJA if v > 0 else AZUL for v in top.values], cornerradius=5),
            text=[("≈0" if abs(100 * v) < 0.05 else f"{'+' if v > 0 else '−'}{num(abs(100 * v), 1)}") for v in top.values],
            textposition="outside",
            textfont=dict(size=12, color=TINTA_2), cliponaxis=False,
            hovertemplate="%{y}<br>Cambia el riesgo en %{x:.1f} puntos<extra></extra>"))
        fig.add_vline(x=0, line_color=EJE, line_width=1)
        estilo(fig, alto=380, leyenda=False, titulo="Qué datos subieron o bajaron el riesgo de este paciente")
        tope = max(2.0, 100 * float(top.abs().max())) * 1.3
        fig.update_xaxes(title="Puntos de riesgo", showgrid=True, gridcolor=REJILLA,
                         zeroline=False, range=[-tope, tope])
        fig.update_yaxes(showgrid=False)
        mostrar(fig)
        st.caption("Cada barra compara al paciente con un paciente sano típico: cuántos puntos cambiaría la probabilidad "
                   "si ese dato fuera el de alguien sin la enfermedad.")
    with c_guia:
        with st.container(border=True):
            rotulo("Cómo leer este resultado")
            st.markdown(
                "- La **probabilidad** va de 0 a 100 %: qué tan probable es que la angiografía muestre una arteria estrechada.\n"
                "- El **umbral** decide desde qué probabilidad se marca riesgo alto. En tamizaje lo bajamos para no dejar pasar enfermos, a costa de más falsas alarmas.\n"
                "- Las barras **naranjas** suben el riesgo y las **azules** lo bajan, comparadas con un paciente sano típico.\n"
                "- Puede cambiar de modelo en la barra lateral y ver si todos coinciden."
            )
        aviso("Este resultado es una estimación estadística hecha con datos de un solo hospital (Cleveland Clinic), "
              "publicados en 1988. Sirve como apoyo académico y no reemplaza el criterio médico ni la angiografía.")

# ---------------------------------------------------------------------------
# Pestaña 6: varios pacientes con un CSV
# ---------------------------------------------------------------------------
with tab_lote:
    intro(f"Para evaluar a varios pacientes a la vez, suba un archivo CSV con una fila por paciente y las 13 columnas del "
          f"dataset. El modelo activo ({MODELO_ACTIVO}) los clasifica con su umbral de tamizaje. La plantilla trae seis "
          "pacientes reales para probar.")
    plantilla = pd.read_csv(RUTA_PLANTILLA)
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        c1.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                           "plantilla_pacientes.csv", "text/csv", width="stretch", icon=":material/download:")
        usar_plantilla = c1.toggle("Probar con la plantilla de ejemplo", key="usar_plantilla")
        archivo = c2.file_uploader("Subir archivo CSV", type=["csv"])

    with st.expander("Códigos de cada columna"):
        filas = []
        for v in VARS:
            info = INFO[v]
            if info["tipo"] in ("binaria", "categorica"):
                regla = "; ".join(f"{k} = {t}" for k, t in info["opciones"].items())
            else:
                dec = 1 if v == "oldpeak" else 0
                regla = (f"número entre {num(info['rango_entrada'][0], dec)} y {num(info['rango_entrada'][1], dec)} "
                         f"{info.get('unidad', '')}").strip()
            filas.append({"Columna": v, "Variable": info["etiqueta"], "Valores permitidos": regla})
        st.dataframe(pd.DataFrame(filas), hide_index=True)

    lote = None
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}. Revise que sea un CSV separado por comas.")
    elif usar_plantilla:
        lote = plantilla.copy()

    if lote is not None:
        faltantes = [v for v in VARS if v not in lote.columns]
        if faltantes:
            st.error("Faltan columnas: " + ", ".join(faltantes) + ". Use la plantilla como guía.")
        else:
            errores = []
            for v in VARS:
                info = INFO[v]
                valores = pd.to_numeric(lote[v], errors="coerce")
                if info["tipo"] in ("binaria", "categorica"):
                    validos = [float(k) for k in info["opciones"].keys()]
                    malos = ~valores.isin(validos) & valores.notna()
                else:
                    lo, hi = info["rango_entrada"]
                    malos = (valores < lo) | (valores > hi)
                if v not in ("ca", "thal"):
                    malos = malos | valores.isna()
                for i in lote.index[malos]:
                    errores.append(f"Fila {i + 1}: valor no válido en '{v}' ({lote.at[i, v]})")
                lote[v] = valores
            if errores:
                st.error("Corrija estos valores antes de evaluar:\n\n- " + "\n- ".join(errores[:15]))
            else:
                prob_lote = predecir(lote)
                umbral_lote = M["umbrales"]["tamizaje"]
                salida = lote.copy()
                salida["probabilidad (%)"] = np.round(100 * prob_lote, 1)
                salida["clasificación"] = np.where(prob_lote >= umbral_lote, "Riesgo alto", "Riesgo bajo")
                n_alto = int((prob_lote >= umbral_lote).sum())
                st.success(f"{len(salida)} pacientes evaluados con el criterio de tamizaje "
                           f"(umbral {pct(umbral_lote, 0)}): {n_alto} con riesgo alto.")
                orden = np.argsort(prob_lote)
                fig = go.Figure(go.Bar(
                    x=100 * prob_lote[orden], y=[f"Paciente {i + 1}" for i in orden], orientation="h", width=0.6,
                    marker=dict(color=[NARANJA if p >= umbral_lote else AZUL for p in prob_lote[orden]], cornerradius=5),
                    text=[pct(p, 0) for p in prob_lote[orden]], textposition="outside", cliponaxis=False,
                    textfont=dict(color=TINTA_2), hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
                fig.add_vline(x=100 * umbral_lote, line_color=AMBAR, line_width=2, line_dash="dot")
                estilo(fig, alto=max(260, 40 * len(salida) + 120), leyenda=False, titulo="Probabilidad de cada paciente")
                fig.update_xaxes(range=[0, 110], ticksuffix=" %", showgrid=True, gridcolor=REJILLA)
                fig.update_yaxes(showgrid=False)
                mostrar(fig)
                st.dataframe(salida, hide_index=True)
                st.download_button("Descargar resultados (CSV)", salida.to_csv(index=False).encode("utf-8"),
                                   "resultados_cardioriesgo.csv", "text/csv", icon=":material/download:")

# ---------------------------------------------------------------------------
# Pestaña 7: el proyecto
# ---------------------------------------------------------------------------
with tab_acerca:
    ds = meta["dataset"]
    st.subheader("El proyecto")
    bloque(f"""
<div class="cr-fichas">
  <div class="cr-ficha"><p class="cr-ficha-titulo">La pregunta</p><p>{PROYECTO['pregunta']}</p></div>
  <div class="cr-ficha"><p class="cr-ficha-titulo">El objetivo</p><p>{PROYECTO['objetivo']}</p></div>
  <div class="cr-ficha"><p class="cr-ficha-titulo">La respuesta</p><p>Una regresión logística con 13 datos clínicos
    alcanza un AUC de {num(meta['modelo']['auc_cv'], 3)} en validación cruzada y, con el umbral de tamizaje, detectó a
    <b>{int(PRUEBA['metricas']['tamizaje']['VP'])} de {int(PRUEBA['metricas']['tamizaje']['VP'] + PRUEBA['metricas']['tamizaje']['FN'])}</b>
    enfermos de prueba. Sirve como apoyo para decidir a quién estudiar primero, no como diagnóstico.</p></div>
</div>""")

    st.subheader("Ruta metodológica")
    bloque("""
<ol class="cr-pasos">
  <li><strong>Exploración</strong><span>Distribuciones, relación de cada variable con el diagnóstico y correlaciones.</span></li>
  <li><strong>Calidad y limpieza</strong><span>6 valores faltantes (4 en vasos coloreados y 2 en talio), imputados dentro del pipeline.</span></li>
  <li><strong>Modelado</strong><span>7 modelos clásicos con validación cruzada 5×5 y búsqueda de hiperparámetros.</span></li>
  <li><strong>Evaluación</strong><span>61 pacientes de prueba, intervalos bootstrap y elección del umbral de tamizaje.</span></li>
  <li><strong>Despliegue</strong><span>Esta aplicación, publicada en Streamlit Community Cloud.</span></li>
</ol>""")

    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.subheader("Referencias")
        for r in sorted(PROYECTO["referencias"] + [REF_DANE]):
            st.markdown(f"- {r}")
    with c2:
        st.subheader("Grupo 13")
        st.markdown("\n".join(f"- {n}" for n in PROYECTO["integrantes"]))
        st.markdown(f"Especialización en Inteligencia Artificial, CUN  \nAprendizaje Automático, {PROYECTO['docente']}")
        enlaces = PROYECTO.get("enlaces", {})
        st.subheader("Enlaces")
        for nombre, url in [("Aplicación", enlaces.get("app")), ("Video de sustentación", enlaces.get("video")),
                            ("Repositorio", enlaces.get("repositorio"))]:
            if url:
                st.markdown(f"- {nombre}: {url}")
        aviso("Proyecto académico. El modelo no está validado para uso clínico.")

# ---------------------------------------------------------------------------
# Puente con el tema de Streamlit. Streamlit guarda el tema activo en el navegador;
# este script lo cambia con el selector del menú principal (oculto por CSS), sin
# recargar la página, y deja guardado el tema claro para la próxima visita. Si el
# menú no está disponible (por ejemplo, con ?embed=true), guarda el tema y recarga.
# ---------------------------------------------------------------------------
PUENTE_TEMA = """
<span hidden></span>
<script>
(async () => {
  const quiero = "__MODO__";
  const clave = "stActiveTheme-" + window.location.pathname + "-v2";
  const espera = (ms) => new Promise((r) => setTimeout(r, ms));
  const actual = () => {
    const app = document.querySelector('[data-testid="stApp"]');
    return app && getComputedStyle(app).colorScheme.includes("dark") ? "oscuro" : "claro";
  };
  const recordarClaro = () => { try { localStorage.setItem(clave, JSON.stringify("Light")); } catch (e) {} };
  if (actual() === quiero) { recordarClaro(); sessionStorage.removeItem("crTemaRecarga"); return; }
  const destino = quiero === "oscuro" ? "Dark" : "Light";
  const velo = document.createElement("style");
  velo.textContent = '[data-testid="stMainMenuPopover"]{opacity:0 !important;pointer-events:none !important}';
  document.head.appendChild(velo);
  let listo = false;
  try {
    const boton = document.querySelector('[data-testid="stMainMenuButton"]');
    if (boton) {
      boton.click();
      let item = null;
      for (let i = 0; i < 40 && !item; i++) {
        await espera(25);
        item = document.querySelector('[data-testid="stMainMenuItem-theme-' + destino + '"]');
      }
      if (item) { item.click(); listo = true; }
      await espera(80);
      if (document.querySelector('[data-testid="stMainMenuPopover"]')) { boton.click(); }
    }
  } catch (e) { listo = false; }
  await espera(120);
  velo.remove();
  if (!listo || actual() !== quiero) {
    if (sessionStorage.getItem("crTemaRecarga")) { return; }  // una sola recarga, nunca en bucle
    try { localStorage.setItem(clave, JSON.stringify(destino)); sessionStorage.setItem("crTemaRecarga", "1"); } catch (e) { return; }
    window.location.reload();
    return;
  }
  sessionStorage.removeItem("crTemaRecarga");
  recordarClaro();
})();
</script>
"""
st.html(PUENTE_TEMA.replace("__MODO__", "oscuro" if MODO_OSCURO else "claro"), unsafe_allow_javascript=True)

# ---------------------------------------------------------------------------
# Escala de pantalla. El CSS agranda la base tipográfica en monitores grandes y
# televisores, pero Plotly dibuja en píxeles. El navegador mide la escala (base / 16 px)
# y la escribe en el campo oculto "escala_pantalla"; Python la usa en mostrar() para
# agrandar letras y alto de los gráficos. En pantallas normales la escala es 1 y no hace nada.
# ---------------------------------------------------------------------------
ESCALA_PANTALLA = r"""
<span hidden></span>
<script>
(() => {
  const informar = () => {
    const s = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) / 16 * 20) / 20;
    const campo = document.querySelector('.st-key-escala_pantalla input');
    if (!campo) return;
    const actual = parseFloat(campo.value || "1");
    if (Math.abs(actual - s) < 0.01) return;
    const fijar = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    fijar.call(campo, s.toFixed(2));
    campo.dispatchEvent(new Event("input", {bubbles: true}));
    campo.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", code: "Enter", keyCode: 13, bubbles: true}));
  };
  if (!window.__crEscala) {
    window.__crEscala = true;
    let espera;
    window.addEventListener("resize", () => { clearTimeout(espera); espera = setTimeout(informar, 400); });
  }
  // Varios intentos: si la app todavía está terminando de ejecutarse, el primer aviso se puede perder
  [50, 700, 1800, 4000].forEach((t) => setTimeout(informar, t));
})();
</script>
"""
st.html(ESCALA_PANTALLA, unsafe_allow_javascript=True)
