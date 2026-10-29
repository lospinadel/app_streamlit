# -*- coding: utf-8 -*-
"""
CardioRiesgo - ACA 1 Aprendizaje Automático (CUN), Grupo 13.
Aplicación Streamlit que usa el modelo de regresión logística entrenado en el
notebook del proyecto con el dataset UCI Heart Disease (Cleveland).

Diseño: tema oscuro inspirado en un monitor de cabecera. Cada color significa
algo: blanco para los datos, cian para el modelo, verde para riesgo bajo, rojo
para riesgo alto y ámbar para el umbral de decisión. Los gráficos usan azul
(sin enfermedad) y naranja (con enfermedad). Las tipografías (Atkinson
Hyperlegible Next para el texto y Saira para títulos y cifras) se cargan desde
.streamlit/config.toml.

Ejecución local:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo" / "modelo_cardio.joblib"
RUTA_META = BASE / "modelo" / "metadata.json"
RUTA_DATOS = BASE / "data" / "heart_clean.csv"
RUTA_PLANTILLA = BASE / "data" / "plantilla_pacientes.csv"

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
.block-container, [data-testid="stMainBlockContainer"] {max-width: 1240px; padding-top: 4.2rem; padding-bottom: 3rem;}
[data-testid="stHeading"] h3 {font-weight: 600; font-size: 1.45rem;}

/* Encabezado: marca, trazo de ECG y canales del monitor */
section.cr-hero {position: relative; padding: 22px 28px 20px; border: 1px solid var(--cr-borde);
  border-radius: 18px; background: var(--cr-hero); overflow: hidden; margin: 0;}
section.cr-hero::before {content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(var(--cr-cuadricula-fina) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula-fina) 1px, transparent 1px);
  background-size: 30px 30px, 30px 30px, 6px 6px, 6px 6px;}
section.cr-hero > * {position: relative;}
section.cr-hero ul.cr-chips {display: flex; flex-wrap: wrap; gap: 8px; list-style: none; margin: 0 0 12px; padding: 0;}
section.cr-hero ul.cr-chips li {margin: 0; padding: 5px 11px; font-size: .84rem; line-height: 1.2; color: var(--cr-tinta-2);
  border: 1px solid var(--cr-borde-2); border-radius: 999px; background: var(--cr-panel-2);}
section.cr-hero ul.cr-chips li::marker {content: none;}
section.cr-hero div.cr-marca {font-family: var(--cr-cifras); font-weight: 700; font-stretch: 125%;
  font-variation-settings: "wdth" 125; font-size: clamp(2.5rem, 3.2vw + 1.2rem, 4rem); line-height: 1;
  letter-spacing: -0.01em; color: var(--cr-tinta); margin: 0; padding: 0;}
section.cr-hero svg.cr-ecg {display: block; width: 100%; height: auto; margin: 6px 0 4px; overflow: visible;}
section.cr-hero .cr-traza {fill: none; stroke: var(--cr-traza); stroke-width: 2.6; stroke-linecap: round;
  stroke-linejoin: round; filter: drop-shadow(0 0 5px var(--cr-brillo-traza));}
section.cr-hero .cr-punto {fill: var(--cr-traza);}
section.cr-hero .cr-onda {fill: none; stroke: var(--cr-traza); stroke-width: 1.6; transform-box: fill-box;
  transform-origin: center; animation: cr-latido 1.1s var(--cr-ease-out) infinite;}
section.cr-hero p.cr-lema {color: var(--cr-tinta-2); font-size: 1.04rem; line-height: 1.55; max-width: 68ch; margin: 0;}

div.cr-fila {container-type: inline-size;}
div.cr-canales {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 14px 0 10px;}
div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: 1 / -1;}
@container (min-width: 620px) {
  div.cr-canales.cr-n4 {grid-template-columns: repeat(4, minmax(0, 1fr));}
  div.cr-canales.cr-n6 {grid-template-columns: repeat(3, minmax(0, 1fr));}
}
@container (min-width: 760px) {
  div.cr-canales.cr-n5 {grid-template-columns: repeat(5, minmax(0, 1fr));}
  div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: auto;}
}
@container (min-width: 960px) {
  div.cr-canales.cr-n6 {grid-template-columns: repeat(6, minmax(0, 1fr));}
  div.cr-canales.cr-n6 .cr-canal-valor {font-size: 1.8rem;}
}
div.cr-canales .cr-canal {position: relative; display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 4px;
  align-content: start; padding: 13px 15px 12px; background: var(--cr-panel); border: 1px solid var(--cr-borde);
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
div.cr-lectura {--estado: var(--cr-verde); position: relative; padding: 16px 20px 14px; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); border-radius: 16px; overflow: hidden;}
div.cr-lectura.cr-alto {--estado: var(--cr-rojo);}
div.cr-lectura::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--estado);}
div.cr-lectura .cr-lectura-cab {display: flex; justify-content: space-between; align-items: center; gap: 8px 14px;
  flex-wrap: wrap; font-size: .92rem; color: var(--cr-tinta-2);}
div.cr-lectura .cr-criterio {font-size: .82rem; color: var(--cr-tinta); padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--cr-borde-2); background: var(--cr-panel-2); white-space: nowrap;}
div.cr-lectura .cr-lectura-valor {font-family: var(--cr-cifras); font-weight: 600; font-size: clamp(4rem, 5vw + 1.4rem, 5.6rem);
  line-height: 1; margin: 14px 0 12px; color: var(--estado); font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
  text-shadow: 0 0 28px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--estado) 40%, transparent));}
div.cr-lectura .cr-unidad {font-size: .4em; font-weight: 500; margin-left: .14em; color: var(--cr-tinta-2); text-shadow: none;}
div.cr-lectura .cr-estado {display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 10px;
  font-size: .98rem; line-height: 1.4; color: var(--cr-tinta); background: color-mix(in srgb, var(--estado) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--estado) 38%, transparent);}
div.cr-lectura .cr-estado svg {flex: none; color: var(--estado);}
div.cr-lectura .cr-barra {position: relative; display: flex; gap: 3px; margin: 20px 0 6px;}
div.cr-lectura .cr-seg {flex: 1; height: 14px; border-radius: 3px; background: var(--cr-panel-2);
  box-shadow: inset 0 0 0 1px var(--cr-borde);}
div.cr-lectura .cr-seg.on {background: var(--estado);
  box-shadow: 0 0 8px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--estado) 45%, transparent));}
div.cr-lectura .cr-umbral {position: absolute; top: -6px; bottom: -6px; width: 2px; margin-left: -1px; border-radius: 2px;
  background: var(--cr-ambar); box-shadow: 0 0 6px var(--cr-brillo-umbral);}
div.cr-lectura .cr-escala {position: relative; height: 20px; font-size: .8rem; color: var(--cr-tinta-3);}
div.cr-lectura .cr-escala span {position: absolute; top: 0; white-space: nowrap;}
div.cr-lectura .cr-escala .cr-e-umbral {color: var(--cr-ambar); transform: translateX(-50%);}
div.cr-lectura .cr-escala .cr-e-fin {right: 0;}
p.cr-nota {color: var(--cr-tinta-2); font-size: .94rem; line-height: 1.55; margin: 12px 2px 0;}
p.cr-nota b {color: var(--cr-tinta);}
@media (min-width: 992px) {
  [data-testid="stColumn"]:has(.cr-lectura) {position: sticky; top: 4.25rem; align-self: flex-start;}
}

/* Rótulos de grupo, avisos, fichas y pasos */
p.cr-rotulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.02rem; color: var(--cr-tinta); margin: 0 0 2px;}
div.cr-aviso {display: flex; gap: 10px; align-items: flex-start; padding: 12px 14px; border-radius: 12px;
  background: var(--cr-panel); border: 1px solid var(--cr-borde); color: var(--cr-tinta-2); font-size: .9rem; line-height: 1.5;}
div.cr-aviso svg {flex: none; margin-top: 1px; color: var(--cr-ambar);}
div.cr-fichas {display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; margin: 4px 0 8px;}
div.cr-fichas .cr-ficha {padding: 18px 20px; background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px;}
div.cr-fichas .cr-ficha-titulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.15rem; color: var(--cr-tinta);
  margin: 0 0 8px; padding: 0;}
div.cr-fichas p {color: var(--cr-tinta-2); font-size: .96rem; line-height: 1.55; margin: 0;}
div.cr-fichas p b {color: var(--cr-tinta);}
ol.cr-pasos {list-style: none; counter-reset: paso; display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
  gap: 10px; padding: 0; margin: 6px 0 18px;}
ol.cr-pasos li {counter-increment: paso; margin: 0; padding: 14px 14px 13px; border: 1px solid var(--cr-borde);
  border-radius: 12px; background: var(--cr-panel);}
ol.cr-pasos li::before {content: counter(paso); display: block; font-family: var(--cr-cifras); font-weight: 700;
  font-size: 1.5rem; line-height: 1; color: var(--cr-cian); margin-bottom: 8px;}
ol.cr-pasos strong {display: block; color: var(--cr-tinta); font-size: 1rem; margin-bottom: 4px;}
ol.cr-pasos span {display: block; color: var(--cr-tinta-2); font-size: .88rem; line-height: 1.45;}

/* Barra lateral */
div.cr-side .cr-side-marca {display: flex; align-items: center; gap: 10px; margin: 0 0 6px; color: var(--cr-tinta);
  font-family: var(--cr-cifras); font-weight: 700; font-size: 1.4rem; font-variation-settings: "wdth" 115;}
div.cr-side .cr-side-marca svg {color: var(--cr-verde); flex: none;}
div.cr-side p {font-size: .9rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0 0 4px;}
div.cr-side .cr-side-titulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1rem; color: var(--cr-tinta);
  margin: 22px 0 6px; padding: 0;}
div.cr-side dl {margin: 0; padding: 0;}
div.cr-side dl > div {display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding: 7px 0;
  margin: 0; border-bottom: 1px solid var(--cr-borde);}
div.cr-side dt {font-size: .86rem; color: var(--cr-tinta-2); margin: 0; font-weight: 400;}
div.cr-side dd {margin: 0; padding: 0; font-family: var(--cr-cifras); font-size: .95rem; color: var(--cr-tinta); text-align: right; font-variant-numeric: tabular-nums;}
div.cr-side ul {margin: 0; padding: 0 0 0 1.05rem;}
div.cr-side li {font-size: .9rem; color: var(--cr-tinta); margin: 3px 0; padding: 0;}
div.cr-side p.cr-side-aviso {margin-top: 20px; padding: 10px 12px; border-radius: 10px; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); font-size: .84rem;}

/* Gráficos dentro de una tarjeta */
[data-testid="stPlotlyChart"] {background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px; overflow: hidden;}

/* El selector de tema del menú se oculta para que el interruptor sea el único control
   (la app lo usa por dentro para cambiar de tema) */
[data-testid="stThemeSwitcher"],
[data-testid="stThemeSwitcher"] + [data-testid="stMainMenuDivider"] {display: none !important;}

/* Pestañas */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 4px;}
[data-testid="stTabs"] button[data-baseweb="tab"] p {font-size: 1rem; font-weight: 600;}

/* Respuesta al presionar (botones) */
[data-testid^="stBaseButton"] {transition: transform 160ms var(--cr-ease-out), background-color 160ms ease,
  border-color 160ms ease, color 160ms ease;}
[data-testid^="stBaseButton"]:active {transform: scale(0.97);}

@media (max-width: 640px) {
  section.cr-hero {padding: 18px 18px 16px;}
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
        font=dict(family=FUENTE_CIFRAS, color=TINTA_2, size=13),
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


def mostrar(fig):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Carga de datos, metadatos y modelo
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_meta():
    return json.loads(RUTA_META.read_text(encoding="utf-8"))


@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


def construir_pipeline(meta):
    """Mismo pipeline del notebook. Se usa solo si el archivo .joblib no se puede cargar."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    g = meta["grupos_variables"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("escalar", StandardScaler())]), g["numericas"]),
        ("bin", SimpleImputer(strategy="most_frequent"), g["binarias"]),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), g["categoricas"]),
    ])
    return Pipeline([("prep", prep), ("modelo", LogisticRegression(**meta["modelo"]["hiperparametros"]))])


@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado; si la versión de scikit-learn no coincide, lo reconstruye
    con los mismos datos de entrenamiento e hiperparámetros (el resultado es determinista)."""
    import sklearn

    meta = cargar_meta()
    datos = cargar_datos()
    variables = meta["orden_variables"]
    entrenamiento = datos[datos["conjunto"] == "entrenamiento"]
    origen = f"modelo entrenado en el notebook (scikit-learn {meta['versiones']['scikit-learn']})"
    try:
        if meta["versiones"]["scikit-learn"] != sklearn.__version__:
            raise RuntimeError("versión de scikit-learn distinta")
        modelo = joblib.load(RUTA_MODELO)
    except Exception:
        modelo = construir_pipeline(meta).fit(entrenamiento[variables], entrenamiento["enfermedad"])
        origen = f"modelo reconstruido con scikit-learn {sklearn.__version__} (mismos datos e hiperparámetros)"
    # Paciente promedio del entrenamiento, para explicar cada predicción
    matriz = modelo.named_steps["prep"].transform(entrenamiento[variables])
    matriz = matriz.toarray() if hasattr(matriz, "toarray") else np.asarray(matriz)
    return modelo, matriz.mean(axis=0), origen


meta = cargar_meta()
datos = cargar_datos()
modelo, media_transformada, origen_modelo = cargar_modelo()
VARS = meta["orden_variables"]
INFO = meta["variables"]
TIPOS = datos[VARS].dtypes.to_dict()
UMBRALES = meta["umbrales"]
PRUEBA = meta["prueba"]
PROYECTO = meta["proyecto"]


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


def predecir(df_pacientes):
    x = df_pacientes[VARS].astype(TIPOS)
    return modelo.predict_proba(x)[:, 1]


def aportes(paciente):
    """Aporte de cada variable al log-odds, comparado con el paciente promedio del entrenamiento
    (para un modelo lineal equivale a coeficiente x (valor - promedio))."""
    prep = modelo.named_steps["prep"]
    coef = modelo.named_steps["modelo"].coef_[0]
    x = prep.transform(paciente[VARS].astype(TIPOS))
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contrib = coef * (x[0] - media_transformada)
    nombres = [n.split("__", 1)[1].split("_")[0] for n in prep.get_feature_names_out()]
    return pd.Series(contrib, index=nombres).groupby(level=0).sum()


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
    <span class="cr-criterio">{UMBRALES[criterio]['nombre']}, umbral {pct(umbral, 0)}</span></div>
  <div class="cr-lectura-valor">{num(100 * prob, 0)}<span class="cr-unidad">%</span></div>
  <div class="cr-estado">{estado}</div>
  <div class="cr-barra" aria-hidden="true">{segmentos}<span class="cr-umbral" style="left:{100 * umbral:.1f}%"></span></div>
  <div class="cr-escala" aria-hidden="true"><span>0 %</span>
    <span class="cr-e-umbral" style="left:{100 * umbral:.1f}%">umbral {pct(umbral, 0)}</span>
    <span class="cr-e-fin">100 %</span></div>
</div>"""


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
fecha = "/".join(reversed(meta["fecha_entrenamiento"].split("-")))
integrantes = "".join(f"<li>{n}</li>" for n in PROYECTO["integrantes"])
with st.sidebar:
    bloque(f"""
<div class="cr-side">
  <div class="cr-side-marca">{ICONO_MARCA}<span>CardioRiesgo</span></div>
  <p>Riesgo de enfermedad coronaria con machine learning clásico.</p>
  <p class="cr-side-titulo">Modelo en uso</p>
  <dl>
    <div><dt>Algoritmo</dt><dd>{meta['modelo']['nombre']}</dd></div>
    <div><dt>Regularización (C)</dt><dd>{num(meta['modelo']['hiperparametros']['C'], 1)}</dd></div>
    <div><dt>AUC validación cruzada</dt><dd>{num(meta['modelo']['auc_cv'], 3)}</dd></div>
    <div><dt>AUC prueba</dt><dd>{num(PRUEBA['auc'], 3)}</dd></div>
    <div><dt>Entrenamiento</dt><dd>{meta['dataset']['n_entrenamiento']} pacientes</dd></div>
    <div><dt>Prueba</dt><dd>{meta['dataset']['n_prueba']} pacientes</dd></div>
    <div><dt>Entrenado el</dt><dd>{fecha}</dd></div>
  </dl>
  <p class="cr-side-titulo">Grupo 13</p>
  <ul>{integrantes}</ul>
  <p style="margin-top:10px">Aprendizaje Automático, CUN<br>Docente: {PROYECTO['docente']}</p>
  <p class="cr-side-aviso">Herramienta académica. No reemplaza el criterio médico ni un examen diagnóstico.</p>
</div>""")
    st.caption(f"Origen: {origen_modelo}.")

# ---------------------------------------------------------------------------
# Encabezado: marca, trazo de ECG y canales del monitor
# ---------------------------------------------------------------------------
# Tema: claro por defecto. El interruptor queda enlazado a la URL (?oscuro=true), así que
# el modo elegido se mantiene al recargar o al compartir el enlace.
with st.container(horizontal_alignment="right"):
    MODO_OSCURO = st.toggle(":material/dark_mode: Modo oscuro", key="oscuro", bind="query-params",
                            help="Cambia entre el tema claro (por defecto) y el oscuro.")

primera_vez = not st.session_state.get("_intro_vista", False)
st.session_state["_intro_vista"] = True
mt = PRUEBA["metricas"]["tamizaje"]
canales_inicio = [
    canal("Pacientes", meta["dataset"]["n_pacientes"],
          f"{meta['dataset']['n_entrenamiento']} para entrenar y {meta['dataset']['n_prueba']} para probar"),
    canal("Variables clínicas", len(VARS), "edad, síntomas, signos y pruebas"),
    canal("AUC en validación cruzada", num(meta["modelo"]["auc_cv"], 3), "promedio de 25 entrenamientos", "modelo"),
    canal("AUC en prueba", num(PRUEBA["auc"], 3),
          f"IC 95 %: {num(PRUEBA['ic95']['auc'][0], 3)} a {num(PRUEBA['ic95']['auc'][1], 3)}", "modelo"),
    canal("Sensibilidad en tamizaje", pct(UMBRALES["tamizaje"]["sensibilidad_cv"], 0),
          f"umbral {num(UMBRALES['tamizaje']['valor'], 2)}; en prueba detectó {int(mt['VP'])} de {int(mt['VP'] + mt['FN'])}",
          "decision"),
]
bloque(f"""
<div class="cr-intro cr-fila{' cr-anim' if primera_vez else ''}">
<section class="cr-hero" aria-label="Presentación">
  <ul class="cr-chips" aria-label="Contexto del proyecto">
    <li>ACA 1</li><li>Grupo 13</li><li>Aprendizaje Automático</li><li>Especialización en IA, CUN</li>
  </ul>
  <div class="cr-marca" role="heading" aria-level="1">CardioRiesgo</div>
  <svg class="cr-ecg" viewBox="0 0 1200 80" aria-hidden="true" focusable="false">
    <path class="cr-traza" pathLength="1" d="{trazo_ecg()}"/>
    <g class="cr-cabeza"><circle class="cr-onda" cx="1172" cy="56" r="4.5"/><circle class="cr-punto" cx="1172" cy="56" r="4.5"/></g>
  </svg>
  <p class="cr-lema">Estimamos la probabilidad de enfermedad coronaria de un paciente a partir de 13 variables clínicas,
  con una regresión logística entrenada y evaluada con datos de 303 pacientes de la Cleveland Clinic.</p>
</section>
<div class="cr-canales cr-n5" aria-label="Resumen del estudio">{''.join(canales_inicio)}</div>
</div>""")

tab_pac, tab_datos, tab_modelo, tab_lote, tab_acerca = st.tabs([
    ":material/monitor_heart: Evaluar paciente",
    ":material/query_stats: Explorar datos",
    ":material/tune: Umbral y desempeño",
    ":material/table_view: Evaluación por lote",
    ":material/menu_book: El proyecto",
])

# ---------------------------------------------------------------------------
# Pestaña 1: evaluar un paciente
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
    ayuda = info.get("ayuda")
    etiqueta = info["etiqueta"] + (f" ({info['unidad']})" if info.get("unidad") else "")
    if info["tipo"] in ("binaria", "categorica"):
        codigos = [int(c) for c in info["opciones"].keys()]
        return contenedor.selectbox(etiqueta, codigos, key=clave, help=ayuda,
                                    format_func=lambda c, i=info: i["opciones"][str(c)])
    paso = 0.1 if var == "oldpeak" else 1
    minimo, maximo = info["rango_entrada"]
    if var == "oldpeak":
        return contenedor.number_input(etiqueta, min_value=float(minimo), max_value=float(maximo),
                                       step=paso, key=clave, help=ayuda, format="%.1f")
    return contenedor.number_input(etiqueta, min_value=int(minimo), max_value=int(maximo),
                                   step=int(paso), key=clave, help=ayuda)


with tab_pac:
    col_form, col_res = st.columns([1.12, 1], gap="large")

    with col_form:
        st.subheader("Datos del paciente")
        b1, b2 = st.columns(2)
        b1.button("Cargar ejemplo de riesgo bajo", on_click=cargar_ejemplo, args=("bajo",), width="stretch")
        b2.button("Cargar ejemplo de riesgo alto", on_click=cargar_ejemplo, args=("alto",), width="stretch")
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
            format_func=lambda c: f"{UMBRALES[c]['nombre']} (umbral {pct(UMBRALES[c]['valor'], 0)})",
            horizontal=True,
            help="Tamizaje prioriza no dejar pasar enfermos (más sensibilidad). "
                 "Estándar trata igual los dos errores y genera menos falsas alarmas (más especificidad).",
        )
        umbral = UMBRALES[criterio]["valor"]
        bloque(lectura(prob, umbral, criterio))
        ref = UMBRALES[criterio]
        bloque(
            f'<p class="cr-nota">Con este criterio, en validación cruzada el modelo detectó al '
            f"<b>{pct(ref['sensibilidad_cv'], 0)}</b> de los pacientes enfermos y descartó correctamente "
            f"al <b>{pct(ref['especificidad_cv'], 0)}</b> de los sanos.</p>"
        )

    st.write("")
    c_graf, c_guia = st.columns([1.45, 1], gap="large")
    with c_graf:
        serie = aportes(paciente)
        top = serie.reindex(serie.abs().sort_values(ascending=False).index).head(8)[::-1]
        etiquetas = [f"{INFO[v]['etiqueta_corta']}: {etiqueta_valor(v, paciente.iloc[0][v])}" for v in top.index]
        colores = [NARANJA if v > 0 else AZUL for v in top.values]
        fig = go.Figure(go.Bar(
            x=top.values, y=etiquetas, orientation="h", marker_color=colores, width=0.55,
            hovertemplate="%{y}<br>Aporte: %{x:.2f} log-odds<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=EJE, line_width=1)
        estilo(fig, alto=360, leyenda=False, titulo="Qué empujó el resultado")
        fig.update_xaxes(title="← baja el riesgo   ·   sube el riesgo →", showgrid=True, gridcolor=REJILLA)
        fig.update_yaxes(showgrid=False)
        mostrar(fig)
        st.caption("Aporte de cada variable al log-odds, comparado con el paciente promedio del entrenamiento.")
    with c_guia:
        with st.container(border=True):
            rotulo("Cómo leer este resultado")
            st.markdown(
                "- La **probabilidad** sale de una regresión logística: el modelo suma el peso de cada variable y lo convierte en un valor entre 0 y 100 %.\n"
                "- El **umbral** decide desde qué probabilidad se marca riesgo alto. En tamizaje lo bajamos para no dejar pasar enfermos, a costa de más falsas alarmas.\n"
                "- El gráfico compara al paciente con el **paciente promedio** del entrenamiento: las barras naranjas suben el riesgo y las azules lo bajan.\n"
                "- El diagnóstico de referencia del dataset es la **angiografía coronaria**: estrechamiento mayor al 50 % en al menos un vaso principal."
            )
        aviso("Este resultado es una estimación estadística hecha con datos de un solo hospital (Cleveland Clinic), "
              "publicados en 1988. Sirve como apoyo académico y no reemplaza el criterio médico ni la angiografía.")

# ---------------------------------------------------------------------------
# Pestaña 2: explorar los datos
# ---------------------------------------------------------------------------
with tab_datos:
    st.subheader("Exploración del dataset")
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
            canal("FC máxima mediana", f"{num(vista['thalach'].median(), 0)} lpm", "frecuencia cardiaca en esfuerzo"),
        ])

        def tasa_por(var):
            t = vista.groupby(var, dropna=True)["enfermedad"].agg(["mean", "count"]).reset_index()
            t["Categoría"] = [etiqueta_valor(var, v) for v in t[var]]
            if var == "ca":
                t["Categoría"] = [f"{c} vaso" if c == "1" else f"{c} vasos" for c in t["Categoría"]]
            return t

        def barra_tasa(var, titulo):
            """Barras horizontales: las etiquetas largas se leen sin rotarlas."""
            t = tasa_por(var)
            fig = go.Figure(go.Bar(
                y=t["Categoría"], x=100 * t["mean"], orientation="h", marker_color=AZUL, width=0.55,
                text=[pct(v, 0) for v in t["mean"]], textposition="outside", textfont=dict(color=TINTA_2),
                customdata=t["count"], cliponaxis=False,
                hovertemplate="%{y}<br>Con enfermedad: %{x:.1f} %<br>Pacientes: %{customdata}<extra></extra>",
            ))
            estilo(fig, alto=300, leyenda=False, titulo=titulo)
            fig.update_xaxes(title="% con enfermedad", range=[0, 118], showgrid=True, gridcolor=REJILLA,
                             tickvals=[0, 25, 50, 75, 100])
            fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
            return fig

        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure()
            for dx, val in [("Sin enfermedad", 0), ("Con enfermedad", 1)]:
                sub = vista[vista["enfermedad"] == val]
                fig.add_trace(go.Histogram(x=sub["age"], name=dx, marker_color=COLOR_DX[dx],
                                           xbins=dict(start=25, end=80, size=5),
                                           hovertemplate="Edad %{x}<br>Pacientes: %{y}<extra>" + dx + "</extra>"))
            estilo(fig, titulo="Edad según diagnóstico")
            fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
            fig.update_xaxes(title="Edad (años)")
            fig.update_yaxes(title="Pacientes")
            mostrar(fig)
        with g2:
            fig = go.Figure()
            for dx, val in [("Sin enfermedad", 0), ("Con enfermedad", 1)]:
                sub = vista[vista["enfermedad"] == val]
                fig.add_trace(go.Scatter(
                    x=sub["age"], y=sub["thalach"], mode="markers", name=dx,
                    marker=dict(color=COLOR_DX[dx], size=8, line=dict(color=PANEL, width=1.5), opacity=0.9),
                    hovertemplate="Edad %{x} · FC máx. %{y} lpm<extra>" + dx + "</extra>"))
            estilo(fig, titulo="Frecuencia cardiaca máxima frente a edad")
            fig.update_xaxes(title="Edad (años)")
            fig.update_yaxes(title="FC máxima (lpm)")
            mostrar(fig)

        g3, g4, g5 = st.columns(3)
        with g3:
            mostrar(barra_tasa("cp", "Enfermedad por tipo de dolor torácico"))
        with g4:
            mostrar(barra_tasa("thal", "Enfermedad por prueba de talio"))
        with g5:
            mostrar(barra_tasa("ca", "Enfermedad por vasos coloreados"))

        g6, g7 = st.columns(2)
        with g6:
            fig = go.Figure()
            for dx, val in [("Sin enfermedad", 0), ("Con enfermedad", 1)]:
                sub = vista[vista["enfermedad"] == val]
                fig.add_trace(go.Box(y=sub["oldpeak"], name=dx, marker_color=COLOR_DX[dx], boxmean=False,
                                     line=dict(width=1.5), fillcolor=rgba(COLOR_DX[dx], 0.2)))
            estilo(fig, alto=370, leyenda=False, titulo="Depresión del ST según diagnóstico")
            fig.update_yaxes(title="Depresión del ST (mm)")
            mostrar(fig)
        with g7:
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
            estilo(fig, alto=370, leyenda=False, titulo="Correlación entre variables numéricas")
            fig.update_xaxes(tickangle=-30, showgrid=False, linecolor="rgba(0,0,0,0)", ticks="")
            fig.update_yaxes(autorange="reversed", showgrid=False, linecolor="rgba(0,0,0,0)")
            mostrar(fig)

        with st.expander("Ver tabla de pacientes"):
            tabla = vista[VARS + ["enfermedad", "conjunto"]].copy()
            tabla["diagnóstico"] = tabla["enfermedad"].map({0: "Sin enfermedad", 1: "Con enfermedad"})
            st.dataframe(tabla.drop(columns="enfermedad"), hide_index=True, height=320)
            st.download_button("Descargar selección (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                               "pacientes_filtrados.csv", "text/csv", icon=":material/download:")

# ---------------------------------------------------------------------------
# Pestaña 3: umbral y desempeño del modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("¿Qué tan bien funciona el modelo?")
    y_true = np.array(PRUEBA["y_true"])
    y_prob = np.array(PRUEBA["y_proba"])

    with st.container(border=True):
        umbral_sel = st.slider("Umbral de decisión para el conjunto de prueba", 0.05, 0.95,
                               float(UMBRALES["tamizaje"]["valor"]), 0.01,
                               help="Mueva el umbral y observe cómo cambian la matriz de confusión y las métricas.")
    pred = (y_prob >= umbral_sel).astype(int)
    vp = int(((pred == 1) & (y_true == 1)).sum())
    vn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    total = len(y_true)
    sens = vp / (vp + fn) if vp + fn else 0
    espec = vn / (vn + fp) if vn + fp else 0
    prec = vp / (vp + fp) if vp + fp else 0
    exac = (vp + vn) / total
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0

    fila_canales([
        canal("Exactitud", pct(exac), f"{vp + vn} de {total} aciertos", "modelo"),
        canal("Sensibilidad", pct(sens), f"{vp} de {vp + fn} enfermos detectados", "modelo"),
        canal("Especificidad", pct(espec), f"{vn} de {vn + fp} sanos descartados", "modelo"),
        canal("Precisión", pct(prec), "de las alertas son enfermos", "modelo"),
        canal("F1", num(f1, 3), "equilibrio entre las dos", "modelo"),
        canal("AUC ROC", num(PRUEBA["auc"], 3), "no depende del umbral", "modelo"),
    ])
    st.caption(f"Conjunto de prueba: {total} pacientes que el modelo no vio durante el entrenamiento.")

    c1, c2 = st.columns(2)
    with c1:
        z = [[vn, fp], [fn, vp]]
        textos = [[f"Verdaderos negativos<br><b>{vn}</b>", f"Falsos positivos<br><b>{fp}</b>"],
                  [f"Falsos negativos<br><b>{fn}</b>", f"Verdaderos positivos<br><b>{vp}</b>"]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicho: sin enfermedad", "Predicho: con enfermedad"],
            y=["Real: sin enfermedad", "Real: con enfermedad"],
            colorscale=[[0, "#334a6b"], [1, "#2f6fd0"]], showscale=False,
            text=textos, texttemplate="%{text}", textfont=dict(size=14, color=TEXTO_CELDA),
            hovertemplate="%{y} · %{x}: %{z}<extra></extra>", xgap=4, ygap=4))
        estilo(fig, alto=370, leyenda=False, titulo=f"Matriz de confusión (umbral {pct(umbral_sel, 0)})")
        fig.update_yaxes(autorange="reversed", showgrid=False, linecolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=False, linecolor="rgba(0,0,0,0)", ticks="")
        mostrar(fig)
    with c2:
        roc = PRUEBA["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS_BARRA, width=1, dash="dot"),
                                 name="Azar", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=AZUL, width=2.5),
                                 fill="tozeroy", fillcolor="rgba(66,146,244,0.08)",
                                 name=f"Regresión logística (AUC {num(PRUEBA['auc'], 3)})",
                                 hovertemplate="1 - especificidad: %{x:.2f}<br>Sensibilidad: %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[1 - espec], y=[sens], mode="markers", name="Umbral elegido",
                                 marker=dict(color=AMBAR, size=12, line=dict(color=PANEL, width=2))))
        estilo(fig, alto=370, titulo="Curva ROC en el conjunto de prueba")
        fig.update_layout(legend=dict(orientation="v", x=0.98, xanchor="right", y=0.04, yanchor="bottom",
                                      bgcolor=rgba(PANEL_2, 0.9), bordercolor=BORDE, borderwidth=1))
        fig.update_xaxes(title="1 - especificidad (falsas alarmas)", range=[-0.02, 1.02])
        fig.update_yaxes(title="Sensibilidad", range=[-0.02, 1.05])
        mostrar(fig)

    c3, c4 = st.columns(2)
    with c3:
        cv = pd.DataFrame(meta["comparacion_cv"]).sort_values("roc_auc")
        elegido = meta["modelo"]["nombre"]
        colores = [AZUL if n == elegido else GRIS_BARRA for n in cv["modelo"]]
        fig = go.Figure(go.Bar(
            x=cv["roc_auc"], y=cv["modelo"], orientation="h", marker_color=colores, width=0.55,
            error_x=dict(type="data", array=cv["roc_auc_sd"], color=TINTA_3, thickness=1, width=4),
            customdata=cv["roc_auc_sd"],
            hovertemplate="%{y}<br>AUC promedio: %{x:.3f} ± %{customdata:.3f}<extra></extra>"))
        for fila in cv.itertuples():
            fig.add_annotation(x=fila.roc_auc + fila.roc_auc_sd + 0.008, y=fila.modelo, text=num(fila.roc_auc, 3),
                               xanchor="left", showarrow=False,
                               font=dict(size=12, color=TINTA if fila.modelo == elegido else TINTA_2))
        estilo(fig, alto=370, leyenda=False, titulo="Comparación de 7 modelos (validación cruzada 5×5)")
        fig.update_xaxes(title="AUC ROC promedio (± desviación estándar)", range=[0.5, 1.03], showgrid=True,
                         gridcolor=REJILLA)
        fig.update_yaxes(showgrid=False)
        mostrar(fig)
    with c4:
        oof = pd.DataFrame(meta["umbral_cv"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=oof["umbral"], y=oof["sensibilidad"], name="Sensibilidad",
                                 line=dict(color=NARANJA, width=2.5)))
        fig.add_trace(go.Scatter(x=oof["umbral"], y=oof["especificidad"], name="Especificidad",
                                 line=dict(color=AZUL, width=2.5)))
        for c in ("tamizaje", "balanceado"):
            fig.add_vline(x=UMBRALES[c]["valor"], line_color=AMBAR, line_width=1, line_dash="dot")
            fig.add_annotation(x=UMBRALES[c]["valor"], y=0.04, xanchor="left", xshift=5, showarrow=False,
                               text=f"{UMBRALES[c]['nombre']} {num(UMBRALES[c]['valor'], 2)}",
                               font=dict(size=11, color=TINTA_2))
        estilo(fig, alto=370, titulo="Sensibilidad y especificidad según el umbral")
        fig.update_xaxes(title="Umbral de decisión (validación cruzada)", range=[0.05, 0.95])
        fig.update_yaxes(title="Proporción", range=[0, 1.05])
        mostrar(fig)

    ors = pd.DataFrame(meta["odds_ratios"]).head(10)[::-1]
    marcas_or = [0.5, 0.67, 1, 1.5, 2, 3]
    fig = go.Figure(go.Bar(
        x=ors["log_or"], y=ors["etiqueta_corta"], orientation="h",
        marker_color=[NARANJA if v > 0 else AZUL for v in ors["log_or"]], width=0.55,
        customdata=np.stack([ors["odds_ratio"], ors["comparacion"]], axis=-1),
        hovertemplate="%{customdata[1]}<br>Odds ratio: %{customdata[0]:.2f}<extra></extra>"))
    for fila in ors.itertuples():
        fig.add_annotation(x=fila.log_or + (0.04 if fila.log_or > 0 else -0.04), y=fila.etiqueta_corta,
                           text=num(fila.odds_ratio, 2), showarrow=False,
                           xanchor="left" if fila.log_or > 0 else "right", font=dict(size=12, color=TINTA_2))
    fig.add_vline(x=0, line_color=EJE, line_width=1)
    estilo(fig, alto=440, leyenda=False, titulo="Odds ratios del modelo (los 10 mayores)")
    fig.update_xaxes(title="Odds ratio en escala logarítmica (1 = sin efecto; >1 sube el riesgo, <1 lo baja)",
                     tickvals=list(np.log(marcas_or)), ticktext=[num(v, 2).rstrip("0").rstrip(",") for v in marcas_or],
                     range=[np.log(0.45), np.log(3.8)], showgrid=True, gridcolor=REJILLA)
    fig.update_yaxes(showgrid=False)
    mostrar(fig)
    st.caption("Cada categoría se compara con su referencia (angina típica, ECG normal, pendiente ascendente, talio normal, "
               "mujer); las numéricas, por unidad clínica. El modelo está regularizado, así que los valores están "
               "encogidos hacia 1 y no son efectos causales.")

    with st.expander("Tabla de comparación de modelos"):
        tabla = pd.DataFrame(meta["comparacion_cv"]).rename(columns={
            "modelo": "Modelo", "roc_auc": "AUC", "roc_auc_sd": "AUC (desv.)", "accuracy": "Exactitud",
            "recall": "Sensibilidad", "precision": "Precisión", "f1": "F1"})
        st.dataframe(tabla.style.format({c: "{:.3f}" for c in tabla.columns if c != "Modelo"}, decimal=","),
                     hide_index=True)
        st.markdown(f"La validación cruzada estima un AUC de **{num(meta['modelo']['auc_cv'], 3)}** para pacientes nuevos. "
                    f"En la prueba salió {num(PRUEBA['auc'], 3)}: con solo {total} pacientes la cifra varía bastante "
                    f"(intervalo de confianza 95 %: {num(PRUEBA['ic95']['auc'][0], 3)} a {num(PRUEBA['ic95']['auc'][1], 3)}).")

# ---------------------------------------------------------------------------
# Pestaña 4: evaluación por lote
# ---------------------------------------------------------------------------
with tab_lote:
    st.subheader("Evaluar varios pacientes con un archivo CSV")
    st.markdown("El archivo debe tener una fila por paciente y las 13 columnas del dataset con los códigos originales "
                "de UCI. Descargue la plantilla para ver el formato.")
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
                umbral_lote = UMBRALES["tamizaje"]["valor"]
                salida = lote.copy()
                salida["probabilidad (%)"] = np.round(100 * prob_lote, 1)
                salida["clasificación"] = np.where(prob_lote >= umbral_lote, "Riesgo alto", "Riesgo bajo")
                n_alto = int((prob_lote >= umbral_lote).sum())
                st.success(f"{len(salida)} pacientes evaluados con el criterio de tamizaje "
                           f"(umbral {pct(umbral_lote, 0)}): {n_alto} con riesgo alto.")
                st.dataframe(salida, hide_index=True)
                st.download_button("Descargar resultados (CSV)", salida.to_csv(index=False).encode("utf-8"),
                                   "resultados_cardioriesgo.csv", "text/csv", icon=":material/download:")

# ---------------------------------------------------------------------------
# Pestaña 5: el proyecto
# ---------------------------------------------------------------------------
with tab_acerca:
    ds = meta["dataset"]
    st.subheader("El proyecto")
    bloque(f"""
<div class="cr-fichas">
  <div class="cr-ficha"><p class="cr-ficha-titulo">La pregunta</p><p>{PROYECTO['pregunta']}</p></div>
  <div class="cr-ficha"><p class="cr-ficha-titulo">Los datos</p><p><b>{ds['n_pacientes']} pacientes</b> de la Cleveland Clinic, publicados
    en 1988, con 13 variables clínicas cada uno. La etiqueta viene de la angiografía coronaria: hay enfermedad si al
    menos un vaso principal tiene un estrechamiento mayor al 50 %. El <b>{pct(ds['prevalencia'])}</b> de los pacientes
    la tiene.</p></div>
  <div class="cr-ficha"><p class="cr-ficha-titulo">La decisión</p><p>Un falso negativo manda a casa a un paciente enfermo sin estudio. Por eso,
    además del umbral estándar de {num(UMBRALES['balanceado']['valor'], 2)}, ofrecemos uno de tamizaje de
    {num(UMBRALES['tamizaje']['valor'], 2)} que en validación cruzada detecta al <b>{pct(UMBRALES['tamizaje']['sensibilidad_cv'], 0)}</b>
    de los enfermos, a cambio de más falsas alarmas.</p></div>
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
        st.subheader("Objetivo")
        st.markdown(PROYECTO["objetivo"])
        st.subheader("Referencias")
        for r in PROYECTO["referencias"]:
            st.markdown(f"- {r}")
    with c2:
        st.subheader("Grupo 13")
        st.markdown("\n".join(f"- {n}" for n in PROYECTO["integrantes"]))
        st.markdown(f"Especialización en Inteligencia Artificial, CUN  \nAprendizaje Automático, {PROYECTO['docente']}")
        enlaces = PROYECTO.get("enlaces", {})
        st.subheader("Enlaces")
        for nombre, url in [("Aplicación", enlaces.get("app")), ("Video de sustentación", enlaces.get("video")),
                            ("Repositorio", enlaces.get("repositorio"))]:
            st.markdown(f"- {nombre}: {url if url else 'pendiente'}")
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
