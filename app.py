# -*- coding: utf-8 -*-
"""
CardioRiesgo - ACA 1 Aprendizaje Automático (CUN), Grupo 13.
Aplicación Streamlit que usa el modelo de regresión logística entrenado en el
notebook del proyecto con el dataset UCI Heart Disease (Cleveland).

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
# Rutas y constantes
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo" / "modelo_cardio.joblib"
RUTA_META = BASE / "modelo" / "metadata.json"
RUTA_DATOS = BASE / "data" / "heart_clean.csv"
RUTA_PLANTILLA = BASE / "data" / "plantilla_pacientes.csv"

AZUL = "#2a78d6"      # Sin enfermedad / disminuye el riesgo
NARANJA = "#eb6834"   # Con enfermedad / aumenta el riesgo
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRIS = "#898781"
GRIS_CLARO = "#c3c2b7"
REJILLA = "#e1e0d9"
FONDO = "#fcfcfb"
ALERTA = "#d03b3b"    # estado: riesgo alto (siempre con ícono y texto)
BIEN = "#0ca30c"      # estado: riesgo bajo (siempre con ícono y texto)
COLOR_DX = {"Sin enfermedad": AZUL, "Con enfermedad": NARANJA}

st.set_page_config(
    page_title="CardioRiesgo · ACA 1 Grupo 13",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1280px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.tarjeta {{
    padding: 16px 18px; border-radius: 14px; background: #ffffff;
    border: 1px solid rgba(11,11,11,.08); box-shadow: 0 4px 14px rgba(0,0,0,.04);
}}
.kpi-label {{font-size: .82rem; color: {TINTA_2}; margin-bottom: 4px;}}
.kpi-valor {{font-size: 1.65rem; font-weight: 650; color: {TINTA}; line-height: 1.15;}}
.kpi-nota {{font-size: .78rem; color: {GRIS}; margin-top: 4px;}}
.hero {{font-size: 3.4rem; font-weight: 700; color: {TINTA}; line-height: 1;}}
.hero-label {{font-size: .9rem; color: {TINTA_2}; margin-bottom: 6px;}}
.estado {{display: inline-block; padding: 6px 12px; border-radius: 999px; font-weight: 600;
          font-size: .95rem; margin: 10px 0 6px 0;}}
.medidor {{position: relative; height: 14px; border-radius: 999px; background: #eceae4; margin: 14px 0 6px 0;}}
.medidor-relleno {{position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;}}
.medidor-umbral {{position: absolute; top: -5px; width: 2px; height: 24px; background: {TINTA};}}
.medidor-escala {{display: flex; justify-content: space-between; font-size: .75rem; color: {GRIS};}}
.aviso {{padding: 10px 14px; border-radius: 10px; background: #f6f5f1; color: {TINTA_2};
         font-size: .85rem; border-left: 3px solid {GRIS_CLARO};}}
.texto-suave {{color: {TINTA_2}; font-size: .92rem;}}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utilidades de formato
# ---------------------------------------------------------------------------
def num(x, dec=1):
    """Número con coma decimal (formato usado en Colombia)."""
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, dec=1):
    return f"{num(100 * x, dec)} %"


def tarjeta_kpi(etiqueta, valor, nota=""):
    nota_html = f'<div class="kpi-nota">{nota}</div>' if nota else ""
    return (
        f'<div class="tarjeta"><div class="kpi-label">{etiqueta}</div>'
        f'<div class="kpi-valor">{valor}</div>{nota_html}</div>'
    )


def estilo(fig, alto=340, leyenda=True, titulo=None):
    fig.update_layout(
        height=alto,
        margin=dict(l=8, r=8, t=56 if titulo else 24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=FONDO,
        separators=",.",
        font=dict(family="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif", color=TINTA_2, size=13),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None,
                    font=dict(color=TINTA_2)),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=TINTA),
        bargap=0.45,
    )
    if titulo:
        fig.update_layout(title=dict(text=titulo, x=0, xanchor="left", font=dict(size=15, color=TINTA)))
    fig.update_xaxes(automargin=True, showgrid=False, linecolor=GRIS_CLARO, tickcolor=GRIS_CLARO, ticks="outside",
                     title_font=dict(color=TINTA_2, size=12), tickfont=dict(color=TINTA_2), title_standoff=10)
    fig.update_yaxes(automargin=True, gridcolor=REJILLA, gridwidth=1, zeroline=False, linecolor=GRIS_CLARO,
                     title_font=dict(color=TINTA_2, size=12), tickfont=dict(color=TINTA_2), title_standoff=10)
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


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🫀 CardioRiesgo")
st.markdown(
    '<p class="texto-suave">Estimación del riesgo de enfermedad coronaria con Machine Learning clásico · '
    "Dataset UCI Heart Disease (Cleveland) · ACA 1 Aprendizaje Automático, CUN · Grupo 13</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Sobre el modelo")
    st.markdown(
        f"**{meta['modelo']['nombre']}** con preprocesamiento en *pipeline*.\n\n"
        f"AUC en validación cruzada: **{num(meta['modelo']['auc_cv'], 3)}**\n\n"
        f"Pacientes de entrenamiento: **{meta['dataset']['n_entrenamiento']}** · prueba: **{meta['dataset']['n_prueba']}**"
    )
    st.caption(f"Origen: {origen_modelo}.")
    st.markdown("---")
    st.caption("Herramienta académica. No reemplaza el criterio médico ni un examen diagnóstico.")

tab_pac, tab_datos, tab_modelo, tab_lote, tab_acerca = st.tabs(
    ["🩺 Evaluar paciente", "📊 Explorar datos", "📈 Desempeño del modelo", "📁 Evaluación por lote", "ℹ️ Acerca del proyecto"]
)

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
        b1, b2, _ = st.columns([1, 1, 0.6])
        b1.button("Cargar ejemplo de riesgo bajo", on_click=cargar_ejemplo, args=("bajo",), width="stretch")
        b2.button("Cargar ejemplo de riesgo alto", on_click=cargar_ejemplo, args=("alto",), width="stretch")
        st.caption("Los ejemplos son pacientes reales del conjunto de prueba (el modelo no los vio al entrenar).")

        st.markdown("**Datos generales y síntomas**")
        c1, c2 = st.columns(2)
        entrada("age", c1)
        entrada("sex", c2)
        entrada("cp", c1)
        entrada("exang", c2)

        st.markdown("**Signos vitales y laboratorio**")
        c1, c2, c3 = st.columns(3)
        entrada("trestbps", c1)
        entrada("chol", c2)
        entrada("fbs", c3)

        st.markdown("**Pruebas diagnósticas**")
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
        alto = prob >= umbral
        color_estado = ALERTA if alto else BIEN
        texto_estado = "⚠️ Riesgo alto: se sugiere estudio cardiológico" if alto else "✅ Riesgo bajo según el modelo"

        st.markdown(
            f"""
<div class="tarjeta">
  <div class="hero-label">Probabilidad estimada de enfermedad coronaria</div>
  <div class="hero">{pct(prob, 0)}</div>
  <div class="estado" style="background:{color_estado}1a; color:{color_estado}; border:1px solid {color_estado}55;">{texto_estado}</div>
  <div class="medidor">
    <div class="medidor-relleno" style="width:{100 * prob:.1f}%; background:{color_estado};"></div>
    <div class="medidor-umbral" style="left:calc({100 * umbral:.1f}% - 1px);" title="Umbral de decisión"></div>
  </div>
  <div class="medidor-escala"><span>0 %</span><span>umbral {pct(umbral, 0)}</span><span>100 %</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        ref = UMBRALES[criterio]
        st.markdown(
            f'<p class="texto-suave" style="margin-top:12px;">Con este criterio, en validación cruzada el modelo '
            f"detectó al <b>{pct(ref['sensibilidad_cv'], 0)}</b> de los pacientes enfermos y descartó correctamente "
            f"al <b>{pct(ref['especificidad_cv'], 0)}</b> de los sanos.</p>",
            unsafe_allow_html=True,
        )

        # Explicación de la predicción
        serie = aportes(paciente)
        top = serie.reindex(serie.abs().sort_values(ascending=False).index).head(8)[::-1]
        etiquetas = [f"{INFO[v]['etiqueta_corta']}: {etiqueta_valor(v, paciente.iloc[0][v])}" for v in top.index]
        colores = [NARANJA if v > 0 else AZUL for v in top.values]
        fig = go.Figure(go.Bar(
            x=top.values, y=etiquetas, orientation="h", marker_color=colores, width=0.55,
            hovertemplate="%{y}<br>Aporte: %{x:.2f} log-odds<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=GRIS_CLARO, line_width=1)
        estilo(fig, alto=340, leyenda=False, titulo="Qué empujó el resultado")
        fig.update_xaxes(title="← baja el riesgo · sube el riesgo →", showgrid=True, gridcolor=REJILLA,
                         zeroline=False)
        fig.update_yaxes(showgrid=False)
        mostrar(fig)
        st.caption("Aporte de cada variable al log-odds, comparado con el paciente promedio del entrenamiento.")

    st.markdown(
        '<div class="aviso">Este resultado es una estimación estadística hecha con datos de un solo hospital '
        "(Cleveland Clinic), publicados en 1988. Sirve como apoyo académico y no reemplaza el criterio médico ni la angiografía.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("¿Cómo leer este resultado?"):
        st.markdown(
            "- La **probabilidad** sale de una regresión logística: el modelo suma el peso de cada variable y lo convierte en un valor entre 0 y 100 %.\n"
            "- El **umbral** decide desde qué probabilidad se marca riesgo alto. En tamizaje lo bajamos para no dejar pasar enfermos, a costa de más falsas alarmas.\n"
            "- El gráfico compara al paciente con el **paciente promedio** del entrenamiento. Las barras naranjas suben el riesgo y las azules lo bajan.\n"
            "- El diagnóstico de referencia del dataset es la **angiografía coronaria**: estrechamiento mayor al 50 % en al menos un vaso principal."
        )

# ---------------------------------------------------------------------------
# Pestaña 2: explorar los datos
# ---------------------------------------------------------------------------
with tab_datos:
    st.subheader("Exploración del dataset")
    f1, f2, f3 = st.columns([1, 1.4, 1])
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
        st.warning("No hay pacientes con esa combinación de filtros.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(tarjeta_kpi("Pacientes", f"{len(vista):,}".replace(",", ".")), unsafe_allow_html=True)
        k2.markdown(tarjeta_kpi("Con enfermedad", pct(vista["enfermedad"].mean()),
                                f"{int(vista['enfermedad'].sum())} pacientes"), unsafe_allow_html=True)
        k3.markdown(tarjeta_kpi("Edad mediana", f"{num(vista['age'].median(), 0)} años"), unsafe_allow_html=True)
        k4.markdown(tarjeta_kpi("Frecuencia cardiaca máx. mediana", f"{num(vista['thalach'].median(), 0)} lpm"),
                    unsafe_allow_html=True)
        st.write("")

        def tasa_por(var):
            t = vista.groupby(var, dropna=True)["enfermedad"].agg(["mean", "count"]).reset_index()
            t["Categoría"] = [etiqueta_valor(var, v) for v in t[var]]
            return t

        def barra_tasa(var, titulo):
            t = tasa_por(var)
            fig = go.Figure(go.Bar(
                x=t["Categoría"], y=100 * t["mean"], marker_color=AZUL, width=0.5,
                text=[pct(v, 0) for v in t["mean"]], textposition="outside", textfont=dict(color=TINTA_2),
                customdata=t["count"],
                hovertemplate="%{x}<br>Con enfermedad: %{y:.1f} %<br>Pacientes: %{customdata}<extra></extra>",
            ))
            estilo(fig, alto=320, leyenda=False, titulo=titulo)
            fig.update_yaxes(title="% con enfermedad", range=[0, 105])
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
                    marker=dict(color=COLOR_DX[dx], size=8, line=dict(color=FONDO, width=1.5), opacity=0.85),
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
                                     line=dict(width=1.5)))
            estilo(fig, leyenda=False, titulo="Depresión del ST según diagnóstico")
            fig.update_yaxes(title="Depresión del ST (mm)")
            mostrar(fig)
        with g7:
            cols_corr = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "enfermedad"]
            corr = vista[cols_corr].corr()
            nombres = [INFO[c]["etiqueta_corta"] if c in INFO else "Enfermedad" for c in cols_corr]
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=nombres, y=nombres, zmin=-1, zmax=1,
                colorscale=[[0, AZUL], [0.5, "#f0efec"], [1, NARANJA]],
                text=[[num(round(v, 2) + 0.0, 2) for v in fila] for fila in corr.values], texttemplate="%{text}",
                textfont=dict(size=11, color=TINTA), xgap=2, ygap=2,
                hovertemplate="%{y} · %{x}<br>r = %{z:.2f}<extra></extra>", colorbar=dict(thickness=10)))
            estilo(fig, alto=360, leyenda=False, titulo="Correlación entre variables numéricas")
            fig.update_xaxes(tickangle=-30, showgrid=False)
            fig.update_yaxes(autorange="reversed", showgrid=False)
            mostrar(fig)

        with st.expander("Ver tabla de pacientes"):
            tabla = vista[VARS + ["enfermedad", "conjunto"]].copy()
            tabla["diagnóstico"] = tabla["enfermedad"].map({0: "Sin enfermedad", 1: "Con enfermedad"})
            st.dataframe(tabla.drop(columns="enfermedad"), hide_index=True, height=320)
            st.download_button("Descargar selección (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                               "pacientes_filtrados.csv", "text/csv")

# ---------------------------------------------------------------------------
# Pestaña 3: desempeño del modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("¿Qué tan bien funciona el modelo?")
    prueba = meta["prueba"]
    y_true = np.array(prueba["y_true"])
    y_prob = np.array(prueba["y_proba"])

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

    m = st.columns(6)
    m[0].markdown(tarjeta_kpi("Exactitud", pct(exac), f"{vp + vn} de {total} aciertos"), unsafe_allow_html=True)
    m[1].markdown(tarjeta_kpi("Sensibilidad", pct(sens), f"{vp} de {vp + fn} enfermos"), unsafe_allow_html=True)
    m[2].markdown(tarjeta_kpi("Especificidad", pct(espec), f"{vn} de {vn + fp} sanos"), unsafe_allow_html=True)
    m[3].markdown(tarjeta_kpi("Precisión", pct(prec), "de las alertas"), unsafe_allow_html=True)
    m[4].markdown(tarjeta_kpi("F1", num(f1, 3)), unsafe_allow_html=True)
    m[5].markdown(tarjeta_kpi("AUC ROC", num(prueba["auc"], 3), "no depende del umbral"), unsafe_allow_html=True)
    st.caption(f"Conjunto de prueba: {total} pacientes que el modelo no vio durante el entrenamiento.")

    c1, c2 = st.columns(2)
    with c1:
        z = [[vn, fp], [fn, vp]]
        textos = [[f"Verdaderos negativos<br><b>{vn}</b>", f"Falsos positivos<br><b>{fp}</b>"],
                  [f"Falsos negativos<br><b>{fn}</b>", f"Verdaderos positivos<br><b>{vp}</b>"]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicho: sin enfermedad", "Predicho: con enfermedad"],
            y=["Real: sin enfermedad", "Real: con enfermedad"],
            colorscale=[[0, "#f3f7fd"], [1, "#86b6ef"]], showscale=False,
            text=textos, texttemplate="%{text}", textfont=dict(size=14, color=TINTA),
            hovertemplate="%{y} · %{x}: %{z}<extra></extra>", xgap=3, ygap=3))
        estilo(fig, alto=360, leyenda=False, titulo=f"Matriz de confusión (umbral {pct(umbral_sel, 0)})")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False)
        mostrar(fig)
    with c2:
        roc = prueba["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS_CLARO, width=1),
                                 name="Azar", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=AZUL, width=2.5),
                                 name=f"Regresión logística (AUC {num(prueba['auc'], 3)})",
                                 hovertemplate="1 - especificidad: %{x:.2f}<br>Sensibilidad: %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[1 - espec], y=[sens], mode="markers", name="Umbral elegido",
                                 marker=dict(color=NARANJA, size=11, line=dict(color="white", width=2))))
        estilo(fig, alto=360, titulo="Curva ROC en el conjunto de prueba")
        fig.update_xaxes(title="1 - especificidad (falsas alarmas)", range=[-0.02, 1.02])
        fig.update_yaxes(title="Sensibilidad", range=[-0.02, 1.05])
        mostrar(fig)

    c3, c4 = st.columns(2)
    with c3:
        cv = pd.DataFrame(meta["comparacion_cv"]).sort_values("roc_auc")
        elegido = meta["modelo"]["nombre"]
        colores = [AZUL if n == elegido else GRIS_CLARO for n in cv["modelo"]]
        fig = go.Figure(go.Bar(
            x=cv["roc_auc"], y=cv["modelo"], orientation="h", marker_color=colores, width=0.55,
            error_x=dict(type="data", array=cv["roc_auc_sd"], color=GRIS, thickness=1, width=4),
            customdata=cv["roc_auc_sd"],
            hovertemplate="%{y}<br>AUC promedio: %{x:.3f} ± %{customdata:.3f}<extra></extra>"))
        for fila in cv.itertuples():
            fig.add_annotation(x=fila.roc_auc + fila.roc_auc_sd + 0.008, y=fila.modelo, text=num(fila.roc_auc, 3),
                               xanchor="left", showarrow=False, font=dict(size=12, color=TINTA_2))
        estilo(fig, alto=360, leyenda=False, titulo="Comparación de 7 modelos (validación cruzada 5×5)")
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
            fig.add_vline(x=UMBRALES[c]["valor"], line_color=GRIS, line_width=1)
            fig.add_annotation(x=UMBRALES[c]["valor"], y=0.04, xanchor="left", xshift=4, showarrow=False,
                               text=f"{UMBRALES[c]['nombre']} {num(UMBRALES[c]['valor'], 2)}",
                               font=dict(size=11, color=TINTA_2))
        estilo(fig, alto=360, titulo="Sensibilidad y especificidad según el umbral (validación cruzada)")
        fig.update_xaxes(title="Umbral de decisión", range=[0.05, 0.95])
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
    fig.add_vline(x=0, line_color=GRIS_CLARO, line_width=1)
    estilo(fig, alto=430, leyenda=False, titulo="Odds ratios del modelo (10 mayores)")
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
                    f"En la prueba salió {num(prueba['auc'], 3)}: con solo {total} pacientes la cifra varía bastante "
                    f"(intervalo de confianza 95 %: {num(prueba['ic95']['auc'][0], 3)} a {num(prueba['ic95']['auc'][1], 3)}).")

# ---------------------------------------------------------------------------
# Pestaña 4: evaluación por lote
# ---------------------------------------------------------------------------
with tab_lote:
    st.subheader("Evaluar varios pacientes con un archivo CSV")
    st.markdown(
        '<p class="texto-suave">El archivo debe tener una fila por paciente y las 13 columnas del dataset con los '
        "códigos originales de UCI. Descargue la plantilla para ver el formato.</p>",
        unsafe_allow_html=True,
    )
    plantilla = pd.read_csv(RUTA_PLANTILLA)
    c1, c2 = st.columns([1, 2])
    c1.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_pacientes.csv", "text/csv", width="stretch")
    usar_plantilla = c1.toggle("Probar con la plantilla de ejemplo")
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
            st.error(f"No se pudo leer el archivo: {e}")
    elif usar_plantilla:
        lote = plantilla.copy()

    if lote is not None:
        faltantes = [v for v in VARS if v not in lote.columns]
        if faltantes:
            st.error("Faltan columnas: " + ", ".join(faltantes))
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
                                   "resultados_cardioriesgo.csv", "text/csv")

# ---------------------------------------------------------------------------
# Pestaña 5: acerca del proyecto
# ---------------------------------------------------------------------------
with tab_acerca:
    p = meta["proyecto"]
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.subheader("Sobre el proyecto")
        st.markdown(f"**Objetivo.** {p['objetivo']}")
        st.markdown(f"**Pregunta orientadora.** {p['pregunta']}")
        st.markdown(
            f"**Datos.** {meta['dataset']['nombre']}: {meta['dataset']['n_pacientes']} pacientes de la Cleveland Clinic "
            f"y 13 variables clínicas. La etiqueta viene de la angiografía coronaria "
            f"({pct(meta['dataset']['prevalencia'])} de los pacientes con enfermedad)."
        )
        st.markdown("**Ruta metodológica**")
        pasos = ["Exploración y EDA", "Calidad y limpieza", "Modelado clásico (7 modelos)",
                 "Evaluación y umbral", "Despliegue en Streamlit"]
        st.markdown(" → ".join(f"**{i}.** {p}" for i, p in enumerate(pasos, start=1)))
        st.markdown("**Referencias**")
        for r in p["referencias"]:
            st.markdown(f"- {r}")
    with c2:
        st.subheader("Grupo 13")
        st.markdown("\n".join(f"- {n}" for n in p["integrantes"]))
        st.markdown(f"Especialización en Inteligencia Artificial · CUN  \nAprendizaje Automático · {p['docente']}")
        enlaces = p.get("enlaces", {})
        st.markdown("**Enlaces**")
        for nombre, url in [("Aplicación", enlaces.get("app")), ("Video de sustentación", enlaces.get("video")),
                            ("Repositorio", enlaces.get("repositorio"))]:
            st.markdown(f"- {nombre}: {url if url else 'pendiente'}")
        st.markdown(
            '<div class="aviso">Proyecto académico. El modelo no está validado para uso clínico.</div>',
            unsafe_allow_html=True,
        )
