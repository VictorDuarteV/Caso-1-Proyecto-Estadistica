"""Dashboard del Proyecto Final de Estadística Aplicada a la Investigación.

Caso 1: FastTrack Logistics. Optimización de rutas y eficiencia de combustible.
Todos los resultados se calculan al abrir la app a partir de datos_logistica_transporte.csv.

Ejecución local:  streamlit run app.py
"""
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats
from statsmodels.stats.libqsturng import qsturng

st.set_page_config(page_title='FastTrack Logistics · Consumo de combustible', layout='wide')

BASE = Path(__file__).parent
ALPHA = 0.05

# ---------------------------------------------------------------- colores
FONDO, SUPERFICIE, BORDE = '#F5F3EE', '#FFFFFF', '#E2DDD3'
TEXTO, TEXTO_SEC = '#1C2B33', '#56656D'
PRIMARIO = '#0E4F5C'
RUTAS = ['Autopista', 'Montaña', 'Urbana']
COLOR_RUTA = {'Autopista': '#1F6F8B', 'Montaña': '#C4572F', 'Urbana': '#D49A3A'}
Y, X1, X2 = 'Consumo_Litros', 'Peso_Carga_Ton', 'Velocidad_Promedio_kmh'
NOMBRE = {Y: 'Consumo (L)', X1: 'Peso de carga (t)', X2: 'Velocidad promedio (km/h)'}

st.markdown('<style>' + (BASE / 'estilo.css').read_text(encoding='utf-8') + """
.cifras.tres-c {grid-template-columns: repeat(3, minmax(0, 1fr));}
.ecuacion {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 22px 28px;
           font-size: 20px; font-weight: 600; color: #0E4F5C; text-align: center; margin: 6px 0 18px;}
.st-key-entrada {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 24px 28px;}
.pregunta {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 24px 28px; margin-bottom: 16px;}
div[data-testid="stMarkdownContainer"] .pregunta h3 {font-size: 19px; font-weight: 600; margin: 0 0 8px; padding: 0; color: #1C2B33;}
div[data-testid="stMarkdownContainer"] .pregunta .resp {font-size: 15px; font-weight: 600; color: #0E4F5C; margin: 0 0 6px;}
div[data-testid="stMarkdownContainer"] .pregunta p {margin: 0; color: #36454D; line-height: 1.6;}
@media (max-width: 900px) {.cifras, .tres, .dos-columnas, .dos-tarjetas, .campos {grid-template-columns: 1fr;}}
</style>""", unsafe_allow_html=True)


def html(texto):
    """Muestra HTML. Se quitan la sangría y las líneas vacías para que Markdown no lo trate como código."""
    st.markdown('\n'.join(linea.strip() for linea in texto.splitlines() if linea.strip()), unsafe_allow_html=True)


def fmt_p(p):
    return f'{p:.4f}' if p >= 0.0001 else f'{p:.2e}'


def tabla_html(df, decimales=4, indice=True, clase='tabla compacta', resaltar=None):
    """Convierte un DataFrame en una tabla con el estilo de la app."""
    cab = (['<th></th>'] if indice else []) + [f'<th class="der">{c}</th>' for c in df.columns]
    filas = ''
    for idx, fila in df.iterrows():
        celdas = f'<td class="fuerte">{idx}</td>' if indice else ''
        for c, v in fila.items():
            if isinstance(v, (float, np.floating)):
                v = fmt_p(v) if c in ('p', 'Valor p', 'p ajustado') else f'{v:,.{decimales}f}'
            clase_td = 'der mejor' if resaltar and resaltar(idx, c) else 'der'
            celdas += f'<td class="{clase_td}">{v}</td>'
        filas += f'<tr>{celdas}</tr>'
    return f'<div class="marco"><table class="{clase}"><thead><tr>{"".join(cab)}</tr></thead><tbody>{filas}</tbody></table></div>'


def grafica(chart, alto=300):
    st.altair_chart(chart.properties(height=alto, background=SUPERFICIE)
                    .configure(font='Montserrat').configure_view(strokeWidth=0)
                    .configure_axis(labelColor=TEXTO_SEC, titleColor=TEXTO, gridColor=BORDE, domainColor='#C9C2B4')
                    .configure_legend(labelColor=TEXTO, titleColor=TEXTO_SEC, orient='top'),
                    width='stretch', theme=None)


ESCALA_RUTA = alt.Scale(domain=RUTAS, range=[COLOR_RUTA[r] for r in RUTAS])


# ---------------------------------------------------------------- datos y cálculos
@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / 'datos_logistica_transporte.csv')
    df['Tipo_Ruta'] = df['Tipo_Ruta'].str.replace('Monta¤a', 'Montaña', regex=False)
    return df


def anova(df, col):
    grupos = [df.loc[df.Tipo_Ruta == r, col] for r in RUTAS]
    n = len(df); k = len(grupos)
    C = df[col].sum() ** 2 / n
    sst = (df[col] ** 2).sum() - C
    ssa = sum(g.sum() ** 2 / len(g) for g in grupos) - C
    sse = sst - ssa
    msa, mse = ssa / (k - 1), sse / (n - k)
    F = msa / mse
    return dict(SSA=ssa, SSE=sse, SST=sst, MSA=msa, MSE=mse, F=F, p=stats.f.sf(F, k - 1, n - k),
                Fc=stats.f.ppf(1 - ALPHA, k - 1, n - k), eta2=ssa / sst,
                levene=stats.levene(*grupos).pvalue)


def tukey(df, col, mse):
    n = len(df)
    q = qsturng(1 - ALPHA, 3, n - 3)
    medias = df.groupby('Tipo_Ruta')[col].mean(); ns = df.groupby('Tipo_Ruta')[col].count()
    filas = []
    for a, b in [('Montaña', 'Autopista'), ('Montaña', 'Urbana'), ('Urbana', 'Autopista')]:
        d = medias[a] - medias[b]
        hsd = q * np.sqrt(mse / 2 * (1 / ns[a] + 1 / ns[b]))
        p = stats.studentized_range.sf(abs(d) / np.sqrt(mse / 2 * (1 / ns[a] + 1 / ns[b])), 3, n - 3)
        filas.append({'Par': f'{a} – {b}', 'Diferencia (L)': d, 'HSD': hsd,
                      'IC 95 %': f'[{d - hsd:.2f}; {d + hsd:.2f}]', 'p ajustado': p,
                      '¿Significativa?': 'Sí' if abs(d) > hsd else 'No'})
    return q, pd.DataFrame(filas).set_index('Par')


@st.cache_resource
def modelos(df):
    base = smf.ols(f'{Y} ~ {X1} + {X2}', data=df).fit()
    con_ruta = smf.ols(f'{Y} ~ {X1} + {X2} + C(Tipo_Ruta, Treatment("Autopista"))', data=df).fit()
    solo_ruta = smf.ols(f'{Y} ~ {X1} + C(Tipo_Ruta, Treatment("Autopista"))', data=df).fit()
    interaccion = smf.ols(f'{Y} ~ {X1} * C(Tipo_Ruta, Treatment("Autopista"))', data=df).fit()
    return base, con_ruta, solo_ruta, interaccion


df = cargar_datos()
N = len(df)
an = {c: anova(df, c) for c in (Y, X1, X2)}
q_tk, tabla_tukey = tukey(df, Y, an[Y]['MSE'])
m_base, m_ruta, m_solo_ruta, m_inter = modelos(df)
b0, b1, b2 = m_base.params
pendientes = {r: stats.linregress(df.loc[df.Tipo_Ruta == r, X1], df.loc[df.Tipo_Ruta == r, Y]) for r in RUTAS}
p_inter = sm.stats.anova_lm(m_solo_ruta, m_inter)['Pr(>F)'].iloc[1]
F_inter = sm.stats.anova_lm(m_solo_ruta, m_inter)['F'].iloc[1]
sd = df[[X1, X2, Y]].std()
beta_std = {X1: b1 * sd[X1] / sd[Y], X2: b2 * sd[X2] / sd[Y]}
medias_y = df.groupby('Tipo_Ruta')[Y].mean()
df['Ajustado'] = m_base.fittedvalues
df['Residual'] = m_base.resid


# ---------------------------------------------------------------- encabezado y navegación
SECCIONES = ['Proyecto', 'Datos', 'Exploración', 'ANOVA', 'Correlación', 'Regresión', 'Predicción', 'Decisiones']
with st.container(key='encabezado'):
    c_marca, c_nav = st.columns([1, 3.2], vertical_alignment='center')
    with c_marca:
        html('<div class="marca"><span class="logo" aria-hidden="true">F</span>FastTrack · Diésel</div>')
    with c_nav:
        seccion = st.radio('Secciones', SECCIONES, horizontal=True, label_visibility='collapsed', key='seccion')


# ================================================================ 1. PROYECTO
if seccion == 'Proyecto':
    html("""
    <section class="tarjeta">
      <span class="etiqueta">Proyecto Final · Estadística Aplicada a la Investigación · Caso 1</span>
      <h1>FastTrack Logistics: optimización de rutas y eficiencia de combustible</h1>
      <div class="campos">
        <div class="campo"><span>Alumno</span><span>Victor Francisco Duarte Villalobos</span></div>
        <div class="campo"><span>Matrícula</span><span>403230</span></div>
        <div class="campo"><span>Materia</span><span>Estadística Aplicada a la Investigación</span></div>
        <div class="campo"><span>Institución</span><span>Universidad Autónoma de Chihuahua</span></div>
        <div class="campo"><span>Programa</span><span>Maestría en Ingeniería en Computación</span></div>
        <div class="campo"><span>Nivel de significancia</span><span>α = 0.05</span></div>
      </div>
    </section>
    """)
    html("""
    <div class="dos-columnas" style="margin-top: 40px">
      <div>
        <h2 class="seccion">Problema</h2>
        <p class="texto">FastTrack Logistics es una empresa de transporte de carga con entregas Just-in-Time. En el
        último trimestre sus costos operativos subieron 18 %, impulsados casi por completo por el gasto en diésel.
        La empresa quiere saber qué factores influyen más en el consumo de combustible de su flota.</p>
      </div>
      <div>
        <h2 class="seccion">Objetivos</h2>
        <ul class="lista">
          <li>Determinar si el consumo difiere según el tipo de ruta (Urbana, Autopista, Montaña).</li>
          <li>Identificar la relación entre velocidad, peso y consumo.</li>
          <li>Construir un modelo predictivo del consumo.</li>
        </ul>
      </div>
    </div>
    """)
    html(f"""
    <h2 class="seccion" style="margin-top: 40px">Resultados principales</h2>
    <div class="cifras">
      <div class="cifra"><span class="rotulo">Viajes analizados</span><span class="valor">{N}</span>
        <span class="detalle">{df[Y].sum():,.1f} L en total</span></div>
      <div class="cifra"><span class="rotulo"><i class="punto" style="background:{COLOR_RUTA['Montaña']}"></i>Ruta con mayor consumo</span>
        <span class="valor">Montaña</span><span class="detalle">{medias_y['Montaña']:.2f} L por viaje</span></div>
      <div class="cifra"><span class="rotulo">Variable más relacionada</span><span class="valor">Peso</span>
        <span class="detalle">r = {df[X1].corr(df[Y]):.4f}</span></div>
      <div class="cifra"><span class="rotulo">R² del modelo</span><span class="valor">{m_base.rsquared:.4f}</span>
        <span class="detalle">Consumo ~ Peso + Velocidad</span></div>
    </div>
    """)
    html("""
    <h2 class="seccion" style="margin-top: 48px">Metodología</h2>
    <div class="lista-modelos">
      <article><span class="num">01</span><div><h3>Análisis exploratorio y ANOVA</h3>
        <div class="sub">Estadística descriptiva · ANOVA de un factor · Levene · Tukey-Kramer</div>
        <p>Se comparan las medias de consumo, peso y velocidad entre los tres tipos de ruta y se identifica qué pares difieren.</p></div></article>
      <article><span class="num">02</span><div><h3>Correlación de Pearson</h3>
        <div class="sub">Matriz de correlación · prueba t de significancia</div>
        <p>Mide la fuerza y dirección de la relación lineal entre peso, velocidad y consumo.</p></div></article>
      <article><span class="num">03</span><div><h3>Regresión lineal múltiple</h3>
        <div class="sub">Consumo = β₀ + β₁·Peso + β₂·Velocidad + ε · mínimos cuadrados ordinarios</div>
        <p>Cuantifica el efecto de cada variable, valida la significancia de los coeficientes y permite pronosticar el consumo.
        Se complementa con un modelo que incluye el tipo de ruta.</p></div></article>
    </div>
    """)


# ================================================================ 2. DATOS
elif seccion == 'Datos':
    html('<h1 class="titulo">Datos</h1>'
         '<p class="texto">Base <code>datos_logistica_transporte</code>: 50 viajes transcritos del PDF original. '
         'La categoría "Monta¤a" se corrigió a "Montaña". No hay valores faltantes.</p>')
    conteo = df.Tipo_Ruta.value_counts()
    html('<div class="cifras">' + f'<div class="cifra"><span class="rotulo">Total de viajes</span><span class="valor">{N}</span>'
         '<span class="detalle">4 variables + identificador</span></div>' +
         ''.join(f'<div class="cifra"><span class="rotulo"><i class="punto" style="background:{COLOR_RUTA[r]}"></i>{r}</span>'
                 f'<span class="valor">{conteo[r]}</span><span class="detalle">{conteo[r] / N * 100:.0f} % de los viajes</span></div>'
                 for r in RUTAS) + '</div>')
    html("""
    <h2 class="seccion">Variables</h2>
    <div class="marco"><table class="tabla compacta"><thead><tr><th>Variable</th><th>Tipo</th><th>Descripción</th></tr></thead><tbody>
    <tr><td class="fuerte">Tipo_Ruta</td><td>Categórica</td><td>Urbana, Autopista o Montaña (factor del ANOVA)</td></tr>
    <tr><td class="fuerte">Peso_Carga_Ton</td><td>Numérica</td><td>Peso de la carga en toneladas (X₁)</td></tr>
    <tr><td class="fuerte">Velocidad_Promedio_kmh</td><td>Numérica</td><td>Velocidad promedio del viaje en km/h (X₂)</td></tr>
    <tr><td class="fuerte">Consumo_Litros</td><td>Numérica</td><td>Diésel consumido en litros (Y, variable objetivo)</td></tr>
    </tbody></table></div>
    """)
    html('<h2 class="seccion">Tabla de viajes</h2>')
    filtro = st.multiselect('Filtrar por tipo de ruta', RUTAS, default=RUTAS)
    vista = df.loc[df.Tipo_Ruta.isin(filtro), ['ID_Viaje', 'Tipo_Ruta', X1, X2, Y]]
    st.dataframe(vista, hide_index=True, width='stretch', height=420)
    st.download_button('Descargar CSV', vista.to_csv(index=False).encode('utf-8'),
                       'datos_logistica_transporte.csv', 'text/csv')


# ================================================================ 3. EXPLORACIÓN
elif seccion == 'Exploración':
    html('<h1 class="titulo">Análisis exploratorio</h1>'
         '<p class="texto">Medidas de tendencia central y dispersión por tipo de ruta. La varianza es muestral (divide entre n − 1).</p>')
    var = st.radio('Variable', [Y, X1, X2], format_func=NOMBRE.get, horizontal=True)

    def moda(s):
        vc = s.value_counts()
        return f'{vc.index[0]:.2f}' if vc.iloc[0] > 1 else '—'

    filas = []
    for r in RUTAS + ['Total']:
        s = df[var] if r == 'Total' else df.loc[df.Tipo_Ruta == r, var]
        filas.append({'Ruta': r, 'n': len(s), 'Media': s.mean(), 'Mediana': s.median(), 'Moda': moda(s),
                      'Varianza': s.var(), 'Desv. est.': s.std(), 'CV %': s.std() / s.mean() * 100,
                      'Mín.': s.min(), 'Máx.': s.max(), 'Rango': s.max() - s.min()})
    desc = pd.DataFrame(filas).set_index('Ruta')
    desc['n'] = desc['n'].astype(str)
    html(f'<div class="fila-titulo"><h2 class="seccion">{NOMBRE[var]}</h2><span>Estadísticos por ruta</span></div>'
         + tabla_html(desc, 2))

    c1, c2 = st.columns(2, gap='large')
    with c1:
        with st.container(key='grafica'):
            html('<b style="font-size:18px">Distribución por ruta</b>')
            caja = alt.Chart(df).mark_boxplot(size=50, extent=1.5).encode(
                x=alt.X('Tipo_Ruta:N', sort=RUTAS, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f'{var}:Q', title=NOMBRE[var], scale=alt.Scale(zero=False)),
                color=alt.Color('Tipo_Ruta:N', scale=ESCALA_RUTA, legend=None))
            grafica(caja)
    with c2:
        with st.container(key='grafica2'):
            html('<b style="font-size:18px">Media por ruta</b>')
            medias = df.groupby('Tipo_Ruta', as_index=False)[var].mean()
            barras = alt.Chart(medias).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=60).encode(
                x=alt.X('Tipo_Ruta:N', sort=RUTAS, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f'{var}:Q', title=f'Media de {NOMBRE[var].lower()}'),
                color=alt.Color('Tipo_Ruta:N', scale=ESCALA_RUTA, legend=None),
                tooltip=['Tipo_Ruta', alt.Tooltip(f'{var}:Q', format='.2f')])
            texto = barras.mark_text(dy=-10, color=TEXTO, fontWeight=600).encode(text=alt.Text(f'{var}:Q', format='.2f'))
            grafica(barras + texto)

    with st.container(key='grafica3'):
        html('<b style="font-size:18px">Histograma</b>')
        hist = alt.Chart(df).mark_bar(opacity=0.85).encode(
            x=alt.X(f'{var}:Q', bin=alt.Bin(maxbins=12), title=NOMBRE[var]),
            y=alt.Y('count():Q', title='Viajes', stack=True),
            color=alt.Color('Tipo_Ruta:N', scale=ESCALA_RUTA, title='Ruta'))
        grafica(hist, 260)
    html('<p class="nota">Montaña concentra las cargas más pesadas y el mayor consumo, con la mayor dispersión. '
         'Las velocidades están separadas por ruta: Autopista opera entre 80 y 105 km/h.</p>')


# ================================================================ 4. ANOVA
elif seccion == 'ANOVA':
    html('<h1 class="titulo">ANOVA de un factor</h1>'
         '<p class="texto">H₀: μ<sub>Autopista</sub> = μ<sub>Montaña</sub> = μ<sub>Urbana</sub> · H₁: al menos una media difiere. '
         'Factor: tipo de ruta (k = 3, N = 50).</p>')
    var = st.radio('Variable dependiente', [Y, X1, X2], format_func=NOMBRE.get, horizontal=True, key='var_anova')
    a = an[var]
    decision = 'Se rechaza H₀' if a['p'] < ALPHA else 'No se rechaza H₀'
    html(f"""
    <div class="cifras">
      <div class="cifra"><span class="rotulo">Estadístico F</span><span class="valor">{a['F']:.4f}</span>
        <span class="detalle">F crítico (0.05; 2, 47) = {a['Fc']:.4f}</span></div>
      <div class="cifra"><span class="rotulo">Valor p</span><span class="valor">{fmt_p(a['p'])}</span>
        <span class="detalle">{decision}</span></div>
      <div class="cifra"><span class="rotulo">η² (tamaño del efecto)</span><span class="valor">{a['eta2']:.4f}</span>
        <span class="detalle">{a['eta2'] * 100:.1f} % de la variabilidad</span></div>
      <div class="cifra"><span class="rotulo">Levene (p)</span><span class="valor">{a['levene']:.4f}</span>
        <span class="detalle">{'Varianzas homogéneas' if a['levene'] > ALPHA else 'Varianzas distintas'}</span></div>
    </div>
    """)
    tabla = pd.DataFrame({'SS': [a['SSA'], a['SSE'], a['SST']], 'gl': ['2', '47', '49'],
                          'MS': [a['MSA'], a['MSE'], np.nan], 'F': [a['F'], np.nan, np.nan],
                          'Valor p': [a['p'], np.nan, np.nan]}, index=['Entre rutas', 'Dentro (error)', 'Total'])
    html('<div class="fila-titulo"><h2 class="seccion">Tabla ANOVA</h2>'
         f'<span>{NOMBRE[var]}</span></div>' + tabla_html(tabla).replace('>nan<', '>—<'))

    resumen = pd.DataFrame({NOMBRE[c]: {'F': an[c]['F'], 'Valor p': an[c]['p'], 'η²': an[c]['eta2'],
                                        'Levene (p)': an[c]['levene'],
                                        'Decisión': 'Rechazar H₀' if an[c]['p'] < ALPHA else 'No rechazar'}
                            for c in (Y, X1, X2)}).T
    html('<div class="fila-titulo"><h2 class="seccion">Resumen de los tres ANOVA</h2></div>' + tabla_html(resumen))

    if var == Y:
        html(f'<div class="fila-titulo"><h2 class="seccion">Prueba post-hoc de Tukey-Kramer</h2>'
             f'<span>q(0.05; 3, 47) = {q_tk:.4f}</span></div>' + tabla_html(tabla_tukey))
        html('<p class="nota">Montaña difiere de Autopista y de Urbana. Urbana y Autopista no difieren entre sí, por lo '
             'que forman un solo grupo homogéneo.</p>')


# ================================================================ 5. CORRELACIÓN
elif seccion == 'Correlación':
    html('<h1 class="titulo">Correlación de Pearson</h1>'
         '<p class="texto">r mide la fuerza y dirección de la relación lineal entre dos variables (de −1 a 1). '
         'Se prueba H₀: ρ = 0 con t = r√(n−2)/√(1−r²) y 48 grados de libertad.</p>')
    cols = [X1, X2, Y]
    corr = df[cols].corr()
    largo = corr.rename(index=NOMBRE, columns=NOMBRE).reset_index(names='A').melt('A', var_name='B', value_name='r')
    c1, c2 = st.columns([1.1, 1], gap='large')
    with c1:
        with st.container(key='grafica'):
            html('<b style="font-size:18px">Matriz de correlación</b>')
            orden = [NOMBRE[c] for c in cols]
            base = alt.Chart(largo).encode(x=alt.X('B:N', sort=orden, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
                                           y=alt.Y('A:N', sort=orden, title=None))
            calor = base.mark_rect().encode(color=alt.Color('r:Q', scale=alt.Scale(domain=[-1, 1], range=['#C4572F', '#F5F3EE', '#0E4F5C']), legend=None))
            txt = base.mark_text(fontSize=16, fontWeight=600).encode(
                text=alt.Text('r:Q', format='.4f'),
                color=alt.condition('abs(datum.r) > 0.6', alt.value('#FFFFFF'), alt.value(TEXTO)))
            grafica(calor + txt, 300)
    with c2:
        filas = []
        for a_, b_ in combinations(cols, 2):
            r, p = stats.pearsonr(df[a_], df[b_])
            t = r * np.sqrt(N - 2) / np.sqrt(1 - r ** 2)
            filas.append({'Par': f'{NOMBRE[a_].split(" (")[0]} – {NOMBRE[b_].split(" (")[0]}', 'r': r, 't': t, 'Valor p': p,
                          '¿Signif.?': 'Sí' if p < ALPHA else 'No'})
        html('<div class="fila-titulo" style="margin-top:0"><h2 class="seccion">Significancia</h2>'
             f'<span>t crítico = {stats.t.ppf(1 - ALPHA / 2, N - 2):.4f}</span></div>'
             + tabla_html(pd.DataFrame(filas).set_index('Par')))
        html(f'<p class="nota">El peso es la variable con la relación más fuerte con el consumo (r = {corr.loc[X1, Y]:.4f}); '
             f'por sí solo explica {corr.loc[X1, Y] ** 2 * 100:.1f} % de su variabilidad. La velocidad casi no se relaciona '
             'con el consumo, y peso y velocidad casi no se relacionan entre sí (no hay multicolinealidad).</p>')
    eje = st.radio('Eje horizontal', [X1, X2], format_func=NOMBRE.get, horizontal=True)
    with st.container(key='grafica2'):
        html(f'<b style="font-size:18px">Consumo frente a {NOMBRE[eje].split(" (")[0].lower()}</b>')
        puntos = alt.Chart(df).mark_circle(size=80, opacity=0.85).encode(
            x=alt.X(f'{eje}:Q', title=NOMBRE[eje], scale=alt.Scale(zero=False)),
            y=alt.Y(f'{Y}:Q', title=NOMBRE[Y], scale=alt.Scale(zero=False)),
            color=alt.Color('Tipo_Ruta:N', scale=ESCALA_RUTA, title='Ruta'),
            tooltip=['ID_Viaje', 'Tipo_Ruta', X1, X2, Y])
        recta = alt.Chart(df).transform_regression(eje, Y).mark_line(color=TEXTO, strokeDash=[6, 4]).encode(x=f'{eje}:Q', y=f'{Y}:Q')
        grafica(puntos + recta, 340)
    html('<p class="nota">La línea punteada es la recta de regresión simple. Con la velocidad, los puntos se agrupan por '
         'ruta: la relación aparente mezcla el efecto del terreno con el de la velocidad.</p>')


# ================================================================ 6. REGRESIÓN
elif seccion == 'Regresión':
    html('<h1 class="titulo">Regresión lineal múltiple</h1>'
         '<p class="texto">Modelo ajustado por mínimos cuadrados ordinarios con los 50 viajes.</p>')
    html(f'<div class="ecuacion">Consumo = {b0:.4f} + {b1:.4f}·Peso − {abs(b2):.4f}·Velocidad</div>')
    rmse = np.sqrt(m_base.ssr / N)
    mae = m_base.resid.abs().mean()
    html(f"""
    <div class="cifras">
      <div class="cifra"><span class="rotulo">R²</span><span class="valor">{m_base.rsquared:.4f}</span>
        <span class="detalle">{m_base.rsquared * 100:.1f} % de la variabilidad explicada</span></div>
      <div class="cifra"><span class="rotulo">R² ajustada</span><span class="valor">{m_base.rsquared_adj:.4f}</span>
        <span class="detalle">Penaliza el número de predictores</span></div>
      <div class="cifra"><span class="rotulo">Prueba F global</span><span class="valor">{m_base.fvalue:.2f}</span>
        <span class="detalle">p = {fmt_p(m_base.f_pvalue)}</span></div>
      <div class="cifra"><span class="rotulo">Error</span><span class="valor">{rmse:.2f} L</span>
        <span class="detalle">RMSE · MAE = {mae:.2f} L · MSE = {m_base.ssr / N:.2f}</span></div>
    </div>
    """)
    coef = pd.DataFrame({'β': m_base.params, 'Error est.': m_base.bse, 't': m_base.tvalues, 'Valor p': m_base.pvalues,
                         'IC 95 % inf.': m_base.conf_int()[0], 'IC 95 % sup.': m_base.conf_int()[1]})
    coef.index = ['Intercepto', 'Peso (t)', 'Velocidad (km/h)']
    coef['β estandarizado'] = [np.nan, beta_std[X1], beta_std[X2]]
    html('<div class="fila-titulo"><h2 class="seccion">Coeficientes</h2>'
         '<span>t crítico (0.025; 47) = 2.0117</span></div>' + tabla_html(coef).replace('>nan<', '>—<'))
    html(f"""
    <div class="tres" style="margin-top:16px">
      <div class="cifra"><h3>β₁ = {b1:.4f}</h3><p>Con la velocidad constante, cada tonelada adicional suma {b1:.2f} L.</p></div>
      <div class="cifra"><h3>β₂ = {b2:.4f}</h3><p>Con el peso constante, cada km/h adicional se asocia con {abs(b2):.3f} L menos.
        No implica que acelerar ahorre combustible (ver abajo).</p></div>
      <div class="cifra"><h3>β* peso = {beta_std[X1]:.4f}</h3><p>El efecto estandarizado del peso es {abs(beta_std[X1] / beta_std[X2]):.1f}
        veces el de la velocidad ({beta_std[X2]:.4f}).</p></div>
    </div>
    """)

    c1, c2 = st.columns(2, gap='large')
    with c1:
        with st.container(key='grafica'):
            html('<b style="font-size:18px">Real vs. ajustado</b>')
            lim = [df[[Y, 'Ajustado']].min().min() - 3, df[[Y, 'Ajustado']].max().max() + 3]
            pts = alt.Chart(df).mark_circle(size=70).encode(
                x=alt.X('Ajustado:Q', title='Consumo ajustado (L)', scale=alt.Scale(domain=lim)),
                y=alt.Y(f'{Y}:Q', title='Consumo real (L)', scale=alt.Scale(domain=lim)),
                color=alt.Color('Tipo_Ruta:N', scale=ESCALA_RUTA, title='Ruta'), tooltip=['ID_Viaje', 'Tipo_Ruta', Y, 'Ajustado'])
            diag = alt.Chart(pd.DataFrame({'x': lim, 'y': lim})).mark_line(color=TEXTO, strokeDash=[6, 4]).encode(x='x', y='y')
            grafica(pts + diag)
    with c2:
        with st.container(key='grafica2'):
            html('<b style="font-size:18px">Residuales vs. ajustados</b>')
            res = alt.Chart(df).mark_circle(size=70).encode(
                x=alt.X('Ajustado:Q', title='Consumo ajustado (L)', scale=alt.Scale(zero=False)),
                y=alt.Y('Residual:Q', title='Residual (L)'),
                color=alt.Color('Tipo_Ruta:N', scale=ESCALA_RUTA, title='Ruta'), tooltip=['ID_Viaje', 'Tipo_Ruta', 'Residual'])
            cero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=TEXTO, strokeDash=[6, 4]).encode(y='y')
            grafica(res + cero)

    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    r12 = df[X1].corr(df[X2])
    diag_t = pd.DataFrame({
        'Estadístico': [stats.shapiro(m_base.resid).statistic, het_breuschpagan(m_base.resid, m_base.model.exog)[0],
                        durbin_watson(m_base.resid), 1 / (1 - r12 ** 2)],
        'Valor p': [stats.shapiro(m_base.resid).pvalue, het_breuschpagan(m_base.resid, m_base.model.exog)[1], np.nan, np.nan]},
        index=['Normalidad (Shapiro-Wilk)', 'Homocedasticidad (Breusch-Pagan)', 'Independencia (Durbin-Watson)', 'Multicolinealidad (VIF)'])
    res_ruta = df.groupby('Tipo_Ruta')['Residual'].agg(['mean', 'min', 'max']).reindex(RUTAS)
    res_ruta.columns = ['Residual medio', 'Mínimo', 'Máximo']
    c1, c2 = st.columns(2, gap='large')
    with c1:
        html('<div class="fila-titulo"><h2 class="seccion">Diagnóstico de residuales</h2></div>' + tabla_html(diag_t).replace('>nan<', '>—<'))
    with c2:
        html('<div class="fila-titulo"><h2 class="seccion">Residuales por ruta</h2></div>' + tabla_html(res_ruta))
    html('<p class="nota">Los residuales de Montaña son todos positivos: el modelo base subestima el consumo en montaña '
         'porque no incluye el tipo de ruta.</p>')

    html('<div class="fila-titulo"><h2 class="seccion">Modelos con tipo de ruta</h2>'
         '<span>Variables indicadoras; Autopista como referencia</span></div>')
    comp = pd.DataFrame({
        'R²': [m_base.rsquared, m_ruta.rsquared, m_inter.rsquared],
        'R² ajustada': [m_base.rsquared_adj, m_ruta.rsquared_adj, m_inter.rsquared_adj],
        'RMSE (L)': [np.sqrt(m.ssr / N) for m in (m_base, m_ruta, m_inter)],
        'AIC': [m_base.aic, m_ruta.aic, m_inter.aic]},
        index=['Base: Peso + Velocidad', 'Peso + Velocidad + Ruta', 'Peso × Ruta (interacción)'])
    html(tabla_html(comp, resaltar=lambda i, c: (c in ('R²', 'R² ajustada') and comp[c].max() == comp.loc[i, c])
                    or (c in ('RMSE (L)', 'AIC') and comp[c].min() == comp.loc[i, c])))
    pend = pd.DataFrame({'Pendiente (L/t)': [pendientes[r].slope for r in RUTAS],
                         'Intercepto': [pendientes[r].intercept for r in RUTAS],
                         'r': [pendientes[r].rvalue for r in RUTAS]}, index=RUTAS)
    html(f'<p class="nota">Al incluir la ruta, la velocidad deja de ser significativa (p = {m_ruta.pvalues[X2]:.4f}): su '
         f'coeficiente negativo en el modelo base refleja que las velocidades altas corresponden a la autopista. '
         f'La interacción Peso × Ruta es significativa (F = {F_inter:.2f}, p = {fmt_p(p_inter)}).</p>')
    html('<div class="fila-titulo"><h2 class="seccion">Efecto del peso dentro de cada ruta</h2></div>' + tabla_html(pend))


# ================================================================ 7. PREDICCIÓN
elif seccion == 'Predicción':
    html('<h1 class="titulo">Predicción del consumo</h1>'
         '<p class="texto">Ajusta el peso, la velocidad y la ruta. El modelo base es la ecuación solicitada en el caso; '
         'el modelo con ruta y la recta de la ruta elegida sirven como comparación.</p>')
    with st.container(key='entrada'):
        e1, e2, e3 = st.columns(3, gap='large')
        peso = e1.slider('Peso de carga (t)', 4.0, 28.0, 15.0, 0.5)
        vel = e2.slider('Velocidad promedio (km/h)', 20.0, 105.0, 80.0, 1.0)
        ruta = e3.selectbox('Tipo de ruta', RUTAS, index=0)
    nuevo = pd.DataFrame({X1: [peso], X2: [vel], 'Tipo_Ruta': [ruta]})
    pred = m_base.get_prediction(nuevo).summary_frame(alpha=ALPHA).iloc[0]
    p_ruta = m_ruta.get_prediction(nuevo).summary_frame(alpha=ALPHA).iloc[0]
    p_recta = pendientes[ruta].intercept + pendientes[ruta].slope * peso
    sub = df[df.Tipo_Ruta == ruta]
    fuera = not (sub[X2].min() <= vel <= sub[X2].max() and sub[X1].min() <= peso <= sub[X1].max())
    html(f"""
    <div class="cifras tres-c">
      <div class="cifra"><span class="rotulo">Modelo base (ecuación del caso)</span><span class="valor">{pred['mean']:.2f} L</span>
        <span class="detalle">IC 95 %: [{pred['mean_ci_lower']:.2f}; {pred['mean_ci_upper']:.2f}] · IP 95 %: [{pred['obs_ci_lower']:.2f}; {pred['obs_ci_upper']:.2f}]</span></div>
      <div class="cifra"><span class="rotulo"><i class="punto" style="background:{COLOR_RUTA[ruta]}"></i>Modelo con ruta</span><span class="valor">{p_ruta['mean']:.2f} L</span>
        <span class="detalle">IC 95 %: [{p_ruta['mean_ci_lower']:.2f}; {p_ruta['mean_ci_upper']:.2f}]</span></div>
      <div class="cifra"><span class="rotulo"><i class="punto" style="background:{COLOR_RUTA[ruta]}"></i>Recta de {ruta}</span><span class="valor">{p_recta:.2f} L</span>
        <span class="detalle">{pendientes[ruta].slope:.2f} L por tonelada en esta ruta</span></div>
    </div>
    """)
    html(f'<p class="nota">Sustitución: Consumo = {b0:.4f} + {b1:.4f}({peso:g}) − {abs(b2):.4f}({vel:g}) = {pred["mean"]:.2f} L. '
         'IC: consumo promedio de muchos camiones con estas características. IP: consumo de un solo viaje.</p>')
    if fuera:
        html(f'<div class="aviso alerta" role="status" style="margin-top:14px"><span>Los valores elegidos están fuera del rango '
             f'observado para {ruta} (peso {sub[X1].min():.1f}–{sub[X1].max():.1f} t, velocidad {sub[X2].min():.1f}–{sub[X2].max():.1f} km/h). '
             'La predicción es una extrapolación.</span></div>')
    with st.container(key='grafica'):
        html(f'<b style="font-size:18px">Consumo estimado según el peso · {ruta}</b>')
        rejilla = pd.DataFrame({X1: np.linspace(4, 28, 49)})
        rejilla[X2] = vel
        rejilla['Tipo_Ruta'] = ruta
        rejilla['Modelo base'] = m_base.predict(rejilla)
        rejilla['Modelo con ruta'] = m_ruta.predict(rejilla)
        largo = rejilla.melt(X1, ['Modelo base', 'Modelo con ruta'], var_name='Modelo', value_name='Consumo')
        lineas = alt.Chart(largo).mark_line(strokeWidth=3).encode(
            x=alt.X(f'{X1}:Q', title='Peso de carga (t)'), y=alt.Y('Consumo:Q', title='Consumo (L)', scale=alt.Scale(zero=False)),
            color=alt.Color('Modelo:N', scale=alt.Scale(range=[PRIMARIO, COLOR_RUTA[ruta]]), title=None))
        obs = alt.Chart(sub).mark_circle(size=60, opacity=0.6, color=COLOR_RUTA[ruta]).encode(x=f'{X1}:Q', y=f'{Y}:Q')
        punto = alt.Chart(pd.DataFrame({X1: [peso], 'Consumo': [pred['mean']]})).mark_point(
            size=220, filled=True, color=TEXTO, shape='diamond').encode(x=f'{X1}:Q', y='Consumo:Q')
        grafica(lineas + obs + punto, 320)
    html(f'<p class="nota">Los puntos son los viajes reales de {ruta}; el rombo es la predicción del modelo base. '
         f'Pronóstico solicitado en el caso (15 t a 80 km/h): {b0 + b1 * 15 + b2 * 80:.2f} L.</p>')


# ================================================================ 8. DECISIONES
elif seccion == 'Decisiones':
    html('<h1 class="titulo">Toma de decisiones</h1>'
         '<p class="texto">Respuestas a las preguntas del caso, basadas en los resultados del análisis.</p>')
    tm = pendientes['Montaña']; ta = pendientes['Autopista']
    html(f"""
    <div class="pregunta"><h3>1. ¿Debería la empresa evitar ciertas rutas para cargas pesadas?</h3>
      <p class="resp">Sí: las rutas de montaña.</p>
      <p>Cada tonelada adicional cuesta {tm.slope:.2f} L en montaña frente a {ta.slope:.2f} L en autopista
      (interacción Peso × Ruta, p = {fmt_p(p_inter)}). Con 25 t se estiman {tm.intercept + 25 * tm.slope:.1f} L en montaña
      y {ta.intercept + 25 * ta.slope:.1f} L en autopista. Las cargas pesadas deben asignarse a autopista cuando exista la alternativa.</p></div>
    <div class="pregunta"><h3>2. ¿En qué tipo de ruta se gasta más combustible y la diferencia es significativa?</h3>
      <p class="resp">En Montaña ({medias_y['Montaña']:.2f} L); la diferencia es significativa.</p>
      <p>ANOVA: F = {an[Y]['F']:.2f}, p = {fmt_p(an[Y]['p'])}. Según Tukey, Montaña
      difiere de Autopista ({medias_y['Autopista']:.2f} L) y de Urbana ({medias_y['Urbana']:.2f} L), que no difieren entre sí.</p></div>
    <div class="pregunta"><h3>3. Para reducir el consumo, ¿controlar la velocidad o reducir el peso?</h3>
      <p class="resp">Reducir el peso de carga.</p>
      <p>Su coeficiente estandarizado ({beta_std[X1]:.4f}) es {abs(beta_std[X1] / beta_std[X2]):.1f} veces el de la velocidad
      ({beta_std[X2]:.4f}), y cada tonelada menos ahorra en promedio {b1:.2f} L. El coeficiente negativo de la velocidad no
      implica que conducir más rápido ahorre diésel: deja de ser significativo al considerar el tipo de ruta (p = {m_ruta.pvalues[X2]:.4f}).</p></div>
    <div class="pregunta"><h3>4. Pronóstico para un camión con 15 toneladas a 80 km/h</h3>
      <p class="resp">{b0 + b1 * 15 + b2 * 80:.2f} litros.</p>
      <p>Consumo = {b0:.4f} + {b1:.4f}(15) − {abs(b2):.4f}(80). Puedes probar otros valores en la sección Predicción.</p></div>
    """)
    html("""
    <h2 class="seccion" style="margin-top:32px">Recomendaciones</h2>
    <div class="dos-tarjetas">
      <div class="cifra"><h3>Controlar el peso por viaje</h3><p>Es la palanca con mayor efecto sobre el consumo en todas las rutas.</p></div>
      <div class="cifra"><h3>Reasignar cargas pesadas</h3><p>Enviarlas por autopista y, si la montaña es inevitable, consolidar o dividir envíos.</p></div>
      <div class="cifra"><h3>Presupuestar por ruta</h3><p>Usar el modelo con tipo de ruta, ya que el modelo base subestima la montaña y sobrestima la autopista.</p></div>
      <div class="cifra"><h3>Registrar más variables</h3><p>Pendiente del terreno, distancia y tiempo en ralentí, para validar con una muestra mayor.</p></div>
    </div>
    """)
