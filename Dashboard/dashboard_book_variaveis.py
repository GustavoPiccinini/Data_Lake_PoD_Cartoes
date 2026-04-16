"""
Dashboard — Book de Variáveis
Projeto: Feature Store / Datalake
Para: Analistas, Cientistas de Dados, Gestores e Recrutadores
"""

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book de Variáveis · Feature Store",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# ESTILO
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .block-container { padding: 1.5rem 2rem 2rem 2rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] > div:first-child {
    background: #0d0f14;
    border-right: 1px solid #1e2433;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stMarkdown p {
    color: #a0aec0 !important;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
    color: #718096 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
    color: #e2e8f0 !important;
  }
  div[data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 0.75rem !important;
  }

  /* Tabs */
  div[data-baseweb="tab-list"] {
    gap: 4px;
    background: #0d0f14;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e2433;
  }
  button[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #718096 !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    background: #2d3748 !important;
    color: #e2e8f0 !important;
  }

  /* Headers */
  h1, h2, h3 { color: #e2e8f0 !important; }
  .stMarkdown h4 { color: #a0aec0 !important; font-size: 0.85rem; }

  /* DataFrames */
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Selectbox e sliders */
  div[data-baseweb="select"] { background: #1a1f2e !important; }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-family: 'Space Mono', monospace;
  }
  .badge-blue  { background: #1e3a5f; color: #63b3ed; border: 1px solid #2b4e7a; }
  .badge-green { background: #1a3a2a; color: #68d391; border: 1px solid #276749; }
  .badge-amber { background: #3a2e10; color: #f6ad55; border: 1px solid #6b4c12; }
  .badge-red   { background: #3a1a1a; color: #fc8181; border: 1px solid #742a2a; }

  /* Code-like numbers */
  .mono { font-family: 'Space Mono', monospace; font-size: 0.9rem; }

  /* Section divider */
  .section-header {
    border-left: 3px solid #4299e1;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
    color: #e2e8f0;
    font-weight: 600;
    font-size: 1rem;
  }
  
  /* Main background */
  .stApp { background: #0a0c10; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# MAPEAMENTO DE VARIÁVEIS
# ──────────────────────────────────────────────────────────────
# Estrutura descoberta nos dados:
# 8 métricas monetárias (blocks de 20 vars: 4 stats × 5 janelas)
# 4 métricas de contagem  (blocks de 20 vars: 4 stats × 5 janelas)
# Janelas: total, 3m, 6m, 9m, 12m
# Stats por janela: soma, média, mínimo, máximo

WINDOWS = {
    "12 meses": 4,   # índice 4 → offset 16 (vars _17 a _20)
    "9 meses":  3,   # índice 3 → offset 12
    "6 meses":  2,   # índice 2 → offset 8
    "3 meses":  1,   # índice 1 → offset 4
    "Total":    0,   # índice 0 → offset 0
}

# 8 blocos de métricas (posição inicial de cada bloco, nome inferido)
METRICAS = {
    "Métrica A — Valor Financeiro":     {"start": 1,   "type": "monetary"},
    "Métrica B — Transações":           {"start": 41,  "type": "monetary"},
    "Métrica C — Comportamento I":      {"start": 81,  "type": "monetary"},
    "Métrica D — Comportamento II":     {"start": 121, "type": "monetary"},
    "Contagem A — Frequência":          {"start": 21,  "type": "count", "vars": [21,22,23,24,25]},
    "Contagem B — Recência":            {"start": 61,  "type": "count", "vars": [61,62,63,64,65]},
    "Contagem C — Intensidade":         {"start": 101, "type": "count", "vars": [101,102,103,104,105]},
    "Contagem D — Engajamento":         {"start": 141, "type": "count", "vars": [141,142,143,144,145]},
}

def var_names(start, window_idx):
    """Retorna os 4 nomes de variáveis (soma, média, min, max) para um bloco e janela."""
    base = start + window_idx * 4
    return [f"VAR_{base}", f"VAR_{base+1}", f"VAR_{base+2}", f"VAR_{base+3}"]

def count_var(vars_list, window_idx):
    return f"VAR_{vars_list[window_idx]}"

# ──────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS (embutido — sem necessidade de upload)
# ──────────────────────────────────────────────────────────────
DATA_FILES = [
    "part-00000-cb0b020f-b207-4b41-ba42-a73cf33b33ee.c000.snappy.parquet",
    "part-00001-cb0b020f-b207-4b41-ba42-a73cf33b33ee.c000.snappy.parquet",
]

@st.cache_data(show_spinner="Carregando book de variáveis...")
def load_data():
    frames = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in DATA_FILES:
        path = os.path.join(script_dir, fname)
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    if not frames:
        # fallback: tenta carregar do diretório atual
        for fname in DATA_FILES:
            if os.path.exists(fname):
                frames.append(pd.read_parquet(fname))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # Converter colunas object para numeric
    for c in df.columns:
        if df[c].dtype == object and c != "id_cliente":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0;'>
      <div style='font-family: Space Mono, monospace; color: #4299e1; font-size: 1.1rem; font-weight: 700;'>
        📦 Feature Store
      </div>
      <div style='color: #4a5568; font-size: 0.75rem; margin-top: 4px;'>
        Book de Variáveis · Datalake
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if df is None:
        st.error("Arquivos parquet não encontrados. Coloque os arquivos no mesmo diretório do script.")
        st.stop()

    # Janela temporal
    janela_sel = st.selectbox(
        "🕐 Janela temporal",
        options=list(WINDOWS.keys()),
        index=0,
        help="Seleciona a janela de observação para análise"
    )
    w_idx = WINDOWS[janela_sel]

    st.markdown("---")
    st.markdown("#### 🔬 Filtros exploratórios")

    # Filtro por cobertura (null %)
    min_cobertura = st.slider(
        "Cobertura mínima dos clientes",
        min_value=0, max_value=100, value=50,
        format="%d%%",
        help="Filtra variáveis com pelo menos X% de preenchimento para a janela selecionada"
    )

    # Amostra para visualização
    n_clientes = st.number_input(
        "Amostra para distribuições",
        min_value=100, max_value=min(5000, len(df)),
        value=min(2000, len(df)),
        step=100,
        help="Quantidade de clientes usados nos gráficos de distribuição (performance)"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.72rem; color: #4a5568; line-height: 1.6;'>
      <b style='color: #718096;'>Projeto:</b> Book de Variáveis<br>
      <b style='color: #718096;'>Pipeline:</b> PySpark → Parquet → Datalake<br>
      <b style='color: #718096;'>Clientes:</b> {:,}<br>
      <b style='color: #718096;'>Features:</b> {:,}<br>
      <b style='color: #718096;'>Janelas:</b> 3m · 6m · 9m · 12m
    </div>
    """.format(len(df), len(df.columns) - 1), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <h1 style='font-family: Space Mono, monospace; font-size: 1.8rem; margin-bottom: 0; color: #e2e8f0;'>
      Book de Variáveis
    </h1>
    <p style='color: #4a5568; font-size: 0.85rem; margin-top: 4px;'>
      Feature engineering · Janela selecionada: <span style='color: #4299e1; font-weight: 600;'>{janela_sel}</span>
    </p>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown(f"""
    <div style='text-align: right; padding-top: 8px;'>
      <span class='badge badge-green'>✓ {len(df):,} clientes</span>&nbsp;
      <span class='badge badge-blue'>160 features</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# ABAS
# ──────────────────────────────────────────────────────────────
tab_overview, tab_metricas, tab_distribuicao, tab_correlacao, tab_missingness, tab_explorador = st.tabs([
    "🏠 Visão Geral",
    "📊 Métricas por Janela",
    "📈 Distribuições",
    "🔗 Correlações",
    "🕳️ Completude",
    "🔍 Explorador de Clientes",
])

# helper
def fmt(val, prefix="R$"):
    if pd.isna(val): return "—"
    if abs(val) >= 1e6: return f"{prefix} {val/1e6:.2f}M"
    if abs(val) >= 1e3: return f"{prefix} {val/1e3:.1f}k"
    return f"{prefix} {val:,.0f}"

def fmt_n(val):
    if pd.isna(val): return "—"
    return f"{val:,.0f}"

# ══════════════════════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════
with tab_overview:

    # KPIs de topo
    m1, m2, m3, m4, m5 = st.columns(5)

    # Pegar vars da janela selecionada para Métrica A
    vs_a = var_names(1, w_idx)    # soma, media, min, max
    cnt_a = count_var([21,22,23,24,25], w_idx)

    m1.metric("Total de Clientes", f"{len(df):,}")
    m2.metric(
        "Valor Médio (Métrica A)",
        fmt(df[vs_a[1]].median()),
        help="Mediana da média por cliente"
    )
    m3.metric(
        "Volume Total (Métrica A)",
        fmt(df[vs_a[0]].sum()),
        help="Soma de todos os valores"
    )
    m4.metric(
        "Frequência Média",
        f"{df[cnt_a].mean():.1f}",
        help="Média de transações por cliente"
    )
    coverage = (df[vs_a[0]].notna().sum() / len(df) * 100)
    m5.metric("Cobertura da Janela", f"{coverage:.1f}%")

    st.markdown("")

    # Gráfico 1: Distribuição de valor médio por janela (linha)
    st.markdown('<div class="section-header">Evolução do Valor Médio por Janela Temporal</div>', unsafe_allow_html=True)

    medians_metric_a = {}
    for w_name, w_i in WINDOWS.items():
        vs = var_names(1, w_i)  # índice 1 = media
        medians_metric_a[w_name] = df[vs[1]].median()

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=list(medians_metric_a.keys()),
        y=list(medians_metric_a.values()),
        mode="lines+markers+text",
        text=[fmt(v) for v in medians_metric_a.values()],
        textposition="top center",
        textfont=dict(family="Space Mono", size=10, color="#a0aec0"),
        line=dict(color="#4299e1", width=2.5),
        marker=dict(size=10, color="#4299e1", line=dict(width=2, color="#1a365d")),
        fill="tozeroy",
        fillcolor="rgba(66,153,225,0.08)",
    ))
    fig_ev.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#718096"),
        xaxis=dict(showgrid=False, tickfont=dict(color="#718096")),
        yaxis=dict(showgrid=True, gridcolor="#1e2433", tickfont=dict(color="#718096"), tickprefix="R$ "),
        margin=dict(l=60, r=20, t=20, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    # Duas colunas: distribuição de frequência + cobertura por janela
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Distribuição de Frequência (Janela Atual)</div>', unsafe_allow_html=True)
        freq_data = df[cnt_a].dropna()
        fig_freq = px.histogram(
            freq_data, nbins=20,
            color_discrete_sequence=["#4299e1"],
            labels={"value": "Frequência", "count": "Clientes"},
        )
        fig_freq.update_layout(
            height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#718096"), showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(color="#718096")),
            yaxis=dict(showgrid=True, gridcolor="#1e2433", tickfont=dict(color="#718096")),
            margin=dict(l=50, r=20, t=10, b=40),
        )
        fig_freq.update_traces(marker_line_width=0)
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Cobertura por Janela (todas as métricas)</div>', unsafe_allow_html=True)
        cov_data = []
        for w_name, w_i in WINDOWS.items():
            for metric_name, meta in METRICAS.items():
                if meta["type"] == "monetary":
                    vs = var_names(meta["start"], w_i)
                    cov = df[vs[0]].notna().mean() * 100
                    cov_data.append({"Janela": w_name, "Métrica": metric_name.split(" — ")[0], "Cobertura": cov})
        df_cov = pd.DataFrame(cov_data)
        fig_cov = px.bar(
            df_cov, x="Janela", y="Cobertura", color="Métrica",
            barmode="group",
            color_discrete_sequence=["#4299e1","#68d391","#f6ad55","#fc8181"],
        )
        fig_cov.update_layout(
            height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#718096"),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, tickfont=dict(color="#718096")),
            yaxis=dict(showgrid=True, gridcolor="#1e2433", tickfont=dict(color="#718096"), ticksuffix="%"),
            margin=dict(l=50, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    # Tabela sumário
    st.markdown('<div class="section-header">Sumário Estatístico por Métrica — Janela: ' + janela_sel + '</div>', unsafe_allow_html=True)
    summary_rows = []
    for metric_name, meta in METRICAS.items():
        if meta["type"] == "monetary":
            vs = var_names(meta["start"], w_idx)
            v_soma, v_media, v_min, v_max = vs
            row = {
                "Métrica": metric_name,
                "Cobertura": f"{df[v_soma].notna().mean()*100:.1f}%",
                "Clientes": f"{df[v_soma].notna().sum():,}",
                "Mediana Soma": fmt(df[v_soma].median()),
                "Mediana Média": fmt(df[v_media].median()),
                "P25 Média": fmt(df[v_media].quantile(0.25)),
                "P75 Média": fmt(df[v_media].quantile(0.75)),
            }
        else:
            cnt_v = count_var(meta["vars"], w_idx)
            row = {
                "Métrica": metric_name,
                "Cobertura": f"{df[cnt_v].notna().mean()*100:.1f}%",
                "Clientes": f"{df[cnt_v].notna().sum():,}",
                "Mediana Soma": "—",
                "Mediana Média": fmt_n(df[cnt_v].median()),
                "P25 Média": fmt_n(df[cnt_v].quantile(0.25)),
                "P75 Média": fmt_n(df[cnt_v].quantile(0.75)),
            }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# ABA 2 — MÉTRICAS POR JANELA
# ══════════════════════════════════════════════════════════════
with tab_metricas:

    st.markdown(f'<div class="section-header">Comparação entre Janelas — Todas as Métricas Monetárias</div>', unsafe_allow_html=True)

    # Box plots comparando janelas para cada métrica
    monetary_metrics = {k: v for k, v in METRICAS.items() if v["type"] == "monetary"}

    for metric_name, meta in monetary_metrics.items():
        with st.expander(f"📊 {metric_name}", expanded=(list(monetary_metrics.keys()).index(metric_name) == 0)):
            
            # Montar df long com média por janela
            traces = []
            palette = {"Total": "#4299e1", "3 meses": "#f6ad55", "6 meses": "#68d391", "9 meses": "#fc8181", "12 meses": "#b794f4"}
            
            fig = go.Figure()
            for w_name, w_i in WINDOWS.items():
                vs = var_names(meta["start"], w_i)
                vals = df[vs[1]].dropna()  # média
                if len(vals) < 10:
                    continue
                fig.add_trace(go.Box(
                    y=vals,
                    name=w_name,
                    marker_color=palette.get(w_name, "#718096"),
                    boxmean="sd",
                    line_width=1.5,
                ))
            
            fig.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#718096"),
                xaxis=dict(showgrid=False, tickfont=dict(color="#a0aec0")),
                yaxis=dict(showgrid=True, gridcolor="#1e2433", tickfont=dict(color="#718096")),
                margin=dict(l=60, r=20, t=20, b=40),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabela de stats por janela
            rows = []
            for w_name, w_i in WINDOWS.items():
                vs = var_names(meta["start"], w_i)
                col_media = df[vs[1]]
                col_soma  = df[vs[0]]
                rows.append({
                    "Janela": w_name,
                    "Clientes": f"{col_media.notna().sum():,}",
                    "Cobertura": f"{col_media.notna().mean()*100:.1f}%",
                    "Média (mediana)": fmt(col_media.median()),
                    "Média (P25)": fmt(col_media.quantile(0.25)),
                    "Média (P75)": fmt(col_media.quantile(0.75)),
                    "Soma (mediana)": fmt(col_soma.median()),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Métricas de contagem
    st.markdown('<div class="section-header">Métricas de Contagem — Comparação entre Janelas</div>', unsafe_allow_html=True)
    count_metrics = {k: v for k, v in METRICAS.items() if v["type"] == "count"}

    col_cnt1, col_cnt2 = st.columns(2)
    for i, (metric_name, meta) in enumerate(count_metrics.items()):
        col = col_cnt1 if i % 2 == 0 else col_cnt2
        with col:
            medians = {}
            for w_name, w_i in WINDOWS.items():
                cnt_v = count_var(meta["vars"], w_i)
                medians[w_name] = df[cnt_v].median()
            
            fig_bar = px.bar(
                x=list(medians.keys()), y=list(medians.values()),
                title=metric_name,
                labels={"x": "Janela", "y": "Mediana"},
                color_discrete_sequence=["#4299e1"],
                text=[f"{v:.1f}" if not pd.isna(v) else "—" for v in medians.values()],
            )
            fig_bar.update_layout(
                height=220, title_font_size=12, title_font_color="#a0aec0",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#718096"),
                xaxis=dict(showgrid=False, tickfont=dict(color="#a0aec0")),
                yaxis=dict(showgrid=True, gridcolor="#1e2433", tickfont=dict(color="#718096")),
                margin=dict(l=40, r=10, t=40, b=40), showlegend=False,
            )
            fig_bar.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 3 — DISTRIBUIÇÕES
# ══════════════════════════════════════════════════════════════
with tab_distribuicao:

    st.markdown(f'<div class="section-header">Distribuições de Variáveis · {janela_sel}</div>', unsafe_allow_html=True)

    sample_df = df.sample(n=min(int(n_clientes), len(df)), random_state=42)

    all_vars_window = []
    for metric_name, meta in METRICAS.items():
        if meta["type"] == "monetary":
            vs = var_names(meta["start"], w_idx)
            for v, label in zip(vs, ["soma", "média", "min", "max"]):
                cov = df[v].notna().mean() * 100
                if cov >= min_cobertura:
                    all_vars_window.append({"var": v, "label": f"{metric_name.split(' — ')[0]} · {label}", "cov": cov})
        else:
            cnt_v = count_var(meta["vars"], w_idx)
            cov = df[cnt_v].notna().mean() * 100
            if cov >= min_cobertura:
                all_vars_window.append({"var": cnt_v, "label": f"{metric_name.split(' — ')[0]} · contagem", "cov": cov})

    if not all_vars_window:
        st.warning(f"Nenhuma variável com cobertura ≥ {min_cobertura}% na janela {janela_sel}. Reduza o filtro de cobertura mínima.")
    else:
        var_options = {row["label"]: row["var"] for row in all_vars_window}
        selected_label = st.selectbox("Selecione a variável", options=list(var_options.keys()))
        selected_var = var_options[selected_label]

        vals = sample_df[selected_var].dropna()
        if len(vals) < 10:
            st.info("Poucos dados para esta variável na janela selecionada.")
        else:
            col_dist1, col_dist2 = st.columns([2, 1])
            with col_dist1:
                # Histograma + KDE
                fig_hist = px.histogram(
                    vals, nbins=50,
                    marginal="box",
                    color_discrete_sequence=["#4299e1"],
                    labels={"value": selected_label},
                )
                fig_hist.update_layout(
                    height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#718096"), showlegend=False,
                    xaxis=dict(showgrid=False, tickfont=dict(color="#718096")),
                    yaxis=dict(showgrid=True, gridcolor="#1e2433", tickfont=dict(color="#718096")),
                    margin=dict(l=60, r=20, t=20, b=50),
                )
                fig_hist.update_traces(marker_line_width=0)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_dist2:
                st.markdown("**Estatísticas descritivas**")
                desc = vals.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                stats_rows = [
                    ("Clientes", f"{int(desc['count']):,}"),
                    ("Média", fmt(desc['mean'], "")),
                    ("Desvio Padrão", fmt(desc['std'], "")),
                    ("P1", fmt(desc['1%'], "")),
                    ("P5", fmt(desc['5%'], "")),
                    ("P25", fmt(desc['25%'], "")),
                    ("Mediana", fmt(desc['50%'], "")),
                    ("P75", fmt(desc['75%'], "")),
                    ("P95", fmt(desc['95%'], "")),
                    ("P99", fmt(desc['99%'], "")),
                    ("Máximo", fmt(desc['max'], "")),
                ]
                for label_s, val_s in stats_rows:
                    c1, c2 = st.columns(2)
                    c1.markdown(f"<small style='color:#718096'>{label_s}</small>", unsafe_allow_html=True)
                    c2.markdown(f"<span class='mono'>{val_s}</span>", unsafe_allow_html=True)

                skew = vals.skew()
                kurt = vals.kurtosis()
                st.markdown("---")
                st.markdown(f"<small style='color:#718096'>Assimetria</small> <span class='mono'>{skew:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"<small style='color:#718096'>Curtose</small> <span class='mono'>{kurt:.2f}</span>", unsafe_allow_html=True)
                
                zeros_pct = (vals == 0).mean() * 100
                null_pct_v = df[selected_var].isna().mean() * 100
                st.markdown("---")
                st.markdown(f"<small style='color:#718096'>Nulos</small> <span class='mono'>{null_pct_v:.1f}%</span>", unsafe_allow_html=True)
                st.markdown(f"<small style='color:#718096'>Zeros</small> <span class='mono'>{zeros_pct:.1f}%</span>", unsafe_allow_html=True)

        # Grid de histogramas das principais variáveis
        st.markdown('<div class="section-header">Visão Geral — Grid de Distribuições (Métricas de Média)</div>', unsafe_allow_html=True)

        main_vars = []
        main_labels = []
        for metric_name, meta in METRICAS.items():
            if meta["type"] == "monetary":
                vs = var_names(meta["start"], w_idx)
                cov = df[vs[1]].notna().mean() * 100
                if cov >= 20:
                    main_vars.append(vs[1])
                    main_labels.append(metric_name.split(" — ")[0])

        if main_vars:
            n_cols = 2
            n_rows = (len(main_vars) + 1) // n_cols
            fig_grid = make_subplots(rows=max(1,n_rows), cols=n_cols,
                                      subplot_titles=main_labels)
            colors = ["#4299e1","#68d391","#f6ad55","#b794f4"]
            for i, (var, lbl) in enumerate(zip(main_vars, main_labels)):
                row = i // n_cols + 1
                col = i % n_cols + 1
                vals_g = sample_df[var].dropna()
                fig_grid.add_trace(
                    go.Histogram(x=vals_g, nbinsx=30, name=lbl,
                                  marker_color=colors[i % len(colors)],
                                  showlegend=False),
                    row=row, col=col
                )
            fig_grid.update_layout(
                height=160 * max(1, n_rows) + 60,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#718096", size=10),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig_grid.update_xaxes(showgrid=False, tickfont=dict(color="#718096", size=9))
            fig_grid.update_yaxes(showgrid=True, gridcolor="#1e2433", tickfont=dict(color="#718096", size=9))
            st.plotly_chart(fig_grid, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 4 — CORRELAÇÕES
# ══════════════════════════════════════════════════════════════
with tab_correlacao:

    st.markdown(f'<div class="section-header">Matriz de Correlação · {janela_sel}</div>', unsafe_allow_html=True)

    # Pegar vars de média para cada métrica
    corr_vars = []
    corr_labels = []
    for metric_name, meta in METRICAS.items():
        if meta["type"] == "monetary":
            vs = var_names(meta["start"], w_idx)
            corr_vars.append(vs[1])  # média
            corr_labels.append(metric_name.split(" — ")[1] if " — " in metric_name else metric_name)
        else:
            cnt_v = count_var(meta["vars"], w_idx)
            corr_vars.append(cnt_v)
            corr_labels.append(metric_name.split(" — ")[1] if " — " in metric_name else metric_name)

    df_corr = df[corr_vars].dropna(how="all")
    corr_matrix = df_corr.corr(method="pearson")
    corr_matrix.columns = corr_labels
    corr_matrix.index = corr_labels

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_corr.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#a0aec0", size=9),
        coloraxis_colorbar=dict(tickfont=dict(color="#718096")),
        margin=dict(l=100, r=20, t=20, b=100),
        xaxis=dict(tickfont=dict(color="#a0aec0", size=9), tickangle=30),
        yaxis=dict(tickfont=dict(color="#a0aec0", size=9)),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter matrix para métricas monetárias
    st.markdown('<div class="section-header">Scatter Matrix — Valores Médios</div>', unsafe_allow_html=True)
    monetary_vars = [var_names(meta["start"], w_idx)[1] for m, meta in METRICAS.items() if meta["type"] == "monetary"]
    monetary_labels = [m.split(" — ")[0] for m, meta in METRICAS.items() if meta["type"] == "monetary"]

    df_scatter = df[monetary_vars].dropna(how="all").sample(n=min(800, len(df)), random_state=42)
    df_scatter.columns = monetary_labels

    fig_scatter = px.scatter_matrix(
        df_scatter,
        dimensions=monetary_labels,
        color_discrete_sequence=["#4299e1"],
        opacity=0.3,
    )
    fig_scatter.update_traces(marker=dict(size=2, line_width=0), diagonal_visible=False)
    fig_scatter.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#718096", size=9),
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 5 — COMPLETUDE / MISSINGNESS
# ══════════════════════════════════════════════════════════════
with tab_missingness:

    st.markdown('<div class="section-header">Análise de Completude do Book de Variáveis</div>', unsafe_allow_html=True)

    # Completude por janela para cada variável principal
    completude_data = []
    for metric_name, meta in METRICAS.items():
        short = metric_name.split(" — ")[0]
        if meta["type"] == "monetary":
            for w_name, w_i in WINDOWS.items():
                vs = var_names(meta["start"], w_i)
                cov = df[vs[0]].notna().mean() * 100
                completude_data.append({"Métrica": short, "Janela": w_name, "Completude (%)": round(cov, 1)})
        else:
            for w_name, w_i in WINDOWS.items():
                cnt_v = count_var(meta["vars"], w_i)
                cov = df[cnt_v].notna().mean() * 100
                completude_data.append({"Métrica": short, "Janela": w_name, "Completude (%)": round(cov, 1)})

    df_comp = pd.DataFrame(completude_data)

    fig_heat = px.density_heatmap(
        df_comp, x="Janela", y="Métrica", z="Completude (%)",
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig_heat.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#a0aec0"),
        coloraxis_colorbar=dict(tickfont=dict(color="#718096")),
        xaxis=dict(tickfont=dict(color="#a0aec0"), categoryorder="array",
                   categoryarray=["3 meses","6 meses","9 meses","12 meses","Total"]),
        yaxis=dict(tickfont=dict(color="#a0aec0")),
        margin=dict(l=150, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Tabela de missingness completa
    st.markdown('<div class="section-header">Taxa de Nulos — Todas as Variáveis do Book</div>', unsafe_allow_html=True)

    miss_rows = []
    for v in df.columns:
        if v == "id_cliente":
            continue
        null_p = df[v].isna().mean() * 100
        zero_p = (df[v] == 0).mean() * 100 if df[v].dtype != object else 0
        miss_rows.append({
            "Variável": v,
            "Nulos": f"{null_p:.1f}%",
            "Zeros": f"{zero_p:.1f}%",
            "Preenchidos": f"{100-null_p:.1f}%",
            "Tipo": str(df[v].dtype),
        })

    df_miss = pd.DataFrame(miss_rows)

    # filtro
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        max_null = st.slider("Filtrar por nulos máximos", 0, 100, 100, format="%d%%")
    with col_f2:
        tipo_filter = st.multiselect("Filtrar por tipo", options=df_miss["Tipo"].unique().tolist(),
                                      default=df_miss["Tipo"].unique().tolist())

    df_miss_f = df_miss[
        (df_miss["Nulos"].str.rstrip("%").astype(float) <= max_null) &
        (df_miss["Tipo"].isin(tipo_filter))
    ]
    st.dataframe(df_miss_f, use_container_width=True, hide_index=True)

    # Download
    buf = io.BytesIO()
    df_miss_f.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        "⬇️ Exportar completude para Excel",
        buf.getvalue(),
        "completude_book_variaveis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ══════════════════════════════════════════════════════════════
# ABA 6 — EXPLORADOR DE CLIENTES
# ══════════════════════════════════════════════════════════════
with tab_explorador:

    st.markdown('<div class="section-header">Explorador de Clientes — Consulta Individual</div>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 2])

    with col_e1:
        search_id = st.number_input(
            "ID do cliente (id_cliente)",
            min_value=int(df["id_cliente"].min()),
            max_value=int(df["id_cliente"].max()),
            value=int(df["id_cliente"].iloc[0]),
            step=1,
        )
        st.markdown("**ou explore aleatoriamente:**")
        if st.button("🎲 Cliente aleatório"):
            search_id = int(df["id_cliente"].sample(1).iloc[0])
            st.rerun()

    row = df[df["id_cliente"] == search_id]
    if row.empty:
        st.warning(f"Cliente {search_id} não encontrado.")
    else:
        row = row.iloc[0]

        with col_e2:
            st.markdown(f"**Cliente:** `{search_id}`")
            # Cobertura de janelas
            janela_coverage = {}
            for w_name, w_i in WINDOWS.items():
                vs = var_names(1, w_i)
                janela_coverage[w_name] = "✅" if not pd.isna(row[vs[0]]) else "❌"
            cols_j = st.columns(5)
            for j, (w_name, status) in enumerate(janela_coverage.items()):
                cols_j[j].markdown(f"<div style='text-align:center'><b style='color:#718096;font-size:0.7rem'>{w_name}</b><br>{status}</div>", unsafe_allow_html=True)

        st.markdown("")

        # Perfil do cliente por janela
        tabs_cliente = st.tabs(list(WINDOWS.keys()))
        for tab_j, (w_name, w_i) in zip(tabs_cliente, WINDOWS.items()):
            with tab_j:
                metrics_row = []
                for metric_name, meta in METRICAS.items():
                    if meta["type"] == "monetary":
                        vs = var_names(meta["start"], w_i)
                        metrics_row.append({
                            "Métrica": metric_name,
                            "Soma": fmt(row.get(vs[0], np.nan)),
                            "Média": fmt(row.get(vs[1], np.nan)),
                            "Mín": fmt(row.get(vs[2], np.nan)),
                            "Máx": fmt(row.get(vs[3], np.nan)),
                        })
                    else:
                        cnt_v = count_var(meta["vars"], w_i)
                        metrics_row.append({
                            "Métrica": metric_name,
                            "Soma": "—",
                            "Média": fmt_n(row.get(cnt_v, np.nan)),
                            "Mín": "—",
                            "Máx": "—",
                        })
                st.dataframe(pd.DataFrame(metrics_row), use_container_width=True, hide_index=True)

        # Radar chart: percentil do cliente em relação à base
        st.markdown('<div class="section-header">Posição Percentílica do Cliente</div>', unsafe_allow_html=True)
        radar_labels = []
        radar_vals = []
        for metric_name, meta in METRICAS.items():
            if meta["type"] == "monetary":
                vs = var_names(meta["start"], w_idx)
                v = row.get(vs[1], np.nan)
                base = df[vs[1]].dropna()
                if not pd.isna(v) and len(base) > 10:
                    pct = (base < v).mean() * 100
                    radar_labels.append(metric_name.split(" — ")[0])
                    radar_vals.append(pct)

        if radar_vals:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(66,153,225,0.15)",
                line=dict(color="#4299e1", width=2),
                name="Percentil",
            ))
            fig_radar.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#718096", size=8), gridcolor="#1e2433"),
                    angularaxis=dict(tickfont=dict(color="#a0aec0", size=9), gridcolor="#1e2433", linecolor="#1e2433"),
                ),
                font=dict(family="Inter"),
                margin=dict(l=60, r=60, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Dados insuficientes para o radar na janela selecionada.")

        # Exportar perfil do cliente
        buf_c = io.BytesIO()
        row_df = pd.DataFrame([row])
        row_df.to_excel(buf_c, index=False, engine="openpyxl")
        st.download_button(
            f"⬇️ Exportar perfil do cliente {search_id}",
            buf_c.getvalue(),
            f"cliente_{search_id}_book.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
