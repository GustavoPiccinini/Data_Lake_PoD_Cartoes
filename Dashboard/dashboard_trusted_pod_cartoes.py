"""
Dashboard — Camada Trusted · PoD Cartões
Projeto: Data Lake 2024 · Engenharia de Dados Júnior · PoD Academy

Arquitetura de performance:
  - DuckDB como engine central (in-process, sem pandas no critical path)
  - @st.cache_resource: conexão DuckDB persiste entre reruns
  - @st.cache_data: resultados de queries pesadas cacheados por safra
  - Consulta de cliente via SQL parametrizado (ms, não segundos)
  - DataFrames pandas apenas para renderização final

Execute: streamlit run dashboard_trusted_pod_cartoes.py
"""

import io
import os
import random

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PoD Cartões · Trusted Layer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# ESTILO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #060810; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; }

section[data-testid="stSidebar"] > div:first-child {
    background: #080b14; border-right: 1px solid #0f1629;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: #4b5563 !important; font-size: 0.78rem; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0c1222 0%, #0a0f1e 100%);
    border: 1px solid #141e38; border-radius: 10px; padding: 18px 20px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: #1d4ed8; }
div[data-testid="metric-container"] label {
    color: #374151 !important; font-size: 0.68rem !important;
    letter-spacing: 0.12em; text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.45rem !important; color: #e2e8f0 !important;
}
div[data-testid="metric-container"] [data-testid="metric-delta"] { font-size: 0.72rem !important; }

div[data-baseweb="tab-list"] {
    background: #080b14; border-radius: 8px; padding: 4px;
    gap: 2px; border: 1px solid #0f1629;
}
button[data-baseweb="tab"] {
    border-radius: 6px !important; font-family: 'DM Sans' !important;
    font-size: 0.8rem !important; font-weight: 500 !important; color: #374151 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #0f1629 !important; color: #e2e8f0 !important;
}
h1, h2, h3 { color: #e2e8f0 !important; }

.sec { border-left: 3px solid #1d4ed8; padding-left: 12px;
       color: #cbd5e1; font-weight: 600; font-size: 0.92rem;
       margin: 1.4rem 0 0.7rem 0; }

.badge { display:inline-block; padding:2px 10px; border-radius:20px;
         font-size:0.68rem; font-weight:600; font-family:'IBM Plex Mono',monospace; letter-spacing:.04em; }
.b-blue  { background:#172554; color:#60a5fa; border:1px solid #1e3a8a; }
.b-green { background:#052e16; color:#4ade80; border:1px solid #166534; }

.info-box { background:#0c1222; border:1px solid #141e38; border-radius:8px;
            padding:12px 16px; margin-bottom:1rem; }
.info-box .title { color:#60a5fa; font-size:0.78rem; font-weight:600; margin-bottom:4px; }
.info-box .body  { color:#4b5563; font-size:0.76rem; line-height:1.75; }

div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def brl(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        v = float(v)
        if abs(v) >= 1_000_000:
            return f"R$ {v/1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"R$ {v/1_000:.1f}k"
        return f"R$ {v:,.2f}"
    except Exception:
        return "—"

def nn(v):
    try:
        if v is None:
            return "—"
        return f"{int(v):,}"
    except Exception:
        return "—"

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#084292"),
    margin=dict(l=55, r=20, t=35, b=45),
)

def ax(fig):
    fig.update_xaxes(showgrid=False, tickfont=dict(color="#1a304e", size=10), linecolor="#0f1629")
    fig.update_yaxes(showgrid=True, gridcolor="#0f1629", tickfont=dict(color="#16253a", size=10), zeroline=False)
    return fig

STATUS_COLORS = {
    "Pago no Prazo":    "#0C1EE9",
    "Pago em Atraso":   "#14026b",
    "Abaixo do Mínimo": "#6B0303",
    "Sem Pagamento":    "#450101",
}

# ══════════════════════════════════════════════════════════════
# DUCKDB — conexão única persistida via cache_resource
# cache_resource = criada UMA vez, reutilizada em todos os reruns
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def init_db() -> duckdb.DuckDBPyConnection:
    base = os.path.dirname(os.path.abspath(__file__))
    con  = duckdb.connect()

    fat_path = os.path.join(base, "tb_faturas",    "*", "*.parquet").replace("\\", "/")
    pag_path = os.path.join(base, "tb_pagamentos", "*", "*.parquet").replace("\\", "/")

    # Tipos explícitos na view — zero conversão no runtime
    con.execute(f"""
        CREATE OR REPLACE VIEW tb_faturas AS
        SELECT
            CAST(id_fatura              AS BIGINT)  AS id_fatura,
            CAST(id_cliente             AS BIGINT)  AS id_cliente,
            CAST(data_emissao           AS DATE)    AS data_emissao,
            CAST(data_vencimento        AS DATE)    AS data_vencimento,
            CAST(valor_fatura           AS DOUBLE)  AS valor_fatura,
            CAST(valor_pagamento_minimo AS DOUBLE)  AS valor_pagamento_minimo,
            regexp_extract(filename, 'ref=([^/\\\\]+)', 1) AS safra
        FROM read_parquet('{fat_path}', filename=true, union_by_name=true)
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW tb_pagamentos AS
        SELECT
            CAST(id_pagamento   AS BIGINT)  AS id_pagamento,
            CAST(id_fatura      AS BIGINT)  AS id_fatura,
            CAST(id_cliente     AS BIGINT)  AS id_cliente,
            CAST(data_pagamento AS DATE)    AS data_pagamento,
            CAST(valor_pagamento AS DOUBLE) AS valor_pagamento,
            regexp_extract(filename, 'ref=([^/\\\\]+)', 1) AS safra
        FROM read_parquet('{pag_path}', filename=true, union_by_name=true)
    """)

    # View join — calculada uma vez pelo DuckDB, memoizada
    con.execute("""
        CREATE OR REPLACE VIEW vw_join AS
        SELECT
            f.id_fatura,
            f.id_cliente,
            f.safra,
            f.data_emissao,
            f.data_vencimento,
            f.valor_fatura,
            f.valor_pagamento_minimo,
            p.data_pagamento,
            COALESCE(p.valor_pagamento, 0)                          AS valor_pagamento,
            DATEDIFF('day', f.data_vencimento, p.data_pagamento)    AS dias_atraso,
            CASE
                WHEN p.data_pagamento IS NULL                               THEN 'Sem Pagamento'
                WHEN DATEDIFF('day', f.data_vencimento, p.data_pagamento) > 0 THEN 'Pago em Atraso'
                WHEN COALESCE(p.valor_pagamento, 0) < f.valor_pagamento_minimo THEN 'Abaixo do Mínimo'
                ELSE 'Pago no Prazo'
            END AS status
        FROM tb_faturas f
        LEFT JOIN tb_pagamentos p
               ON f.id_fatura  = p.id_fatura
              AND f.id_cliente = p.id_cliente
    """)

    return con


# ── Accessors ──
def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

def q1(sql: str):
    r = con.execute(sql).fetchone()
    return r[0] if r else None


# ══════════════════════════════════════════════════════════════
# QUERIES CACHEADAS POR SAFRA
# Cada função recebe safras como tuple (imutável = boa chave de cache).
# Resultado fica em memória — zero I/O no rerun.
# ══════════════════════════════════════════════════════════════
def wf(safras: tuple) -> str:
    """WHERE clause para filtrar safras."""
    if not safras:
        return "1=1"
    lst = ", ".join(f"'{s}'" for s in safras)
    return f"safra IN ({lst})"

@st.cache_data(ttl=3600, show_spinner=False)
def get_kpis(safras: tuple) -> dict:
    w = wf(safras)
    row = con.execute(f"""
        SELECT
            COUNT(DISTINCT f.id_cliente)                                    AS n_clientes,
            COUNT(*)                                                         AS n_faturas,
            SUM(f.valor_fatura)                                              AS vol_faturado,
            (SELECT COUNT(DISTINCT id_cliente) FROM tb_pagamentos WHERE {w}) AS n_pagantes,
            (SELECT SUM(valor_pagamento) FROM tb_pagamentos WHERE {w})       AS vol_pago,
            SUM(CASE WHEN j.status != 'Pago no Prazo' THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0)                                AS taxa_inad
        FROM tb_faturas f
        JOIN vw_join j USING (id_fatura, id_cliente, safra)
        WHERE f.{w}
    """).fetchone()
    return dict(zip(["n_clientes","n_faturas","vol_faturado","n_pagantes","vol_pago","taxa_inad"], row))

@st.cache_data(ttl=3600, show_spinner=False)
def get_por_safra(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"""
        SELECT
            j.safra,
            COUNT(*)                                                                AS n_fat,
            COUNT(DISTINCT j.id_cliente)                                            AS n_cli,
            SUM(j.valor_fatura)                                                     AS vol_fat,
            SUM(j.valor_pagamento)                                                  AS vol_pag,
            ROUND(SUM(j.valor_pagamento)*100.0/NULLIF(SUM(j.valor_fatura),0), 1)   AS cob_pct,
            SUM(CASE WHEN j.status != 'Pago no Prazo' THEN 1 ELSE 0 END)           AS inadimplentes,
            ROUND(SUM(CASE WHEN j.status != 'Pago no Prazo' THEN 1 ELSE 0 END)
                  *100.0/NULLIF(COUNT(*),0), 1)                                     AS taxa_inad
        FROM vw_join j WHERE j.{w}
        GROUP BY j.safra ORDER BY j.safra
    """).df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_status_counts(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"""
        SELECT status, COUNT(*) AS qtd, SUM(valor_fatura) AS volume
        FROM vw_join WHERE {w}
        GROUP BY status ORDER BY qtd DESC
    """).df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_hist_fatura(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"SELECT valor_fatura FROM tb_faturas WHERE {w}").df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_scatter_fat(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"""
        SELECT valor_fatura, valor_pagamento_minimo
        FROM tb_faturas WHERE {w} USING SAMPLE 1500
    """).df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_faixas(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"""
        SELECT
            CASE
                WHEN valor_fatura <  5000 THEN '< R$5k'
                WHEN valor_fatura < 15000 THEN 'R$5k–15k'
                WHEN valor_fatura < 30000 THEN 'R$15k–30k'
                WHEN valor_fatura < 50000 THEN 'R$30k–50k'
                ELSE '> R$50k'
            END AS faixa,
            COUNT(*) AS qtd
        FROM tb_faturas WHERE {w}
        GROUP BY 1 ORDER BY MIN(valor_fatura)
    """).df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_percentis_fatura(safras: tuple) -> dict:
    w = wf(safras)
    row = con.execute(f"""
        SELECT
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY valor_fatura),
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY valor_fatura),
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY valor_fatura),
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY valor_fatura),
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY valor_fatura),
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY valor_fatura)
        FROM tb_faturas WHERE {w}
    """).fetchone()
    return dict(zip(["P10","P25","P50","P75","P90","P99"], row))

@st.cache_data(ttl=3600, show_spinner=False)
def get_hist_pagamento(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"SELECT valor_pagamento FROM tb_pagamentos WHERE {w}").df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_aging(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"""
        SELECT dias_atraso, status FROM vw_join
        WHERE {w} AND dias_atraso IS NOT NULL
    """).df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_top_inad(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"""
        SELECT
            id_cliente,
            COUNT(*)            AS faturas_inad,
            SUM(valor_fatura)   AS vol_risco,
            MAX(status)         AS status_principal
        FROM vw_join
        WHERE {w} AND status != 'Pago no Prazo'
        GROUP BY id_cliente ORDER BY vol_risco DESC
        LIMIT 20
    """).df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_nulls(safras: tuple) -> tuple:
    w = wf(safras)
    df_nf = con.execute(f"""
        SELECT 'id_fatura'             AS col, COUNT(*)-COUNT(id_fatura)             AS nulos, COUNT(*) AS total FROM tb_faturas WHERE {w} UNION ALL
        SELECT 'id_cliente',                   COUNT(*)-COUNT(id_cliente),                    COUNT(*) FROM tb_faturas WHERE {w} UNION ALL
        SELECT 'data_emissao',                 COUNT(*)-COUNT(data_emissao),                  COUNT(*) FROM tb_faturas WHERE {w} UNION ALL
        SELECT 'data_vencimento',              COUNT(*)-COUNT(data_vencimento),               COUNT(*) FROM tb_faturas WHERE {w} UNION ALL
        SELECT 'valor_fatura',                 COUNT(*)-COUNT(valor_fatura),                  COUNT(*) FROM tb_faturas WHERE {w} UNION ALL
        SELECT 'valor_pagamento_minimo',       COUNT(*)-COUNT(valor_pagamento_minimo),        COUNT(*) FROM tb_faturas WHERE {w}
    """).df()
    df_np = con.execute(f"""
        SELECT 'id_pagamento'  AS col, COUNT(*)-COUNT(id_pagamento)  AS nulos, COUNT(*) AS total FROM tb_pagamentos WHERE {w} UNION ALL
        SELECT 'id_fatura',            COUNT(*)-COUNT(id_fatura),            COUNT(*) FROM tb_pagamentos WHERE {w} UNION ALL
        SELECT 'id_cliente',           COUNT(*)-COUNT(id_cliente),           COUNT(*) FROM tb_pagamentos WHERE {w} UNION ALL
        SELECT 'data_pagamento',       COUNT(*)-COUNT(data_pagamento),       COUNT(*) FROM tb_pagamentos WHERE {w} UNION ALL
        SELECT 'valor_pagamento',      COUNT(*)-COUNT(valor_pagamento),      COUNT(*) FROM tb_pagamentos WHERE {w}
    """).df()
    return df_nf, df_np

@st.cache_data(ttl=3600, show_spinner=False)
def get_duplicatas(safras: tuple) -> tuple:
    w = wf(safras)
    dup_fat = q1(f"SELECT COUNT(*)-COUNT(DISTINCT id_fatura||'-'||CAST(id_cliente AS VARCHAR)) FROM tb_faturas WHERE {w}")
    dup_pag = q1(f"SELECT COUNT(*)-COUNT(DISTINCT id_pagamento||'-'||CAST(id_cliente AS VARCHAR)) FROM tb_pagamentos WHERE {w}")
    return int(dup_fat or 0), int(dup_pag or 0)

@st.cache_data(ttl=3600, show_spinner=False)
def get_ref_integridade(safras: tuple) -> pd.DataFrame:
    w = wf(safras)
    return con.execute(f"""
        SELECT f.safra,
               COUNT(*) AS total_fat,
               SUM(CASE WHEN p.id_fatura IS NULL THEN 1 ELSE 0 END) AS sem_pag,
               ROUND(SUM(CASE WHEN p.id_fatura IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_sem_pag
        FROM tb_faturas f
        LEFT JOIN tb_pagamentos p ON f.id_fatura=p.id_fatura AND f.id_cliente=p.id_cliente
        WHERE f.{w} GROUP BY f.safra ORDER BY f.safra
    """).df()

@st.cache_data(ttl=3600, show_spinner=False)
def get_lista_clientes() -> list:
    return con.execute("SELECT DISTINCT id_cliente FROM tb_faturas ORDER BY id_cliente").df()["id_cliente"].tolist()

# ── Cliente: query parametrizada — DuckDB resolve em <10ms
def get_cliente_fat(id_cli: int) -> pd.DataFrame:
    return con.execute(f"""
        SELECT safra, data_emissao, data_vencimento, valor_fatura, valor_pagamento_minimo
        FROM tb_faturas WHERE id_cliente = {id_cli} ORDER BY data_emissao
    """).df()

def get_cliente_pag(id_cli: int) -> pd.DataFrame:
    return con.execute(f"""
        SELECT data_pagamento, valor_pagamento, id_fatura
        FROM tb_pagamentos WHERE id_cliente = {id_cli} ORDER BY data_pagamento
    """).df()

def get_cliente_status(id_cli: int) -> pd.DataFrame:
    return con.execute(f"""
        SELECT status, COUNT(*) AS qtd, SUM(valor_fatura) AS volume
        FROM vw_join WHERE id_cliente = {id_cli} GROUP BY status
    """).df()


# ══════════════════════════════════════════════════════════════
# INICIALIZAÇÃO
# ══════════════════════════════════════════════════════════════
con = init_db()
safras_disp = sorted(q("SELECT DISTINCT safra FROM tb_faturas ORDER BY safra")["safra"].tolist())

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
      <div style='font-family:IBM Plex Mono,monospace;color:#1d4ed8;font-size:1rem;font-weight:600;'>
        💳 PoD Cartões
      </div>
      <div style='color:#1f2937;font-size:0.7rem;margin-top:3px;'>
        Data Lake 
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
      <div class='title'> Camada Trusted</div>
      <div class='body'>
        Dados tratados, tipados e deduplicados.<br>
        Particionados por <b style='color:#6b7280'>safra mensal</b>.<br>
        Engine: <b style='color:#6b7280'>DuckDB · Parquet</b>.<br>
        Pipeline: <b style='color:#6b7280'>S3 → PySpark → Trusted</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📅 Referências disponíveis")
    safras_sel = st.multiselect("", options=safras_disp, default=safras_disp)
    if not safras_sel:
        safras_sel = safras_disp

    safras_key = tuple(sorted(safras_sel))

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem;color:#1f2937;line-height:1.9;'>
      <b style='color:#374151'>Tabelas:</b> tb_faturas · tb_pagamentos<br>
      <b style='color:#374151'>Data ref.:</b> 2024-02-01<br>
      <b style='color:#374151'>Fonte:</b> S3 · sa-east-1
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
hc1, hc2 = st.columns([4, 1])
with hc1:
    st.markdown("""
    <h1 style='font-family:IBM Plex Mono,monospace;font-size:1.65rem;margin-bottom:2px;'>
      💳 PoD Cartões — Camada Trusted
    </h1>
    <p style='color:#94a3b8;font-size:0.82rem;margin:0;'>
      Dados financeiros de faturas e pagamentos · Projeto Data Lake 
    </p>
    """, unsafe_allow_html=True)
with hc2:
    ref_min = min(safras_sel)
    ref_max = max(safras_sel)
    st.markdown(f"""
    <div style='text-align:right;padding-top:12px;'>
      <span class='badge b-blue'>Trusted Layer</span>&nbsp;
      <span class='badge b-green'>ref {ref_min}</span>
    </div>
    <div style='text-align:right;margin-top:4px;font-size:0.7rem;color:#374151;'>até {ref_max}</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# ABAS
# ══════════════════════════════════════════════════════════════
tab_exec, tab_fat_aba, tab_pag_aba, tab_inad, tab_quality, tab_cliente = st.tabs([
    "|   Visão Geral   |",
    "|   Faturas   |",
    "|   Pagamentos   |",
    "|   Inadimplência   |",
    "|   Qualidade dos Dados   |",
    "|   Perfil do Cliente   |",
]) 

# ══════════════════════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ════════════════════════════════════════════════
with tab_exec:

    kpis      = get_kpis(safras_key)
    df_safra  = get_por_safra(safras_key)
    df_status = get_status_counts(safras_key)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Clientes",          nn(kpis["n_clientes"]))
    k2.metric("Faturas emitidas",  nn(kpis["n_faturas"]))
    k3.metric("Volume faturado",   brl(kpis["vol_faturado"]))
    k4.metric("Volume pago",       brl(kpis["vol_pago"]))
    k5.metric("Clientes pagantes", nn(kpis["n_pagantes"]))
    taxa = float(kpis["taxa_inad"] or 0)
    k6.metric("Taxa inadimplência", f"{taxa:.1f}%",
              delta="⚠️ atenção" if taxa > 40 else "✅ controlada", delta_color="inverse")

    st.markdown("")
    st.markdown('<div class="sec">Volume Faturado vs Pago · por Referência</div>', unsafe_allow_html=True)
    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df_safra["safra"], y=df_safra["vol_fat"], name="Faturado",
                                  marker_color="#0d5fc4",
                                  text=[brl(v) for v in df_safra["vol_fat"]],
                                  textposition="outside", textfont=dict(size=9)))
        fig_vol.add_trace(go.Bar(x=df_safra["safra"], y=df_safra["vol_pag"], name="Pago",
                                  marker_color="#7265D3",
                                  text=[brl(v) for v in df_safra["vol_pag"]],
                                  textposition="outside", textfont=dict(size=9)))
        fig_vol.update_layout(height=300, barmode="group",
                               legend=dict(bgcolor="rgba(0,0,0,0)"), **LAYOUT)
        ax(fig_vol)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_ev2:
        fig_cob = go.Figure()
        fig_cob.add_trace(go.Bar(x=df_safra["safra"], y=df_safra["cob_pct"],
                                  marker_color="#2a08ed",
                                  text=[f"{v}%" for v in df_safra["cob_pct"]],
                                  textposition="outside", textfont=dict(size=9)))
        fig_cob.add_hline(y=100, line_dash="dash", line_color="#032254", line_width=1)
        fig_cob.update_layout(height=300, yaxis_range=[0, 120], showlegend=False,
                               title_text="% Volume Pago / Faturado",
                               title_font=dict(size=12, color="#141d2a"), **LAYOUT)
        ax(fig_cob)
        st.plotly_chart(fig_cob, use_container_width=True)

    st.markdown('<div class="sec">Distribuição de Status de Pagamento</div>', unsafe_allow_html=True)
    col_st1, col_st2 = st.columns([1, 2])
    df_status["pct"] = (df_status["qtd"] / df_status["qtd"].sum() * 100).round(1)

    with col_st1:
        fig_pie = px.pie(df_status, values="qtd", names="status",
                          color="status", color_discrete_map=STATUS_COLORS, hole=0.55)
        fig_pie.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#c5ccd4"),
                               legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
        fig_pie.update_traces(textfont_size=10, textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_st2:
        fig_bar_st = px.bar(df_status, x="status", y="qtd", color="status",
                             color_discrete_map=STATUS_COLORS,
                             text=df_status["pct"].apply(lambda x: f"{x:.1f}%"))
        fig_bar_st.update_layout(height=290, showlegend=False, **LAYOUT)
        fig_bar_st.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_bar_st)
        st.plotly_chart(fig_bar_st, use_container_width=True)

    st.markdown('<div class="sec">Resumo por Referência</div>', unsafe_allow_html=True)
    df_res = df_safra.copy()
    df_res["vol_fat"]   = df_res["vol_fat"].apply(brl)
    df_res["vol_pag"]   = df_res["vol_pag"].apply(brl)
    df_res["n_fat"]     = df_res["n_fat"].apply(nn)
    df_res["n_cli"]     = df_res["n_cli"].apply(nn)
    df_res["cob_pct"]   = df_res["cob_pct"].apply(lambda x: f"{x:.1f}%")
    df_res["taxa_inad"] = df_res["taxa_inad"].apply(lambda x: f"{x:.1f}%")
    df_res.columns = ["Ref","Faturas","Clientes","Vol Faturado","Vol Pago","% Pago","Inad.","Taxa Inad."]
    st.dataframe(df_res, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# ABA 2 — FATURAS
# ══════════════════════════════════════════════════════════════
with tab_fat_aba:

    df_h  = get_hist_fatura(safras_key)
    df_sc = get_scatter_fat(safras_key)
    df_fx = get_faixas(safras_key)
    pcts  = get_percentis_fatura(safras_key)

    kf1,kf2,kf3,kf4 = st.columns(4)
    kf1.metric("Ticket Médio",     brl(df_h["valor_fatura"].mean()))
    kf2.metric("Ticket Mediana",   brl(df_h["valor_fatura"].median()))
    kf3.metric("Maior Fatura",     brl(df_h["valor_fatura"].max()))
    kf4.metric("Pgto Mínimo Med.", brl(pcts.get("P50")))

    st.markdown("")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown('<div class="sec">Distribuição do Valor de Fatura</div>', unsafe_allow_html=True)
        fig_hf = px.histogram(df_h, x="valor_fatura", nbins=50,
                               color_discrete_sequence=["#0A40D5"], marginal="box",
                               labels={"valor_fatura":"Valor Fatura (R$)"})
        fig_hf.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_hf.update_traces(marker_line_width=0)
        ax(fig_hf)
        st.plotly_chart(fig_hf, use_container_width=True)

    with col_f2:
        st.markdown('<div class="sec">Fatura vs Pagamento Mínimo</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(df_sc, x="valor_fatura", y="valor_pagamento_minimo",
                             color_discrete_sequence=["#044594"], opacity=0.35,
                             labels={"valor_fatura":"Fatura (R$)","valor_pagamento_minimo":"Pgto Mín (R$)"})
        fig_sc.update_traces(marker=dict(size=3))
        fig_sc.update_layout(height=300, **LAYOUT)
        ax(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown('<div class="sec">Concentração por Faixa de Valor</div>', unsafe_allow_html=True)
    faixas_ord = ["< R$5k","R$5k–15k","R$15k–30k","R$30k–50k","> R$50k"]
    df_fx = df_fx.set_index("faixa").reindex(faixas_ord).fillna(0).reset_index()
    df_fx["pct"] = (df_fx["qtd"] / df_fx["qtd"].sum() * 100).round(1)

    col_fx1, col_fx2 = st.columns(2)
    with col_fx1:
        fig_fx = px.bar(df_fx, x="faixa", y="qtd", color_discrete_sequence=["#042171"],
                         text=df_fx["pct"].apply(lambda x: f"{x}%"))
        fig_fx.update_layout(height=260, showlegend=False, **LAYOUT)
        fig_fx.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_fx)
        st.plotly_chart(fig_fx, use_container_width=True)

    with col_fx2:
        fig_fx_pie = px.pie(df_fx, values="qtd", names="faixa",
                             color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.45)
        fig_fx_pie.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#031f45"),
                                  legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
        fig_fx_pie.update_traces(textfont_size=9)
        st.plotly_chart(fig_fx_pie, use_container_width=True)

    st.markdown('<div class="sec">Percentis do Valor de Fatura</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns([1, 2])
    with pc1:
        df_pct_tab = pd.DataFrame({"Percentil":list(pcts.keys()), "Valor":[brl(v) for v in pcts.values()]})
        st.dataframe(df_pct_tab, use_container_width=True, hide_index=True)
    with pc2:
        fig_pct = go.Figure(go.Bar(
            x=list(pcts.keys()), y=list(pcts.values()),
            marker_color=["#055cf1","#0638c2","#011027","#60a5fa","#252f3a","#132c49"],
            text=[brl(v) for v in pcts.values()],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_pct.update_layout(height=220, showlegend=False, **LAYOUT)
        ax(fig_pct)
        st.plotly_chart(fig_pct, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# ABA 3 — PAGAMENTOS
# ══════════════════════════════════════════════════════════════
with tab_pag_aba:

    df_hp    = get_hist_pagamento(safras_key)
    df_aging = get_aging(safras_key)
    df_sp    = get_por_safra(safras_key)
    kpis_p   = get_kpis(safras_key)

    kp1,kp2,kp3,kp4 = st.columns(4)
    kp1.metric("Volume Total Pago",  brl(kpis_p["vol_pago"]))
    kp2.metric("Média de Pagamento",  brl(df_hp["valor_pagamento"].mean()))
    kp3.metric("Quantidade de Pgto.", nn(len(df_hp)))
    kp4.metric("Clientes Pagantes",  nn(kpis_p["n_pagantes"]))

    st.markdown("")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown('<div class="sec">Distribuição do Valor de Pagamento</div>', unsafe_allow_html=True)
        fig_ph = px.histogram(df_hp, x="valor_pagamento", nbins=50,
                               color_discrete_sequence=["#125EE1"], marginal="box",
                               labels={"valor_pagamento":"Valor Pago (R$)"})
        fig_ph.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_ph.update_traces(marker_line_width=0)
        ax(fig_ph)
        st.plotly_chart(fig_ph, use_container_width=True)

    with col_p2:
        st.markdown('<div class="sec">Volume Pago por Referência</div>', unsafe_allow_html=True)
        fig_pvol = px.bar(df_sp, x="safra", y="vol_pag",
                           color_discrete_sequence=["#203671"],
                           text=[brl(v) for v in df_sp["vol_pag"]],
                           labels={"vol_pag":"Volume Pago (R$)"})
        fig_pvol.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_pvol.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_pvol)
        st.plotly_chart(fig_pvol, use_container_width=True)

    st.markdown('<div class="sec"> Dias entre Vencimento e Pagamento </div>', unsafe_allow_html=True)
    col_ag1, col_ag2 = st.columns(2)

    with col_ag1:
        fig_ag = px.histogram(df_aging, x="dias_atraso", nbins=60,
                               color_discrete_sequence=["#3964D9"],
                               labels={"dias_atraso":"Dias (negativo = antecipado)"},
                               marginal="box")
        fig_ag.add_vline(x=0, line_dash="dash", line_color="#120456", line_width=1.5)
        fig_ag.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_ag.update_traces(marker_line_width=0)
        ax(fig_ag)
        st.plotly_chart(fig_ag, use_container_width=True)

    with col_ag2:
        n_prazo  = int((df_aging["dias_atraso"] <= 0).sum())
        n_atraso = int((df_aging["dias_atraso"] > 0).sum())
        aging_df = pd.DataFrame({"Status":["No prazo / antecipado","Em atraso"],"Qtd":[n_prazo,n_atraso]})
        fig_ag_pie = px.pie(aging_df, values="Qtd", names="Status",
                             color_discrete_sequence=["#030B42","#0a05a7"], hole=0.5)
        fig_ag_pie.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#385d90"),
                                  legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_ag_pie, use_container_width=True)
    
# ══════════════════════════════════════════════════════════════
# ABA 4 — INADIMPLÊNCIA
# ══════════════════════════════════════════════════════════════
with tab_inad:

    df_sp_i   = get_por_safra(safras_key)
    df_st_i   = get_status_counts(safras_key)
    df_top    = get_top_inad(safras_key)

    n_total  = int(df_st_i["qtd"].sum())
    n_inad   = int(df_st_i.loc[df_st_i["status"] != "Pago no Prazo","qtd"].sum())
    vol_inad = float(df_st_i.loc[df_st_i["status"] != "Pago no Prazo","volume"].sum())
    vol_sp   = float(df_st_i.loc[df_st_i["status"] == "Sem Pagamento","volume"].sum()) if "Sem Pagamento" in df_st_i["status"].values else 0
    taxa_i   = n_inad / n_total * 100 if n_total else 0

    ki1,ki2,ki3,ki4 = st.columns(4)
    ki1.metric("Taxa de Inadimplência",  f"{taxa_i:.1f}%",
               delta=f"{nn(n_inad)} faturas", delta_color="inverse")
    ki2.metric("Clientes em risco",      nn(len(df_top)))
    ki3.metric("Volume em Risco",        brl(vol_inad))
    ki4.metric("Volume Sem Pagamento",   brl(vol_sp))

    st.markdown("")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown('<div class="sec">Status de Pagamento</div>', unsafe_allow_html=True)
        fig_st = px.bar(df_st_i, x="status", y="qtd", color="status",
                         color_discrete_map=STATUS_COLORS, text=df_st_i["qtd"].apply(nn))
        fig_st.update_layout(height=290, showlegend=False, **LAYOUT)
        fig_st.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_st)
        st.plotly_chart(fig_st, use_container_width=True)

    with col_i2:
        st.markdown('<div class="sec">Volume em Risco por Status</div>', unsafe_allow_html=True)
        fig_vst = px.pie(df_st_i, values="volume", names="status",
                          color="status", color_discrete_map=STATUS_COLORS, hole=0.48)
        fig_vst.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#0d4fab"),
                               legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_vst, use_container_width=True)

    st.markdown('<div class="sec">Evolução da Inadimplência por Referência</div>', unsafe_allow_html=True)
    col_is1, col_is2 = st.columns(2)

    with col_is1:
        fig_is = go.Figure()
        fig_is.add_trace(go.Bar(x=df_sp_i["safra"], y=df_sp_i["n_fat"],
                                 name="Total Faturas", marker_color="#0D3BD5"))
        fig_is.add_trace(go.Bar(x=df_sp_i["safra"], y=df_sp_i["inadimplentes"],
                                 name="Inadimplentes", marker_color="#4A0319"))
        fig_is.update_layout(height=250, barmode="overlay",
                              legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"), **LAYOUT)
        ax(fig_is)
        st.plotly_chart(fig_is, use_container_width=True)

    with col_is2:
        max_t = float(df_sp_i["taxa_inad"].max()) if len(df_sp_i) else 10
        fig_taxa = go.Figure(go.Scatter(
            x=df_sp_i["safra"], y=df_sp_i["taxa_inad"],
            mode="lines+markers+text",
            text=[f"{v}%" for v in df_sp_i["taxa_inad"]],
            textposition="top center", textfont=dict(size=10, color="#a80923"),
            line=dict(color="#6c0516", width=2.5), marker=dict(size=9, color="#5e0413"),
            fill="tozeroy", fillcolor="rgba(244,63,94,0.07)",
        ))
        fig_taxa.update_layout(height=250,
                                yaxis=dict(ticksuffix="%", range=[0, max_t*1.3+5]), **LAYOUT)
        ax(fig_taxa)
        st.plotly_chart(fig_taxa, use_container_width=True)

    # Top inadimplentes — clicável → navega para perfil do cliente
    st.markdown('<div class="sec">Top 20 · Maior Volume em Risco · <span style="color:#60a5fa;font-weight:400;font-size:0.8rem;">clique na linha para ver o perfil</span></div>', unsafe_allow_html=True)

    df_top_fmt = df_top.copy()
    df_top_fmt["vol_risco"] = df_top_fmt["vol_risco"].apply(brl)
    df_top_fmt.columns = ["ID Cliente","Faturas Inad.","Volume em Risco","Status Principal"]

    ev_inad = st.dataframe(
        df_top_fmt,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if ev_inad.selection.rows:
        idx = ev_inad.selection.rows[0]
        cli_clicado = int(df_top.iloc[idx]["id_cliente"])
        st.session_state["cliente_selecionado"] = cli_clicado
        st.success(f"✅ Cliente **#{cli_clicado}** selecionado. Acesse **🔍 Perfil do Cliente** para o histórico completo.")

    # Download
    df_exp = get_aging(safras_key).copy()
    buf_inad = io.BytesIO()
    df_exp.to_excel(buf_inad, index=False, engine="openpyxl")
    st.download_button("⬇️ Exportar análise de inadimplência",
                       buf_inad.getvalue(), "inadimplencia_trusted.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════
# ABA 5 — QUALIDADE DOS DADOS
# ══════════════════════════════════════════════════════════════
with tab_quality:

    st.markdown("""
    <div class='info-box'>
      <div class='title'> Dados da Camada Trusted</div>
      <div class='body'>
        <b style='color:#6b7280'>Tipagem:</b> datas em DATE · valores em DOUBLE · IDs em BIGINT &nbsp;·&nbsp;
        <b style='color:#6b7280'>Deduplicação:</b> por (id_fatura, id_cliente) &nbsp;·&nbsp;
        <b style='color:#6b7280'>Particionamento:</b> ref = YYYYMM &nbsp;·&nbsp;
        <b style='color:#6b7280'>Engine:</b> PySpark → Parquet (snappy) &nbsp;·&nbsp;
        <b style='color:#6b7280'>Ingestão:</b> S3 → Raw → Trusted
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_nf, df_np = get_nulls(safras_key)
    for df_n in [df_nf, df_np]:
        df_n["% Nulo"] = (df_n["nulos"] / df_n["total"] * 100).round(2)
        df_n["Status"] = df_n["% Nulo"].apply(
            lambda x: "✅ OK" if x == 0 else ("⚠️ Atenção" if x < 5 else "❌ Crítico")
        )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown('<div class="sec">Diagnóstico de Nulos · tb_faturas</div>', unsafe_allow_html=True)
        st.dataframe(df_nf.rename(columns={"col":"Coluna","nulos":"Nulos","total":"Total"}),
                     use_container_width=True, hide_index=True)
    with col_q2:
        st.markdown('<div class="sec">Diagnóstico de Nulos · tb_pagamentos</div>', unsafe_allow_html=True)
        st.dataframe(df_np.rename(columns={"col":"Coluna","nulos":"Nulos","total":"Total"}),
                     use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">Verificação de Duplicatas</div>', unsafe_allow_html=True)
    dup_fat, dup_pag = get_duplicatas(safras_key)
    dc1, dc2 = st.columns(2)
    dc1.metric("Duplicatas · tb_faturas",    "✅ Nenhuma" if dup_fat == 0 else f"⚠️ {dup_fat}")
    dc2.metric("Duplicatas · tb_pagamentos", "✅ Nenhuma" if dup_pag == 0 else f"⚠️ {dup_pag}")

    st.markdown('<div class="sec">Integridade Referencial · Faturas sem Pagamento</div>', unsafe_allow_html=True)
    df_ref = get_ref_integridade(safras_key)
    fig_ref = go.Figure()
    fig_ref.add_trace(go.Bar(x=df_ref["safra"], y=df_ref["total_fat"],
                              name="Total Faturas", marker_color="#172554"))
    fig_ref.add_trace(go.Bar(x=df_ref["safra"], y=df_ref["sem_pag"],
                              name="Sem Pagamento", marker_color="#4d030f"))
    fig_ref.add_trace(go.Scatter(x=df_ref["safra"], y=df_ref["pct_sem_pag"],
                                  name="% Sem Pag.", yaxis="y2",
                                  line=dict(color="#22045f", width=2), mode="lines+markers"))
    fig_ref.update_layout(
        height=280, barmode="overlay",
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    ticksuffix="%", tickfont=dict(color="#140442", size=10), range=[0, 130]),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"), **LAYOUT,
    )
    ax(fig_ref)
    st.plotly_chart(fig_ref, use_container_width=True)

    st.markdown('<div class="sec"> Camada Trusted </div>', unsafe_allow_html=True)
    inv = pd.DataFrame([
        {"Tabela":"tb_faturas",    "Safras":", ".join(safras_sel), "Formato":"Parquet (snappy)","Partição":"ref","Engine":"DuckDB"},
        {"Tabela":"tb_pagamentos", "Safras":", ".join(safras_sel), "Formato":"Parquet (snappy)","Partição":"ref","Engine":"DuckDB"},
    ])
    st.dataframe(inv, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# ABA 6 — PERFIL DO CLIENTE
# Query SQL parametrizada — resposta em <10ms via DuckDB
# ══════════════════════════════════════════════════════════════
with tab_cliente:

    st.markdown('<div class="sec">Consulta Individual por Cliente</div>', unsafe_allow_html=True)

    lista_clientes = get_lista_clientes()

    if "cliente_selecionado" not in st.session_state:
        st.session_state["cliente_selecionado"] = lista_clientes[0] if lista_clientes else 1

    # Garante que o valor do session_state é válido
    if st.session_state["cliente_selecionado"] not in lista_clientes:
        st.session_state["cliente_selecionado"] = lista_clientes[0]

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

    with col_ctrl1:
        id_cli = st.selectbox(
            "Selecione o cliente",
            options=lista_clientes,
            index=lista_clientes.index(st.session_state["cliente_selecionado"]),
            format_func=lambda x: f"Cliente #{x}",
        )
        st.session_state["cliente_selecionado"] = id_cli

    with col_ctrl2:
        id_manual = st.number_input(
            "ou digite o ID",
            min_value=int(min(lista_clientes)),
            max_value=int(max(lista_clientes)),
            value=int(id_cli),
            step=1,
        )
        if id_manual in lista_clientes and id_manual != id_cli:
            st.session_state["cliente_selecionado"] = id_manual
            st.rerun()

    with col_ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Aleatório ", use_container_width=True):
            st.session_state["cliente_selecionado"] = random.choice(lista_clientes)
            st.rerun()

    id_cli = st.session_state["cliente_selecionado"]

    # Queries parametrizadas — DuckDB resolve em milissegundos
    df_cli_fat    = get_cliente_fat(id_cli)
    df_cli_pag    = get_cliente_pag(id_cli)
    df_cli_status = get_cliente_status(id_cli)

    if df_cli_fat.empty:
        st.warning(f"Cliente {id_cli} não encontrado.")
    else:
        status_cli  = df_cli_status.loc[df_cli_status["qtd"].idxmax(), "status"] if not df_cli_status.empty else "—"
        cor_status  = STATUS_COLORS.get(status_cli, "#eef0f4")
        vol_fat_cli = float(df_cli_fat["valor_fatura"].sum())
        vol_pag_cli = float(df_cli_pag["valor_pagamento"].sum()) if not df_cli_pag.empty else 0.0
        saldo_cli   = vol_fat_cli - vol_pag_cli
        pct_pago    = (vol_pag_cli / vol_fat_cli * 100) if vol_fat_cli else 0

        st.markdown(f"""
        <div style='background:#0c1222;border:1px solid #141e38;border-radius:10px;
                    padding:14px 20px;display:flex;gap:2.5rem;align-items:center;margin-bottom:1rem;'>
          <div>
            <div style='color:#374151;font-size:0.65rem;text-transform:uppercase;letter-spacing:.12em;'>Cliente</div>
            <div style='font-family:IBM Plex Mono,monospace;color:#e2e8f0;font-size:1.3rem;font-weight:600;'>#{id_cli}</div>
          </div>
          <div>
            <div style='color:#374151;font-size:0.65rem;text-transform:uppercase;letter-spacing:.12em;'>Status Principal</div>
            <div style='color:{cor_status};font-weight:600;font-size:0.9rem;'>{status_cli}</div>
          </div>
          <div>
            <div style='color:#374151;font-size:0.65rem;text-transform:uppercase;letter-spacing:.12em;'>Faturas</div>
            <div style='font-family:IBM Plex Mono,monospace;color:#e2e8f0;font-size:1rem;'>{len(df_cli_fat)}</div>
          </div>
          <div>
            <div style='color:#374151;font-size:0.65rem;text-transform:uppercase;letter-spacing:.12em;'>Pagamentos</div>
            <div style='font-family:IBM Plex Mono,monospace;color:#e2e8f0;font-size:1rem;'>{len(df_cli_pag)}</div>
          </div>
          <div>
            <div style='color:#374151;font-size:0.65rem;text-transform:uppercase;letter-spacing:.12em;'>% Pago</div>
            <div style='font-family:IBM Plex Mono,monospace;color:#e2e8f0;font-size:1rem;'>{pct_pago:.1f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ck1, ck2, ck3, ck4 = st.columns(4)
        ck1.metric("Volume Faturado", brl(vol_fat_cli))
        ck2.metric("Volume Pago",     brl(vol_pag_cli))
        ck3.metric("Saldo em Aberto", brl(saldo_cli),
                   delta="em risco" if saldo_cli > 0 else "quitado", delta_color="inverse")
        ck4.metric("% Pago",          f"{pct_pago:.1f}%")

        col_cl1, col_cl2 = st.columns(2)

        with col_cl1:
            st.markdown("**Histórico de Faturas**")
            df_cf_fmt = df_cli_fat.copy()
            df_cf_fmt["valor_fatura"]           = df_cf_fmt["valor_fatura"].apply(brl)
            df_cf_fmt["valor_pagamento_minimo"]  = df_cf_fmt["valor_pagamento_minimo"].apply(brl)
            df_cf_fmt.columns = ["Safra","Emissão","Vencimento","Fatura","Pgto Mínimo"]
            st.dataframe(df_cf_fmt, use_container_width=True, hide_index=True)

        with col_cl2:
            st.markdown("**Histórico de Pagamentos**")
            if df_cli_pag.empty:
                st.info("Nenhum pagamento registrado.")
            else:
                df_cp_fmt = df_cli_pag[["data_pagamento","valor_pagamento"]].copy()
                df_cp_fmt["valor_pagamento"] = df_cp_fmt["valor_pagamento"].apply(brl)
                df_cp_fmt.columns = ["Data","Valor Pago"]
                st.dataframe(df_cp_fmt, use_container_width=True, hide_index=True)

        # Gráficos do cliente
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            if not df_cli_status.empty:
                st.markdown("**Status das Faturas**")
                fig_cs = px.pie(df_cli_status, values="qtd", names="status",
                                 color="status", color_discrete_map=STATUS_COLORS, hole=0.5)
                fig_cs.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#dfd4d8"),
                                      legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                                      margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cs, use_container_width=True)

        with col_g2:
            st.markdown("**Fatura vs Pagamento por Safra**")
            fig_tl = go.Figure()
            df_fat_g = df_cli_fat.groupby("safra")["valor_fatura"].sum().reset_index()
            fig_tl.add_trace(go.Bar(x=df_fat_g["safra"], y=df_fat_g["valor_fatura"],
                                     name="Fatura", marker_color="#83aee2",
                                     text=[brl(v) for v in df_fat_g["valor_fatura"]],
                                     textposition="outside", textfont=dict(size=9)))
            if not df_cli_pag.empty:
                df_pag_g = (df_cli_pag
                            .assign(safra=df_cli_pag["data_pagamento"].astype(str).str[:7])
                            .groupby("safra")["valor_pagamento"].sum().reset_index())
                fig_tl.add_trace(go.Bar(x=df_pag_g["safra"], y=df_pag_g["valor_pagamento"],
                                         name="Pago", marker_color="#090241",
                                         text=[brl(v) for v in df_pag_g["valor_pagamento"]],
                                         textposition="outside", textfont=dict(size=9)))
            fig_tl.update_layout(height=240, barmode="group",
                                  legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                                  **LAYOUT)
            ax(fig_tl)
            st.plotly_chart(fig_tl, use_container_width=True)

        # Export
        buf_cli = io.BytesIO()
        with pd.ExcelWriter(buf_cli, engine="openpyxl") as wr:
            df_cli_fat.to_excel(wr, sheet_name="Faturas", index=False)
            if not df_cli_pag.empty:
                df_cli_pag.to_excel(wr, sheet_name="Pagamentos", index=False)
        st.download_button(f"⬇️ Exportar perfil — Cliente #{id_cli}",
                           buf_cli.getvalue(), f"cliente_{id_cli}_trusted.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
