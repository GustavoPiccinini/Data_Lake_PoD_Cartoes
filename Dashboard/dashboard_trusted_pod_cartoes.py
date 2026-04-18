"""
Dashboard — Camada Trusted · PoD Cartões
Projeto: Data Lake 

Dados disponiveis para analise .
Execute: streamlit run dashboard_trusted_pod_cartoes.py
"""

import io
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
# LEITURA DOS DADOS DO DATA LAKE
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    import os
    base_dir = os.path.dirname(__file__)  # garante que pega a pasta do .py
    base_fat = os.path.join(base_dir, "tb_faturas")
    base_pag = os.path.join(base_dir, "tb_pagamentos")

    # Lista todas as pastas safra=YYYY-MM
    safras_fat = [d for d in os.listdir(base_fat) if d.startswith("safra=")]
    safras_pag = [d for d in os.listdir(base_pag) if d.startswith("safra=")]

    # Carrega faturas
    fat = pd.concat([
        pd.read_parquet(os.path.join(base_fat, safra, file)).assign(safra=safra.replace("safra=", ""))
        for safra in safras_fat
        for file in os.listdir(os.path.join(base_fat, safra))
        if file.endswith(".parquet")
    ])

    # Carrega pagamentos
    pag = pd.concat([
        pd.read_parquet(os.path.join(base_pag, safra, file)).assign(safra=safra.replace("safra=", ""))
        for safra in safras_pag
        for file in os.listdir(os.path.join(base_pag, safra))
        if file.endswith(".parquet")
    ])

    # Ajusta tipos de data
    fat["data_emissao"]    = pd.to_datetime(fat["data_emissao"])
    fat["data_vencimento"] = pd.to_datetime(fat["data_vencimento"])
    pag["data_pagamento"]  = pd.to_datetime(pag["data_pagamento"])

    return fat, pag

df_fat, df_pag = load_data()

# ══════════════════════════════════════════════════════════════
# JOIN COMPLETO
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_join(fat, pag):
    df = fat.merge(
        pag[["id_fatura","id_cliente","data_pagamento","valor_pagamento","safra"]],
        on=["id_fatura","id_cliente","safra"], how="left"
    )
    df["dias_atraso"] = (df["data_pagamento"] - df["data_vencimento"]).dt.days
    df["status"] = np.where(df["data_pagamento"].isna(), "Sem Pagamento",
                   np.where(df["dias_atraso"] > 0, "Pago em Atraso",
                   np.where(df["valor_pagamento"] < df["valor_pagamento_minimo"],
                            "Abaixo do Mínimo", "Pago no Prazo")))
    return df

df_join = build_join(df_fat, df_pag)

# ══════════════════════════════════════════════════════════════
# ESTILO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

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
.b-red   { background:#3b0764; color:#e879f9; border:1px solid #6b21a8; }
.b-amber { background:#431407; color:#fb923c; border:1px solid #9a3412; }

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
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if abs(v) >= 1e6: return f"R$ {v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"R$ {v/1e3:.1f}k"
    return f"R$ {v:,.2f}"

def nn(v):
    if v is None: return "—"
    return f"{int(v):,}"

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#4b5563"),
    margin=dict(l=55, r=20, t=35, b=45),
)
def ax(fig, y_prefix=""):
    fig.update_xaxes(showgrid=False, tickfont=dict(color="#4b5563", size=10), linecolor="#0f1629")
    fig.update_yaxes(showgrid=True, gridcolor="#0f1629", tickfont=dict(color="#4b5563", size=10),
                     zeroline=False, tickprefix=y_prefix)
    return fig

STATUS_COLORS = {
    "Pago no Prazo":    "#4a74de",
    "Pago em Atraso":   "#b83cfb",
    "Abaixo do Mínimo": "#ba24fb",
    "Sem Pagamento":    "#7b3ff4",
}

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
        Data Lake 2024 · Engenharia de Dados Jr
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
        Leitura via <b style='color:#6b7280'>DuckDB + Parquet</b>.<br>
        Ingestão: <b style='color:#6b7280'>S3 → PySpark → Trusted</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📅 Safras disponíveis")
    safras_disp = sorted(df_fat["safra"].unique().tolist())
    safras_sel  = st.multiselect("", options=safras_disp, default=safras_disp)
    if not safras_sel:
        safras_sel = safras_disp

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem;color:#1f2937;line-height:1.9;'>
      <b style='color:#374151'>Tabelas:</b> tb_faturas · tb_pagamentos<br>
      <b style='color:#374151'>Registros:</b> 10.000 fat · 5.780 pag<br>
      <b style='color:#374151'>Clientes:</b> 10.000<br>
      <b style='color:#374151'>Safra ref.:</b> 2023-01<br>
      <b style='color:#374151'>Data ref.:</b> 2024-02-01<br>
      <b style='color:#374151'>Fonte:</b> S3 · sa-east-1
    </div>
    """, unsafe_allow_html=True)

# ── Filtra pelo multiselect ──
df_fat_f  = df_fat[df_fat["safra"].isin(safras_sel)]
df_pag_f = df_pag[df_pag["safra"].isin(safras_sel)]
df_join_f = df_join[df_join["safra"].isin(safras_sel)]

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
hc1, hc2 = st.columns([4, 1])
with hc1:
    st.markdown("""
    <h1 style='font-family:IBM Plex Mono,monospace;font-size:1.65rem;margin-bottom:2px;letter-spacing:-0.01em;'>
      PoD Cartões &mdash; Camada Trusted
    </h1>
    <p style='color:#1f2937;font-size:0.8rem;margin:0;'>
      Dados financeiros de faturas e pagamentos · Projeto Data Lake 
    </p>
    """, unsafe_allow_html=True)
with hc2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:12px;'>
      <span class='badge b-blue'>Trusted Layer</span>&nbsp;
      <span class='badge b-green'>safra 2023-01</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# ABAS
# ══════════════════════════════════════════════════════════════
tab_exec, tab_fat, tab_pag, tab_inad, tab_quality, tab_cliente = st.tabs([
    "🏠 Visão Geral",
    "📄 Faturas",
    "💸 Pagamentos",
    "🚨 Inadimplência",
    "🔬 Qualidade de Dados",
    "🔍 Perfil do Cliente",
])

# ══════════════════════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════
with tab_exec:

    # KPIs
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    vol_fat  = df_fat_f["valor_fatura"].sum()
    vol_pag  = df_pag_f["valor_pagamento"].sum()
    pct_pag  = (df_join_f["data_pagamento"].notna().sum() / len(df_join_f) * 100) if len(df_join_f) else 0
    inad_pct = (df_join_f["status"].isin(["Sem Pagamento","Pago em Atraso","Abaixo do Mínimo"]).sum()
                / len(df_join_f) * 100) if len(df_join_f) else 0

    k1.metric("Clientes",         nn(df_fat_f["id_cliente"].nunique()))
    k2.metric("Faturas emitidas", nn(len(df_fat_f)))
    k3.metric("Volume faturado",  brl(vol_fat))
    k4.metric("Volume pago",      brl(vol_pag))
    k5.metric("% Clientes pagaram", f"{pct_pag:.1f}%")
    k6.metric("Taxa inadimplência", f"{inad_pct:.1f}%", delta_color="inverse",
              delta="⚠️ atenção" if inad_pct > 40 else "✅ controlada")

    st.markdown("")

    # Volume faturado vs pago por safra
    st.markdown('<div class="sec">Volume Faturado vs Pago · por Safra</div>', unsafe_allow_html=True)
    df_safra = (
        df_fat_f.groupby("safra")
        .agg(vol_fat=("valor_fatura","sum"), n_fat=("id_fatura","count"), n_cli=("id_cliente","nunique"))
        .reset_index()
    )
    df_pag_safra = (
        df_pag_f.assign(safra=df_pag_f["data_pagamento"].dt.to_period("M").astype(str))
        .groupby("safra")
        .agg(vol_pag=("valor_pagamento","sum"))
        .reset_index()
    )
    df_ev = df_safra.merge(df_pag_safra, on="safra", how="left").fillna(0)
    df_ev["cobertura_pct"] = (df_ev["vol_pag"] / df_ev["vol_fat"] * 100).round(1)

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df_ev["safra"], y=df_ev["vol_fat"],
                                  name="Faturado", marker_color="#1d4ed8",
                                  text=[brl(v) for v in df_ev["vol_fat"]],
                                  textposition="outside", textfont=dict(size=9,color="#4b5563")))
        fig_vol.add_trace(go.Bar(x=df_ev["safra"], y=df_ev["vol_pag"],
                                  name="Pago", marker_color="#4ade80",
                                  text=[brl(v) for v in df_ev["vol_pag"]],
                                  textposition="outside", textfont=dict(size=9,color="#4b5563")))
        fig_vol.update_layout(height=290, barmode="group", title_text="Volume (R$)",
                               title_font=dict(size=12,color="#6b7280"),
                               legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"), **LAYOUT)
        ax(fig_vol)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_ev2:
        fig_cob = go.Figure()
        fig_cob.add_trace(go.Bar(x=df_ev["safra"], y=df_ev["cobertura_pct"],
                                  marker_color="#7c3aed",
                                  text=[f"{v}%" for v in df_ev["cobertura_pct"]],
                                  textposition="outside", textfont=dict(size=9,color="#4b5563")))
        fig_cob.add_hline(y=100, line_dash="dash", line_color="#1f2937", line_width=1)
        fig_cob.update_layout(height=290, title_text="% Volume Pago / Faturado",
                               title_font=dict(size=12,color="#6b7280"),
                               yaxis_range=[0,115], showlegend=False, **LAYOUT)
        ax(fig_cob)
        st.plotly_chart(fig_cob, use_container_width=True)

    # Status geral
    st.markdown('<div class="sec">Distribuição de Status de Pagamento</div>', unsafe_allow_html=True)
    col_st1, col_st2 = st.columns([1,2])
    status_vc = df_join_f["status"].value_counts().reset_index()
    status_vc.columns = ["Status","Qtd"]
    status_vc["Pct"] = (status_vc["Qtd"] / status_vc["Qtd"].sum() * 100).round(1)
    status_vc["Volume"] = status_vc["Status"].map(
        df_join_f.groupby("status")["valor_fatura"].sum().to_dict()
    )

    with col_st1:
        fig_pie = px.pie(status_vc, values="Qtd", names="Status",
                         color="Status", color_discrete_map=STATUS_COLORS, hole=0.52)
        fig_pie.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#4b5563"),
                               legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
        fig_pie.update_traces(textfont_size=10)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_st2:
        fig_bar_st = px.bar(status_vc, x="Status", y="Qtd",
                             color="Status", color_discrete_map=STATUS_COLORS,
                             text=status_vc["Pct"].apply(lambda x: f"{x}%"))
        fig_bar_st.update_layout(height=280, showlegend=False, **LAYOUT)
        fig_bar_st.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_bar_st)
        st.plotly_chart(fig_bar_st, use_container_width=True)

    # Tabela resumo
    st.markdown('<div class="sec">Resumo por Safra</div>', unsafe_allow_html=True)
    df_resumo = df_ev.copy()
    df_resumo["vol_fat"]  = df_resumo["vol_fat"].apply(brl)
    df_resumo["vol_pag"]  = df_resumo["vol_pag"].apply(brl)
    df_resumo["n_fat"]    = df_resumo["n_fat"].apply(nn)
    df_resumo["n_cli"]    = df_resumo["n_cli"].apply(nn)
    df_resumo["cobertura_pct"] = df_resumo["cobertura_pct"].apply(lambda x: f"{x}%")
    df_resumo.columns = ["Safra","Vol Faturado","Faturas","Clientes","Vol Pago","% Pago"]
    st.dataframe(df_resumo, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# ABA 2 — FATURAS
# ══════════════════════════════════════════════════════════════
with tab_fat:

    kf1,kf2,kf3,kf4 = st.columns(4)
    kf1.metric("Ticket Médio",   brl(df_fat_f["valor_fatura"].mean()))
    kf2.metric("Ticket Mediana", brl(df_fat_f["valor_fatura"].median()))
    kf3.metric("Maior Fatura",   brl(df_fat_f["valor_fatura"].max()))
    kf4.metric("Pgto Mínimo Med.", brl(df_fat_f["valor_pagamento_minimo"].median()))

    st.markdown("")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown('<div class="sec">Distribuição do Valor de Fatura</div>', unsafe_allow_html=True)
        fig_hf = px.histogram(df_fat_f, x="valor_fatura", nbins=50,
                               color_discrete_sequence=["#1d4ed8"], marginal="box",
                               labels={"valor_fatura":"Valor Fatura (R$)"})
        fig_hf.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_hf.update_traces(marker_line_width=0)
        ax(fig_hf)
        st.plotly_chart(fig_hf, use_container_width=True)

    with col_f2:
        st.markdown('<div class="sec">Fatura vs Pagamento Mínimo</div>', unsafe_allow_html=True)
        df_sc = df_fat_f[["valor_fatura","valor_pagamento_minimo"]].dropna().sample(min(1500,len(df_fat_f)), random_state=42)
        fig_sc = px.scatter(df_sc, x="valor_fatura", y="valor_pagamento_minimo",
                             color_discrete_sequence=["#60a5fa"], opacity=0.35,
                             labels={"valor_fatura":"Fatura (R$)","valor_pagamento_minimo":"Pgto Mínimo (R$)"})
        fig_sc.update_traces(marker=dict(size=3))
        fig_sc.update_layout(height=300, **LAYOUT)
        ax(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Faixas de valor
    st.markdown('<div class="sec">Concentração por Faixa de Valor</div>', unsafe_allow_html=True)
    bins   = [0,5000,15000,30000,50000,float("inf")]
    labels_fx = ["< R$5k","R$5k–15k","R$15k–30k","R$30k–50k","> R$50k"]
    df_fat_f2 = df_fat_f.copy()
    df_fat_f2["faixa"] = pd.cut(df_fat_f2["valor_fatura"], bins=bins, labels=labels_fx)
    df_fx = df_fat_f2["faixa"].value_counts().reindex(labels_fx).fillna(0).reset_index()
    df_fx.columns = ["Faixa","Qtd"]
    df_fx["Pct"] = (df_fx["Qtd"] / df_fx["Qtd"].sum() * 100).round(1)

    col_fx1, col_fx2 = st.columns(2)
    with col_fx1:
        fig_fx = px.bar(df_fx, x="Faixa", y="Qtd",
                         color_discrete_sequence=["#1d4ed8"],
                         text=df_fx["Pct"].apply(lambda x: f"{x}%"))
        fig_fx.update_layout(height=260, showlegend=False, **LAYOUT)
        fig_fx.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_fx)
        st.plotly_chart(fig_fx, use_container_width=True)

    with col_fx2:
        fig_fx_pie = px.pie(df_fx, values="Qtd", names="Faixa",
                             color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.45)
        fig_fx_pie.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#4b5563"),
                                  legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
        fig_fx_pie.update_traces(textfont_size=9)
        st.plotly_chart(fig_fx_pie, use_container_width=True)

    # Percentis
    st.markdown('<div class="sec">Percentis do Valor de Fatura</div>', unsafe_allow_html=True)
    pcts = {f"P{int(q*100)}": round(df_fat_f["valor_fatura"].quantile(q),2)
            for q in [.10,.25,.50,.75,.90,.99]}
    df_pct_tab = pd.DataFrame({"Percentil": list(pcts.keys()), "Valor": [brl(v) for v in pcts.values()]})
    pc1, pc2 = st.columns([1,2])
    with pc1:
        st.dataframe(df_pct_tab, use_container_width=True, hide_index=True)
    with pc2:
        fig_pct = go.Figure(go.Bar(
            x=list(pcts.keys()), y=list(pcts.values()),
            marker_color=["#1e3a5f","#1d4ed8","#3b82f6","#60a5fa","#93c5fd","#bfdbfe"],
            text=[brl(v) for v in pcts.values()],
            textposition="outside", textfont=dict(size=9,color="#4b5563"),
        ))
        fig_pct.update_layout(height=220, showlegend=False, **LAYOUT)
        ax(fig_pct)
        st.plotly_chart(fig_pct, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 3 — PAGAMENTOS
# ══════════════════════════════════════════════════════════════
with tab_pag:

    kp1,kp2,kp3,kp4 = st.columns(4)
    kp1.metric("Volume Total Pago",  brl(df_pag_f["valor_pagamento"].sum()))
    kp2.metric("Mediana Pagamento",  brl(df_pag_f["valor_pagamento"].median()))
    kp3.metric("Registros de Pgto.", nn(len(df_pag_f)))
    kp4.metric("Clientes Pagantes",  nn(df_pag_f["id_cliente"].nunique()))

    st.markdown("")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown('<div class="sec">Distribuição do Valor de Pagamento</div>', unsafe_allow_html=True)
        fig_ph = px.histogram(df_pag_f, x="valor_pagamento", nbins=50,
                               color_discrete_sequence=["#4ade80"], marginal="box",
                               labels={"valor_pagamento":"Valor Pago (R$)"})
        fig_ph.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_ph.update_traces(marker_line_width=0)
        ax(fig_ph)
        st.plotly_chart(fig_ph, use_container_width=True)

    with col_p2:
        st.markdown('<div class="sec">Volume Pago por Safra</div>', unsafe_allow_html=True)
        df_pvol = (df_pag_f.assign(safra=df_pag_f["data_pagamento"].dt.to_period("M").astype(str))
                   .groupby("safra").agg(vol=("valor_pagamento","sum"),n=("id_cliente","nunique")).reset_index())
        fig_pvol = px.bar(df_pvol, x="safra", y="vol",
                           color_discrete_sequence=["#4ade80"],
                           text=[brl(v) for v in df_pvol["vol"]],
                           labels={"vol":"Volume Pago (R$)"})
        fig_pvol.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_pvol.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_pvol)
        st.plotly_chart(fig_pvol, use_container_width=True)

    # Aging
    st.markdown('<div class="sec">Aging — Dias entre Vencimento e Pagamento</div>', unsafe_allow_html=True)
    df_aging = df_join_f[df_join_f["dias_atraso"].notna()].copy()
    col_ag1, col_ag2 = st.columns(2)

    with col_ag1:
        fig_ag = px.histogram(df_aging, x="dias_atraso", nbins=60,
                               color_discrete_sequence=["#f59e0b"],
                               labels={"dias_atraso":"Dias (negativo = antecipado)"},
                               marginal="box")
        fig_ag.add_vline(x=0, line_dash="dash", line_color="#f43f5e", line_width=1.5)
        fig_ag.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_ag.update_traces(marker_line_width=0)
        ax(fig_ag)
        st.plotly_chart(fig_ag, use_container_width=True)

    with col_ag2:
        n_prazo  = (df_aging["dias_atraso"] <= 0).sum()
        n_atraso = (df_aging["dias_atraso"] > 0).sum()
        n_antec  = (df_aging["dias_atraso"] < 0).sum()
        aging_df = pd.DataFrame({
            "Status": ["No prazo / antecipado","Em atraso"],
            "Qtd":    [int(n_prazo), int(n_atraso)],
        })
        fig_ag_pie = px.pie(aging_df, values="Qtd", names="Status",
                             color_discrete_sequence=["#4ade80","#f43f5e"], hole=0.5)
        fig_ag_pie.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#4b5563"),
                                  legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_ag_pie, use_container_width=True)

        ag_med = df_aging["dias_atraso"].median()
        ag_p75 = df_aging["dias_atraso"].quantile(0.75)
        st.metric("Mediana aging (dias)", f"{ag_med:.0f}")
        st.metric("P75 aging (dias)",     f"{ag_p75:.0f}")


# ══════════════════════════════════════════════════════════════
# ABA 4 — INADIMPLÊNCIA
# ══════════════════════════════════════════════════════════════
with tab_inad:

    n_total   = len(df_join_f)
    n_inad    = df_join_f["status"].isin(["Sem Pagamento","Pago em Atraso","Abaixo do Mínimo"]).sum()
    pct_inad  = n_inad / n_total * 100 if n_total else 0
    vol_inad  = df_join_f.loc[df_join_f["status"] != "Pago no Prazo", "valor_fatura"].sum()
    cli_inad  = df_join_f.loc[df_join_f["status"] != "Pago no Prazo", "id_cliente"].nunique()
    vol_sp    = df_join_f.loc[df_join_f["status"] == "Sem Pagamento", "valor_fatura"].sum()

    ki1,ki2,ki3,ki4 = st.columns(4)
    ki1.metric("Taxa de Inadimplência",  f"{pct_inad:.1f}%",
               delta=f"{nn(n_inad)} faturas", delta_color="inverse")
    ki2.metric("Clientes Inadimplentes", nn(cli_inad))
    ki3.metric("Volume em Risco",        brl(vol_inad))
    ki4.metric("Volume Sem Pagamento",   brl(vol_sp))

    st.markdown("")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown('<div class="sec">Status de Pagamento — Todas as Faturas</div>', unsafe_allow_html=True)
        df_st = df_join_f["status"].value_counts().reset_index()
        df_st.columns = ["Status","Qtd"]
        df_st["Volume"] = df_st["Status"].map(df_join_f.groupby("status")["valor_fatura"].sum().to_dict())
        fig_st = px.bar(df_st, x="Status", y="Qtd",
                         color="Status", color_discrete_map=STATUS_COLORS,
                         text=df_st["Qtd"].apply(nn))
        fig_st.update_layout(height=300, showlegend=False, **LAYOUT)
        fig_st.update_traces(textposition="outside", marker_line_width=0)
        ax(fig_st)
        st.plotly_chart(fig_st, use_container_width=True)

    with col_i2:
        st.markdown('<div class="sec">Volume em Risco por Status</div>', unsafe_allow_html=True)
        fig_vst = px.pie(df_st, values="Volume", names="Status",
                          color="Status", color_discrete_map=STATUS_COLORS, hole=0.48)
        fig_vst.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#4b5563"),
                               legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_vst, use_container_width=True)

    # Evolução por safra
    st.markdown('<div class="sec">Evolução da Inadimplência por Safra</div>', unsafe_allow_html=True)
    df_inad_safra = (
        df_join_f.groupby("safra")
        .apply(lambda g: pd.Series({
            "total":        len(g),
            "inadimplentes": (g["status"] != "Pago no Prazo").sum(),
            "vol_risco":    g.loc[g["status"] != "Pago no Prazo","valor_fatura"].sum(),
        }))
        .reset_index()
    )
    df_inad_safra["taxa"] = (df_inad_safra["inadimplentes"] / df_inad_safra["total"] * 100).round(1)

    col_is1, col_is2 = st.columns(2)
    with col_is1:
        fig_is = go.Figure()
        fig_is.add_trace(go.Bar(x=df_inad_safra["safra"], y=df_inad_safra["total"],
                                 name="Total Faturas", marker_color="#172554"))
        fig_is.add_trace(go.Bar(x=df_inad_safra["safra"], y=df_inad_safra["inadimplentes"],
                                 name="Inadimplentes", marker_color="#f43f5e"))
        fig_is.update_layout(height=260, barmode="overlay",
                              legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"), **LAYOUT)
        ax(fig_is)
        st.plotly_chart(fig_is, use_container_width=True)

    with col_is2:
        fig_taxa = go.Figure(go.Scatter(
            x=df_inad_safra["safra"], y=df_inad_safra["taxa"],
            mode="lines+markers+text",
            text=[f"{v}%" for v in df_inad_safra["taxa"]],
            textposition="top center", textfont=dict(size=10,color="#f43f5e"),
            line=dict(color="#f43f5e",width=2.5),
            marker=dict(size=9,color="#f43f5e"),
            fill="tozeroy", fillcolor="rgba(244,63,94,0.07)",
        ))
        fig_taxa.update_layout(height=260,
                                yaxis=dict(ticksuffix="%",range=[0,max(df_inad_safra["taxa"])*1.3+5]),
                                **LAYOUT)
        ax(fig_taxa)
        st.plotly_chart(fig_taxa, use_container_width=True)

    # Top inadimplentes
    st.markdown('<div class="sec">Top 20 Clientes · Maior Volume em Risco</div>', unsafe_allow_html=True)
    df_top = (
        df_join_f[df_join_f["status"] != "Pago no Prazo"]
        .groupby("id_cliente")
        .agg(Faturas=("id_fatura","count"),
             Volume_Risco=("valor_fatura","sum"),
             Status=("status", lambda x: x.mode()[0]))
        .sort_values("Volume_Risco", ascending=False).head(20).reset_index()
    )
    df_top["Volume_Risco"] = df_top["Volume_Risco"].apply(brl)
    df_top.columns = ["ID Cliente","Faturas Inad.","Volume em Risco","Status Principal"]
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    # Download
    df_exp = df_join_f[["id_fatura","id_cliente","safra","data_vencimento",
                          "valor_fatura","valor_pagamento","dias_atraso","status"]].copy()
    df_exp["data_vencimento"] = df_exp["data_vencimento"].dt.strftime("%Y-%m-%d")
    df_exp["data_pagamento"]  = df_exp["data_pagamento"].dt.strftime("%Y-%m-%d") if "data_pagamento" in df_exp.columns else ""
    buf_inad = io.BytesIO()
    df_exp.to_excel(buf_inad, index=False, engine="openpyxl")
    st.download_button("⬇️ Exportar análise completa de inadimplência",
                       buf_inad.getvalue(), "inadimplencia_trusted.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════════════════════
# ABA 5 — QUALIDADE DE DADOS
# ══════════════════════════════════════════════════════════════
with tab_quality:

    st.markdown("""
    <div class='info-box'>
      <div class='title'>🔬 Garantias da Camada Trusted</div>
      <div class='body'>
        <b style='color:#6b7280'>Tipagem:</b> datas em DATE, valores em DOUBLE, IDs em BIGINT &nbsp;·&nbsp;
        <b style='color:#6b7280'>Deduplicação:</b> por (id_fatura, id_cliente) &nbsp;·&nbsp;
        <b style='color:#6b7280'>Particionamento:</b> safra = YYYY-MM (data_emissao) &nbsp;·&nbsp;
        <b style='color:#6b7280'>Engine:</b> PySpark → Parquet (snappy) &nbsp;·&nbsp;
        <b style='color:#6b7280'>Ingestão:</b> S3 → Raw → Trusted
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown('<div class="sec">Diagnóstico de Nulos · tb_faturas</div>', unsafe_allow_html=True)
        cols_fat = ["id_fatura","id_cliente","data_emissao","data_vencimento",
                    "valor_fatura","valor_pagamento_minimo"]
        null_fat = {c: int(df_fat[c].isna().sum()) for c in cols_fat}
        df_nf = pd.DataFrame({
            "Coluna": cols_fat,
            "Nulos":  [null_fat[c] for c in cols_fat],
            "Total":  [len(df_fat)]*len(cols_fat),
        })
        df_nf["% Nulo"] = (df_nf["Nulos"]/df_nf["Total"]*100).round(2)
        df_nf["Status"] = df_nf["% Nulo"].apply(
            lambda x: "✅ OK" if x == 0 else ("⚠️ Atenção" if x < 5 else "❌ Crítico")
        )
        st.dataframe(df_nf, use_container_width=True, hide_index=True)

    with col_q2:
        st.markdown('<div class="sec">Diagnóstico de Nulos · tb_pagamentos</div>', unsafe_allow_html=True)
        cols_pag = ["id_pagamento","id_fatura","id_cliente","data_pagamento","valor_pagamento"]
        null_pag = {c: int(df_pag[c].isna().sum()) for c in cols_pag}
        df_np = pd.DataFrame({
            "Coluna": cols_pag,
            "Nulos":  [null_pag[c] for c in cols_pag],
            "Total":  [len(df_pag)]*len(cols_pag),
        })
        df_np["% Nulo"] = (df_np["Nulos"]/df_np["Total"]*100).round(2)
        df_np["Status"] = df_np["% Nulo"].apply(
            lambda x: "✅ OK" if x == 0 else ("⚠️ Atenção" if x < 5 else "❌ Crítico")
        )
        st.dataframe(df_np, use_container_width=True, hide_index=True)

    # Duplicatas
    st.markdown('<div class="sec">Verificação de Duplicatas</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    dup_fat = int(df_fat.duplicated(subset=["id_fatura","id_cliente"]).sum())
    dup_pag = int(df_pag.duplicated(subset=["id_pagamento","id_cliente"]).sum())
    dc1.metric("Duplicatas · tb_faturas",   "✅ Nenhuma" if dup_fat == 0 else f"⚠️ {dup_fat}",
               delta=f"{len(df_fat):,} registros únicos")
    dc2.metric("Duplicatas · tb_pagamentos","✅ Nenhuma" if dup_pag == 0 else f"⚠️ {dup_pag}",
               delta=f"{len(df_pag):,} registros únicos")

    # Integridade referencial
    st.markdown('<div class="sec">Integridade Referencial · Faturas sem Pagamento por Safra</div>', unsafe_allow_html=True)
    df_ref = (
        df_join_f.groupby("safra")
        .apply(lambda g: pd.Series({
            "total_fat":    len(g),
            "sem_pagamento":(g["data_pagamento"].isna()).sum(),
        }))
        .reset_index()
    )
    df_ref["pct_sem_pag"] = (df_ref["sem_pagamento"] / df_ref["total_fat"] * 100).round(1)
    fig_ref = go.Figure()
    fig_ref.add_trace(go.Bar(x=df_ref["safra"], y=df_ref["total_fat"],
                              name="Total Faturas", marker_color="#172554"))
    fig_ref.add_trace(go.Bar(x=df_ref["safra"], y=df_ref["sem_pagamento"],
                              name="Sem Pagamento", marker_color="#f43f5e"))
    fig_ref.add_trace(go.Scatter(x=df_ref["safra"], y=df_ref["pct_sem_pag"],
                                  name="% Sem Pagamento", yaxis="y2",
                                  line=dict(color="#fbbf24",width=2), mode="lines+markers"))
    fig_ref.update_layout(
        height=290, barmode="overlay",
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    ticksuffix="%", tickfont=dict(color="#fbbf24",size=10), range=[0,130]),
        legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"), **LAYOUT,
    )
    ax(fig_ref)
    st.plotly_chart(fig_ref, use_container_width=True)

    # Estatísticas das colunas
    st.markdown('<div class="sec">Estatísticas Descritivas · Colunas Numéricas</div>', unsafe_allow_html=True)
    df_desc = pd.DataFrame({
        "Coluna": ["valor_fatura","valor_pagamento_minimo","valor_pagamento"],
        "Mínimo": [brl(df_fat["valor_fatura"].min()),
                   brl(df_fat["valor_pagamento_minimo"].min()),
                   brl(df_pag["valor_pagamento"].min())],
        "Mediana": [brl(df_fat["valor_fatura"].median()),
                    brl(df_fat["valor_pagamento_minimo"].median()),
                    brl(df_pag["valor_pagamento"].median())],
        "Média":   [brl(df_fat["valor_fatura"].mean()),
                    brl(df_fat["valor_pagamento_minimo"].mean()),
                    brl(df_pag["valor_pagamento"].mean())],
        "Máximo":  [brl(df_fat["valor_fatura"].max()),
                    brl(df_fat["valor_pagamento_minimo"].max()),
                    brl(df_pag["valor_pagamento"].max())],
        "Desvio Padrão": [brl(df_fat["valor_fatura"].std()),
                          brl(df_fat["valor_pagamento_minimo"].std()),
                          brl(df_pag["valor_pagamento"].std())],
    })
    st.dataframe(df_desc, use_container_width=True, hide_index=True)

    # Inventário de arquivos da camada
    st.markdown('<div class="sec">Inventário de Arquivos · Camada Trusted</div>', unsafe_allow_html=True)
    inv = pd.DataFrame([
        {"Tabela":"tb_faturas",    "Safra":"2023-01","Arquivos":"2 (part-00000, part-00001)",
         "Registros":"10.000","Partição":"data_emissao","Formato":"Parquet (snappy)"},
        {"Tabela":"tb_pagamentos", "Safra":"2023-01","Arquivos":"1 (part-00000)",
         "Registros":"5.780","Partição":"data_pagamento","Formato":"Parquet (snappy)"},
    ])
    st.dataframe(inv, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# ABA 6 — PERFIL DO CLIENTE
# ══════════════════════════════════════════════════════════════
with tab_cliente:

    st.markdown('<div class="sec">Consulta Individual por Cliente</div>', unsafe_allow_html=True)

    col_ec1, col_ec2 = st.columns([1,2])
    with col_ec1:
        id_cli = st.number_input("ID do Cliente",
                                  min_value=int(df_fat["id_cliente"].min()),
                                  max_value=int(df_fat["id_cliente"].max()),
                                  value=int(df_fat["id_cliente"].iloc[0]), step=1)
        if st.button("🎲 Cliente Aleatório", use_container_width=True):
            id_cli = int(df_fat["id_cliente"].sample(1, random_state=None).iloc[0])
            st.rerun()

    df_cli_fat = df_fat[df_fat["id_cliente"] == id_cli].sort_values("data_emissao")
    df_cli_pag = df_pag[df_pag["id_cliente"] == id_cli].sort_values("data_pagamento")
    df_cli_join= df_join_f[df_join_f["id_cliente"] == id_cli]

    if df_cli_fat.empty:
        st.warning(f"Cliente {id_cli} não encontrado.")
    else:
        with col_ec2:
            status_cli = df_cli_join["status"].mode()[0] if not df_cli_join.empty else "—"
            cor_status = STATUS_COLORS.get(status_cli,"#6b7280")
            st.markdown(f"""
            <div style='background:#0c1222;border:1px solid #141e38;border-radius:8px;
                        padding:12px 18px;display:flex;gap:2rem;align-items:center;'>
              <div>
                <div style='color:#374151;font-size:0.68rem;text-transform:uppercase;letter-spacing:.1em;'>Cliente</div>
                <div style='font-family:IBM Plex Mono,monospace;color:#e2e8f0;font-size:1.2rem;'>#{id_cli}</div>
              </div>
              <div>
                <div style='color:#374151;font-size:0.68rem;text-transform:uppercase;letter-spacing:.1em;'>Status</div>
                <div style='color:{cor_status};font-weight:600;font-size:0.85rem;'>{status_cli}</div>
              </div>
              <div>
                <div style='color:#374151;font-size:0.68rem;text-transform:uppercase;letter-spacing:.1em;'>Faturas</div>
                <div style='font-family:IBM Plex Mono,monospace;color:#e2e8f0;'>{len(df_cli_fat)}</div>
              </div>
              <div>
                <div style='color:#374151;font-size:0.68rem;text-transform:uppercase;letter-spacing:.1em;'>Pagamentos</div>
                <div style='font-family:IBM Plex Mono,monospace;color:#e2e8f0;'>{len(df_cli_pag)}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        ck1,ck2,ck3,ck4 = st.columns(4)
        vol_fat_cli = df_cli_fat["valor_fatura"].sum()
        vol_pag_cli = df_cli_pag["valor_pagamento"].sum() if not df_cli_pag.empty else 0
        saldo_cli   = vol_fat_cli - vol_pag_cli
        ck1.metric("Volume Faturado", brl(vol_fat_cli))
        ck2.metric("Volume Pago",     brl(vol_pag_cli))
        ck3.metric("Saldo em Aberto", brl(saldo_cli), delta_color="inverse",
                   delta="em risco" if saldo_cli > 0 else "quitado")
        pct_pago_cli = (vol_pag_cli/vol_fat_cli*100) if vol_fat_cli else 0
        ck4.metric("% Pago",          f"{pct_pago_cli:.1f}%")

        col_cl1, col_cl2 = st.columns(2)
        with col_cl1:
            st.markdown("**Faturas**")
            df_cf_fmt = df_cli_fat[["safra","data_emissao","data_vencimento",
                                     "valor_fatura","valor_pagamento_minimo"]].copy()
            df_cf_fmt["valor_fatura"]          = df_cf_fmt["valor_fatura"].apply(brl)
            df_cf_fmt["valor_pagamento_minimo"]= df_cf_fmt["valor_pagamento_minimo"].apply(brl)
            df_cf_fmt.columns = ["Safra","Emissão","Vencimento","Fatura","Pgto Mínimo"]
            st.dataframe(df_cf_fmt, use_container_width=True, hide_index=True)

        with col_cl2:
            st.markdown("**Pagamentos**")
            if df_cli_pag.empty:
                st.info("Nenhum pagamento registrado.")
            else:
                df_cp_fmt = df_cli_pag[["data_pagamento","valor_pagamento"]].copy()
                df_cp_fmt["valor_pagamento"] = df_cp_fmt["valor_pagamento"].apply(brl)
                df_cp_fmt.columns = ["Data Pagamento","Valor Pago"]
                st.dataframe(df_cp_fmt, use_container_width=True, hide_index=True)

        # Gráfico timeline
        if not df_cli_fat.empty:
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Bar(
                x=df_cli_fat["safra"], y=df_cli_fat["valor_fatura"],
                name="Fatura", marker_color="#1d4ed8",
                text=[brl(v) for v in df_cli_fat["valor_fatura"]],
                textposition="outside", textfont=dict(size=9,color="#4b5563"),
            ))
            if not df_cli_pag.empty:
                df_cp_safra = (df_cli_pag
                               .assign(safra=df_cli_pag["data_pagamento"].dt.to_period("M").astype(str))
                               .groupby("safra")["valor_pagamento"].sum().reset_index())
                fig_tl.add_trace(go.Bar(
                    x=df_cp_safra["safra"], y=df_cp_safra["valor_pagamento"],
                    name="Pago", marker_color="#4ade80",
                    text=[brl(v) for v in df_cp_safra["valor_pagamento"]],
                    textposition="outside", textfont=dict(size=9,color="#4b5563"),
                ))
            fig_tl.update_layout(height=260, barmode="group",
                                  title_text=f"Fatura vs Pagamento — Cliente #{id_cli}",
                                  title_font=dict(size=12,color="#6b7280"),
                                  legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
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
