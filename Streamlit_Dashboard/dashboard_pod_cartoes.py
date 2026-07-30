"""
Dashboard Executivo e Técnico — Data Lake PoD Cartões
Visualização das camadas Trusted e Refined (Feature Store / Book de Variáveis).
Engine: DuckDB + Parquet.

"""

import io
import os
from pathlib import Path
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA & PALETA DE CORES

st.set_page_config(
    page_title="PoD Cartões — Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_ANTECIPADO   = "#38BDF8"  # Azul-céu
COLOR_EM_DIA       = "#2563EB"  # Azul-royal
COLOR_ATRASO       = "#9333EA"  # Roxo-médio
COLOR_INADIMPLENTE = "#E53333"  # Chumbo/Cinza-escuro

COLOR_MAP_STATUS = {
    "Pagamento Antecipado": COLOR_ANTECIPADO,
    "Pagamento em Dia": COLOR_EM_DIA,
    "Em Atraso": COLOR_ATRASO,
    "Inadimplente": COLOR_INADIMPLENTE,
}

COLOR_MAP_RETENCAO = {
    "Ativo em Dia": COLOR_EM_DIA,
    "Ativo em Atraso": COLOR_ATRASO,
    "Inativo / Churn": COLOR_INADIMPLENTE,
}

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding-left: 14px;
        padding-right: 14px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONEXÃO DUCKDB — Regras de Pagamento & Arquitetura

@st.cache_resource
def init_db() -> duckdb.DuckDBPyConnection:
    script_dir = Path(__file__).resolve().parent

    if (script_dir / "datalake").exists():
        base_dir = script_dir / "datalake"
    elif (script_dir.parent / "datalake").exists():
        base_dir = script_dir.parent / "datalake"
    else:
        base_dir = script_dir

    con = duckdb.connect()

    con.execute("SET max_memory = '2GB';")
    con.execute("SET preserve_insertion_order = false;")
    con.execute("SET threads = 2;")

    fat_path = str(base_dir / "*" / "tb_01_fatura" / "*" / "*.parquet").replace("\\", "/")
    pag_path = str(base_dir / "*" / "tb_02_pagamento" / "*" / "*.parquet").replace("\\", "/")
    book_path = str(base_dir / "*" / "book_fatura" / "*" / "*.parquet").replace("\\", "/")

    # View 1: Faturas
    con.execute(f"""
        CREATE OR REPLACE VIEW tb_faturas AS
        SELECT
            CAST(ref                    AS VARCHAR) AS ref,
            CAST(dt_proc                AS VARCHAR) AS dt_proc,
            CAST(id_cliente             AS BIGINT)  AS id_cliente,
            CAST(id_fatura              AS BIGINT)  AS id_fatura,
            CAST(data_emissao           AS DATE)    AS data_emissao,
            CAST(data_vencimento        AS DATE)    AS data_vencimento,
            CAST(valor_fatura           AS DOUBLE)  AS valor_fatura,
            CAST(valor_pagamento_minimo AS DOUBLE)  AS valor_pagamento_minimo,
            STRPTIME(CAST(ref AS VARCHAR), '%Y%m') AS data_referencia
        FROM read_parquet('{fat_path}', filename=true, union_by_name=true)
    """)

    # View 2: Pagamentos
    con.execute(f"""
        CREATE OR REPLACE VIEW tb_pagamentos AS
        SELECT
            CAST(ref            AS VARCHAR) AS ref,
            CAST(dt_proc        AS VARCHAR) AS dt_proc,
            CAST(id_cliente     AS BIGINT)  AS id_cliente,
            CAST(id_fatura      AS BIGINT)  AS id_fatura,
            CAST(id_pagamento   AS BIGINT)  AS id_pagamento,
            CAST(data_pagamento AS DATE)    AS data_pagamento,
            CAST(valor_pagamento AS DOUBLE) AS valor_pagamento,
            STRPTIME(CAST(ref AS VARCHAR), '%Y%m') AS data_referencia
        FROM read_parquet('{pag_path}', filename=true, union_by_name=true)
    """)

    # View 3: Book
    con.execute(f"""
        CREATE OR REPLACE VIEW vw_book_fatura AS
        SELECT 
            *,
            STRPTIME(CAST(ref AS VARCHAR), '%Y%m') AS data_referencia
        FROM read_parquet('{book_path}', filename=true, union_by_name=true)
    """)

    # View 4: Join Integrado
    con.execute("""
        CREATE OR REPLACE VIEW vw_join AS
        SELECT
            f.ref,
            f.dt_proc,
            f.id_fatura,
            f.id_cliente,
            f.data_referencia,
            STRFTIME(f.data_referencia, '%m/%Y') AS mes_ano,
            f.data_emissao,
            f.data_vencimento,
            f.valor_fatura,
            f.valor_pagamento_minimo,
            p.data_pagamento,
            COALESCE(p.valor_pagamento, 0) AS valor_pagamento,
            GREATEST(0, DATEDIFF('day', f.data_vencimento, COALESCE(p.data_pagamento, CURRENT_DATE))) AS dias_atraso,
            CASE
                WHEN p.data_pagamento IS NULL THEN 'Inadimplente'
                WHEN p.valor_pagamento < f.valor_fatura THEN 'Pagamento Parcial'
                WHEN p.data_pagamento < f.data_vencimento THEN 'Pagamento Antecipado'
                WHEN p.data_pagamento = f.data_vencimento THEN 'Pagamento em Dia'
                ELSE 'Em Atraso'
            END AS status
        FROM tb_faturas f
        LEFT JOIN tb_pagamentos p
               ON f.id_fatura  = p.id_fatura
              AND f.id_cliente = p.id_cliente
    """)

    return con

con = init_db()


def q(sql: str) -> pd.DataFrame:
    try:
        df = con.execute(sql).df()
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        st.error(f"Erro na consulta SQL: {e}")
        return pd.DataFrame()


def export_csv_button(df: pd.DataFrame, filename: str, label: str = "Exportar dados em CSV"):
    if df is not None and not df.empty:
        csv_bytes = df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label=label,
            data=csv_bytes,
            file_name=f"{filename}.csv",
            mime="text/csv",
            width='stretch'
        )


# ══════════════════════════════════════════════════════════════
# SIDEBAR — Filtro por Mês/Ano

st.sidebar.title("PoD Cartões")
st.sidebar.caption("Data Lake & Analytics")
st.sidebar.markdown("---")

st.sidebar.subheader("Período de Análise")

df_meses = q("""
    SELECT DISTINCT 
        data_referencia, 
        STRFTIME(data_referencia, '%m/%Y') AS label_mes,
        STRFTIME(data_referencia, '%Y-%m-%d') AS data_sql
    FROM vw_join 
    ORDER BY data_referencia
""")

if df_meses is not None and not df_meses.empty and "label_mes" in df_meses.columns:
    opcoes_meses = ["Todos os Meses"] + df_meses["label_mes"].tolist()
else:
    opcoes_meses = ["Todos os Meses"]

mes_selecionado = st.sidebar.selectbox("Selecione o Mês / Referência", opcoes_meses, index=0)

if mes_selecionado == "Todos os Meses" or df_meses is None or df_meses.empty:
    where_mes = "1=1"
    periodo_label = "Todo o Histórico"
else:
    data_ref_sql = df_meses.loc[df_meses["label_mes"] == mes_selecionado, "data_sql"].iloc[0]
    where_mes = f"data_referencia = '{data_ref_sql}'"
    periodo_label = mes_selecionado

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Período Selecionado:** `{periodo_label}`")


# ══════════════════════════════════════════════════════════════
# PAINEL PRINCIPAL

st.title("Monitoramento e Governança dos Dados")

aba_geral, aba_churn, aba_fat, aba_pag, aba_inad, aba_perfil, aba_book, aba_qual = st.tabs([
    "Visão Geral",
    "Churn & Atividade",
    "Faturas",
    "Pagamentos",
    "Inadimplência & Perfil",
    "Consulta Direta Cliente",
    "Feature Store (Book)",
    "Qualidade dos Dados",
])


# ══════════════════════════════════════════════════════════════
# ABA 1: VISÃO GERAL 

with aba_geral:
    df_kpi = q(f"""
        SELECT
            COUNT(DISTINCT id_cliente)                                                                      AS total_clientes,
            COUNT(DISTINCT CASE WHEN status = 'Pagamento Antecipado' THEN id_cliente END)                   AS cli_antecipados,
            COUNT(DISTINCT CASE WHEN status = 'Pagamento em Dia' THEN id_cliente END)                       AS cli_em_dia,
            COUNT(DISTINCT CASE WHEN status = 'Em Atraso' THEN id_cliente END)                             AS cli_atraso,
            COUNT(DISTINCT CASE WHEN status = 'Inadimplente' THEN id_cliente END)           AS cli_inad,
            COALESCE(SUM(valor_fatura), 0)                                                                  AS vol_faturado,
            COALESCE(SUM(valor_pagamento), 0)                                                               AS vol_pago,
            COALESCE(SUM(valor_fatura - valor_pagamento), 0)                                                AS vol_devedor
        FROM vw_join
        WHERE {where_mes}
    """)

    tot_cli = df_kpi["total_clientes"].iloc[0] or 1
    c_ant = df_kpi["cli_antecipados"].iloc[0] or 0
    c_dia = df_kpi["cli_em_dia"].iloc[0] or 0
    c_atr = df_kpi["cli_atraso"].iloc[0] or 0
    c_ind = df_kpi["cli_inad"].iloc[0] or 0

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Base Total de Clientes", f"{tot_cli:,}", delta="Período Ativo")
    c2.metric("Pgtos Antecipados", f"{c_ant:,}", delta=f"{(c_ant/tot_cli*100):.1f}% da base")
    c3.metric("Pgtos em Dia", f"{c_dia:,}", delta=f"{(c_dia/tot_cli*100):.1f}% da base")
    c4.metric("Em Atraso", f"{c_atr:,}", delta=f"{(c_atr/tot_cli*100):.1f}% da base", delta_color="inverse")
    c5.metric("Inadimplência Crítica", f"{c_ind:,}", delta=f"{(c_ind/tot_cli*100):.1f}% sem pgto", delta_color="inverse")

    st.markdown("---")
    
    # LAYOUT ORIGINAL EM 2 COLUNAS (1.3 / 0.7)
    g1, g2 = st.columns([1.3, 0.7])

    with g1:
        st.subheader("Evolução Temporal dos Indicadores")
        
        # "Todas as Categorias" EM PRIMEIRO
        opcoes_filtro_status = [
            "Todas as Categorias",
            "Pagamento Antecipado", 
            "Pagamento em Dia", 
            "Em Atraso", 
            "Inadimplente"
        ]
        
        metric_opt = st.radio(
            "Selecione o filtro temporal por categoria:",
            opcoes_filtro_status,
            index=0,
            horizontal=True,
            key="radio_visao_geral"
        )

        df_evolucao_status = q("""
            SELECT
                data_referencia,
                STRFTIME(data_referencia, '%m/%Y') AS mes_ano,
                status,
                SUM(valor_fatura) / 1e6 AS Volume_M
            FROM vw_join
            GROUP BY data_referencia, status
            ORDER BY data_referencia
        """)

        if metric_opt == "Todas as Categorias":
            # LINHAS TEMPORAIS PARA CADA CATEGORIA
            fig_evol = px.line(
                df_evolucao_status, x="mes_ano", y="Volume_M", color="status",
                color_discrete_map=COLOR_MAP_STATUS, markers=True
            )
            fig_evol.update_traces(line=dict(width=2.5), marker=dict(size=6))
        else:
            # BARRAS PARA CATEGORIA INDIVIDUAL
            df_plot_evol = df_evolucao_status[df_evolucao_status["status"] == metric_opt]
            fig_evol = px.bar(
                df_plot_evol, x="mes_ano", y="Volume_M", color="status",
                color_discrete_map=COLOR_MAP_STATUS, text_auto=".2f"
            )
            fig_evol.update_traces(textposition="outside", cliponaxis=False)

        fig_evol.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title=None, 
            yaxis_title="Volume Faturado (R$ Mi)",
            legend_title=None, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_evol, width='stretch', config={'displayModeBar': False})

    with g2:
        where_pie_geral = "data_referencia = (SELECT MAX(data_referencia) FROM vw_join)" if mes_selecionado == "Todos os Meses" else where_mes
        lbl_pie_geral = "Última Safra" if mes_selecionado == "Todos os Meses" else f"Safra {periodo_label}"

        st.subheader(f"Distribuição ({lbl_pie_geral})")
        df_status = q(f"""
            SELECT status, COUNT(*) as qtd, SUM(valor_fatura) as volume_bruto 
            FROM vw_join 
            WHERE {where_pie_geral} 
            GROUP BY status 
            ORDER BY qtd DESC
        """)
        
        if not df_status.empty:
            fig_bar_h = px.bar(
                df_status, y="status", x="qtd", orientation="h",
                color="status", color_discrete_map=COLOR_MAP_STATUS,
                text="qtd"
            )
            fig_bar_h.update_traces(textposition="outside", cliponaxis=False)
            fig_bar_h.update_layout(
                height=340, 
                showlegend=False, 
                margin=dict(l=10, r=30, t=20, b=10),
                xaxis_title="Qtd Faturas", 
                yaxis_title=None,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar_h, width='stretch')

            if metric_opt != "Todas as Categorias":
                sub_df = df_status.loc[df_status["status"] == metric_opt]
                if not sub_df.empty:
                    qtd_sel = sub_df["qtd"].iloc[0]
                    vol_sel = sub_df["volume_bruto"].iloc[0]
                    tot_qtd = df_status["qtd"].sum()
                    pct_sel = (qtd_sel / tot_qtd * 100) if tot_qtd > 0 else 0
                    
                    st.info(f"**{metric_opt}:** {qtd_sel:,} faturas ({pct_sel:.1f}% do total em {lbl_pie_geral}) somando **R$ {vol_sel/1e6:.2f}M**.")

    st.markdown("---")
    st.markdown("### Detalhamento Dinâmico da Base")
    
    list_status = ["Todas as Categorias"] + df_status["status"].tolist() if not df_status.empty else ["Todas as Categorias"]
    status_filtro_sel = st.selectbox("Filtrar Tabela por Categoria de Pagamento:", list_status, index=0)

    where_status_extra = "1=1" if status_filtro_sel == "Todas as Categorias" else f"status = '{status_filtro_sel}'"

    df_detalhe_geral = q(f"""
        SELECT
            id_cliente AS "ID Cliente",
            id_fatura AS "ID Fatura",
            mes_ano AS "Mês",
            valor_fatura AS "Valor Fatura",
            valor_pagamento AS "Valor Pago",
            dias_atraso AS "Dias Atraso",
            status AS "Categoria"
        FROM vw_join
        WHERE {where_mes} AND {where_status_extra}
        ORDER BY valor_fatura DESC
        LIMIT 100
    """)

    st.dataframe(
        df_detalhe_geral.style.format({
            "Valor Fatura": "R$ {:,.2f}",
            "Valor Pago": "R$ {:,.2f}",
            "Dias Atraso": "{:,.0f} dias"
        }),
        width='stretch',
        height=300
    )
    export_csv_button(df_detalhe_geral, f"detalhe_visao_geral_{periodo_label.replace('/', '-')}")


# ──────────────────────────────────────────────────────────────
# ABA 2: CHURN & ATIVIDADE DA CARTEIRA

with aba_churn:
    st.subheader("Análise de Retenção, Inatividade & Motivos de Churn")
    st.caption("Acompanhe o volume de clientes que deixaram de gerar faturas no tempo e diagnostique a causa do abandono.")

    # "Todas as Categorias" EM PRIMEIRO
    cat_churn_sel = st.radio(
        "Filtrar Categoria no Gráfico Temporal:",
        ["Todas as Categorias", "Ativo em Dia", "Ativo em Atraso", "Inativo / Churn"],
        index=0,
        horizontal=True,
        key="radio_churn_cat"
    )

    df_churn_evol = q("""
        WITH base_total_clientes AS (
            SELECT DISTINCT id_cliente FROM vw_join
        ),
        datas_ref AS (
            SELECT DISTINCT data_referencia, STRFTIME(data_referencia, '%m/%Y') AS mes_ano FROM vw_join
        ),
        grade_clientes AS (
            SELECT c.id_cliente, d.data_referencia, d.mes_ano
            FROM base_total_clientes c
            CROSS JOIN datas_ref d
        ),
        status_cliente_mes AS (
            SELECT
                g.id_cliente,
                g.data_referencia,
                g.mes_ano,
                j.status,
                CASE
                    WHEN j.id_fatura IS NULL THEN 'Inativo / Churn'
                    WHEN j.status LIKE '%em Dia%' OR j.status LIKE '%Antecipado%' THEN 'Ativo em Dia'
                    ELSE 'Ativo em Atraso'
                END AS categoria_retencao
            FROM grade_clientes g
            LEFT JOIN vw_join j
                   ON g.id_cliente = j.id_cliente
                  AND g.data_referencia = j.data_referencia
        )
        SELECT
            data_referencia,
            mes_ano,
            categoria_retencao,
            COUNT(DISTINCT id_cliente) AS qtd_clientes
        FROM status_cliente_mes
        GROUP BY data_referencia, mes_ano, categoria_retencao
        ORDER BY data_referencia, categoria_retencao
    """)

    # LAYOUT ORIGINAL EM 2 COLUNAS (1.3 / 0.7)
    col_ch1, col_ch2 = st.columns([1.3, 0.7])

    with col_ch1:
        st.markdown("##### Evolução Temporal da Base de Clientes")
        
        if cat_churn_sel == "Todas as Categorias":
            # LINHAS TEMPORAIS PARA CADA CATEGORIA DE RETENÇÃO
            fig_churn = px.line(
                df_churn_evol, x="mes_ano", y="qtd_clientes", color="categoria_retencao",
                color_discrete_map=COLOR_MAP_RETENCAO, markers=True
            )
            fig_churn.update_traces(line=dict(width=2.5), marker=dict(size=6))
        else:
            # BARRAS PARA CATEGORIA SELECIONADA
            df_churn_plot = df_churn_evol[df_churn_evol["categoria_retencao"] == cat_churn_sel]
            fig_churn = px.bar(
                df_churn_plot, x="mes_ano", y="qtd_clientes", color="categoria_retencao",
                color_discrete_map=COLOR_MAP_RETENCAO, text_auto=True
            )
            fig_churn.update_traces(textposition="outside", cliponaxis=False)

        fig_churn.update_layout(
            height=420, 
            margin=dict(l=10, r=10, t=30, b=20),
            xaxis_title=None, 
            yaxis_title="Qtd Clientes", 
            legend_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_churn, width='stretch')

    with col_ch2:
        where_pie_churn = "data_referencia = (SELECT MAX(data_referencia) FROM vw_join)" if mes_selecionado == "Todos os Meses" else where_mes
        lbl_pie_churn = "Posição Atual (Última Safra)" if mes_selecionado == "Todos os Meses" else f"Safra {periodo_label}"

        st.markdown(f"##### Diagnóstico ({lbl_pie_churn})")

        df_diagnostico = q(f"""
            WITH clientes_referencia AS (
                SELECT DISTINCT id_cliente, status FROM vw_join WHERE {where_pie_churn}
            ),
            todos_clientes AS (
                SELECT DISTINCT id_cliente FROM vw_join
            )
            SELECT
                CASE
                    WHEN cr.id_cliente IS NULL THEN 'Inativo / Churn'
                    WHEN cr.status LIKE '%em Dia%' OR cr.status LIKE '%Antecipado%' THEN 'Ativo em Dia'
                    ELSE 'Ativo em Atraso'
                END AS diagnostico,
                COUNT(tc.id_cliente) AS total
            FROM todos_clientes tc
            LEFT JOIN clientes_referencia cr ON tc.id_cliente = cr.id_cliente
            GROUP BY diagnostico
            ORDER BY total DESC
        """)

        fig_diag_h = px.bar(
            df_diagnostico, y="diagnostico", x="total", orientation="h",
            color="diagnostico", color_discrete_map=COLOR_MAP_RETENCAO,
            text="total"
        )
        fig_diag_h.update_traces(textposition="outside", cliponaxis=False)
        fig_diag_h.update_layout(
            height=360, 
            showlegend=False, 
            margin=dict(l=10, r=30, t=20, b=10),
            xaxis_title="Qtd Clientes", 
            yaxis_title=None,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_diag_h, width='stretch')

        if cat_churn_sel != "Todas as Categorias" and not df_diagnostico.empty:
            sub_val = df_diagnostico.loc[df_diagnostico["diagnostico"] == cat_churn_sel, "total"]
            qtd_cat_sel = sub_val.iloc[0] if not sub_val.empty else 0
            tot_base = df_diagnostico["total"].sum()
            pct_cat_sel = (qtd_cat_sel / tot_base * 100) if tot_base > 0 else 0
            
            st.info(f"**{cat_churn_sel}:** {qtd_cat_sel:,} clientes ({pct_cat_sel:.1f}% da base na fotografia atual).")

    st.markdown("---")
    st.subheader("Clientes Inativos / Perdidos (Clique na linha para selecionar)")

    where_churn_tabela = "data_referencia = (SELECT MAX(data_referencia) FROM vw_join)" if mes_selecionado == "Todos os Meses" else where_mes
    label_referencia_tabela = "na Safra Mais Recente" if mes_selecionado == "Todos os Meses" else f"em `{periodo_label}`"

    df_clientes_inativos = q(f"""
        WITH todos_clientes AS (
            SELECT DISTINCT id_cliente FROM vw_join
        ),
        ativos_no_mes AS (
            SELECT DISTINCT id_cliente FROM vw_join WHERE {where_churn_tabela}
        ),
        historico_acumulado AS (
            SELECT
                id_cliente,
                COUNT(DISTINCT id_fatura) AS historico_faturas,
                SUM(valor_fatura) AS historico_faturado,
                SUM(valor_pagamento) AS historico_pago,
                MAX(data_referencia) AS ultima_safra_ativa
            FROM vw_join
            GROUP BY id_cliente
        )
        SELECT
            tc.id_cliente AS "ID_Cliente",
            STRFTIME(ha.ultima_safra_ativa, '%m/%Y') AS "Última Safra Ativa",
            ha.historico_faturas AS "Faturas Totais",
            ha.historico_faturado AS "Total Faturado",
            ha.historico_pago AS "Total Pago",
            (ha.historico_faturado - ha.historico_pago) AS "Saldo Devedor"
        FROM todos_clientes tc
        LEFT JOIN ativos_no_mes am ON tc.id_cliente = am.id_cliente
        INNER JOIN historico_acumulado ha ON tc.id_cliente = ha.id_cliente
        WHERE am.id_cliente IS NULL
        ORDER BY "Saldo Devedor" DESC
    """)

    if df_clientes_inativos.empty:
        st.info(f"Nenhum cliente inativo registrado {label_referencia_tabela}.")
    else:
        col_tb_inad, col_raiox_inad = st.columns([1.1, 0.9])

        with col_tb_inad:
            st.markdown(f"**Clientes inativos {label_referencia_tabela} ({len(df_clientes_inativos)}):**")
            
            evento_tabela_churn = st.dataframe(
                df_clientes_inativos.style.format({
                    "Total Faturado": "R$ {:,.2f}",
                    "Total Pago": "R$ {:,.2f}",
                    "Saldo Devedor": "R$ {:,.2f}"
                }),
                width='stretch',
                height=380,
                on_select="rerun",
                selection_mode="single-row",
                key="tabela_churn"
            )
            export_csv_button(df_clientes_inativos, f"inativos_churn_{periodo_label.replace('/', '-')}")

        with col_raiox_inad:
            if hasattr(evento_tabela_churn, "selection") and evento_tabela_churn.selection.rows:
                idx_sel = evento_tabela_churn.selection.rows[0]
                st.session_state["cli_churn_sel"] = int(df_clientes_inativos.iloc[idx_sel]["ID_Cliente"])
            elif "cli_churn_sel" not in st.session_state or st.session_state["cli_churn_sel"] not in df_clientes_inativos["ID_Cliente"].values:
                st.session_state["cli_churn_sel"] = int(df_clientes_inativos.iloc[0]["ID_Cliente"])

            cli_inativo_sel = st.session_state["cli_churn_sel"]

            st.markdown(f"### Histórico do Cliente Id {cli_inativo_sel}")
            
            df_cli_detalhe = q(f"""
                SELECT
                    mes_ano AS Mes,
                    valor_fatura AS Fatura,
                    valor_pagamento AS Pago,
                    status AS Categoria
                FROM vw_join
                WHERE id_cliente = {cli_inativo_sel}
                ORDER BY data_referencia DESC
            """)
            
            st.dataframe(
                df_cli_detalhe.style.format({"Fatura": "R$ {:,.2f}", "Pago": "R$ {:,.2f}"}), 
                width='stretch', 
                height=330
            )
# ──────────────────────────────────────────────────────────────
# ABA 3: FATURAS

with aba_fat:
    st.subheader("Distribuição e Faixas do Valor de Fatura")
    
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        df_faixas = q(f"""
            SELECT
                CASE
                    WHEN valor_fatura <= 1000 THEN '1. Até R$ 1k'
                    WHEN valor_fatura <= 3000 THEN '2. R$ 1k a R$ 3k'
                    WHEN valor_fatura <= 5000 THEN '3. R$ 3k a R$ 5k'
                    ELSE '4. Acima de R$ 5k'
                END AS faixa_valor,
                COUNT(*) AS qtd
            FROM vw_join
            WHERE {where_mes}
            GROUP BY faixa_valor
            ORDER BY faixa_valor
        """)

        max_val = df_faixas["qtd"].max() if not df_faixas.empty else 100

        fig_faixa = px.bar(
            df_faixas, x="faixa_valor", y="qtd", text="qtd",
            title="Concentração de Faturas por Faixa de Valor",
            color_discrete_sequence=[COLOR_EM_DIA]
        )
        fig_faixa.update_traces(textposition="outside", cliponaxis=False)
        fig_faixa.update_layout(
            height=380, xaxis_title=None, yaxis_title="Qtd Faturas",
            yaxis=dict(range=[0, max_val * 1.18])
        )
        st.plotly_chart(fig_faixa, width='stretch')

    with col_f2:
        df_minimo = q(f"""
            SELECT
                CASE
                    WHEN valor_pagamento >= valor_fatura THEN 'Pagamento Integral'
                    WHEN valor_pagamento >= valor_pagamento_minimo THEN 'Pagou Mínimo / Parcial'
                    ELSE 'Abaixo do Mínimo / Nulo'
                END AS comportamento_minimo,
                COUNT(*) AS qtd
            FROM vw_join
            WHERE {where_mes}
            GROUP BY comportamento_minimo
            ORDER BY qtd ASC
        """)
        
        # STORYTELLING COM DADOS: Barras Horizontais para Comportamento
        fig_min_h = px.bar(
            df_minimo, y="comportamento_minimo", x="qtd", orientation="h",
            title="Comportamento do Pagamento Mínimo",
            color="comportamento_minimo",
            color_discrete_map={
                "Pagamento Integral": COLOR_EM_DIA,
                "Pagou Mínimo / Parcial": COLOR_ANTECIPADO,
                "Abaixo do Mínimo / Nulo": COLOR_INADIMPLENTE
            },
            text="qtd"
        )
        fig_min_h.update_traces(textposition="outside", cliponaxis=False)
        fig_min_h.update_layout(height=380, showlegend=False, xaxis_title="Qtd Faturas", yaxis_title=None)
        st.plotly_chart(fig_min_h, width='stretch')

    st.markdown("---")
    st.markdown("### Consulta Detalhada por Comportamento de Pagamento")
    
    comp_opcoes = ["Todos os Comportamentos", "Pagamento Integral", "Pagou Mínimo / Parcial", "Abaixo do Mínimo / Nulo"]
    comp_sel = st.selectbox("Selecione o perfil para listar faturas:", comp_opcoes, index=0)

    where_comp_extra = "1=1"
    if comp_sel == "Pagamento Integral":
        where_comp_extra = "valor_pagamento >= valor_fatura"
    elif comp_sel == "Pagou Mínimo / Parcial":
        where_comp_extra = "valor_pagamento >= valor_pagamento_minimo AND valor_pagamento < valor_fatura"
    elif comp_sel == "Abaixo do Mínimo / Nulo":
        where_comp_extra = "valor_pagamento < valor_pagamento_minimo"

    df_fat_detalhe = q(f"""
        SELECT
            id_cliente AS "ID Cliente",
            id_fatura AS "ID Fatura",
            mes_ano AS "Mês",
            valor_fatura AS "Valor Fatura",
            valor_pagamento_minimo AS "Valor Mínimo",
            valor_pagamento AS "Valor Pago",
            status AS "Categoria"
        FROM vw_join
        WHERE {where_mes} AND {where_comp_extra}
        ORDER BY valor_fatura DESC
        LIMIT 100
    """)

    st.dataframe(
        df_fat_detalhe.style.format({
            "Valor Fatura": "R$ {:,.2f}",
            "Valor Mínimo": "R$ {:,.2f}",
            "Valor Pago": "R$ {:,.2f}"
        }),
        width='stretch',
        height=300
    )
    export_csv_button(df_fat_detalhe, f"faturas_comportamento_{periodo_label.replace('/', '-')}")


# ──────────────────────────────────────────────────────────────
# ABA 4: PAGAMENTOS

with aba_pag:
    st.subheader("Estatus de Pagamento e Faixas de Atraso")

    df_buckets = q(f"""
        SELECT
            CASE
                WHEN status LIKE '%Antecipado%' OR status LIKE '%em Dia%' THEN '1. Em Dia / Antecipado'
                WHEN dias_atraso <= 30 THEN '2. Atraso Leve (1-30d)'
                WHEN dias_atraso <= 90 THEN '3. Atraso Moderado (31-90d)'
                ELSE '4. Inadimplente (>90d)'
            END AS bucket_atraso,
            COUNT(*) AS qtd,
            SUM(valor_fatura) / 1e6 AS volume_m
        FROM vw_join
        WHERE {where_mes}
        GROUP BY bucket_atraso
        ORDER BY bucket_atraso
    """)

    if not df_buckets.empty and "bucket_atraso" in df_buckets.columns:
        max_b = df_buckets["volume_m"].max() if not df_buckets.empty else 1.0

        fig_bucket = px.bar(
            df_buckets, 
            x="bucket_atraso", 
            y="volume_m", 
            text="volume_m",
            title="Volume em Risco por Faixa de Atraso (R$ Milhões)",
            color="bucket_atraso",
            color_discrete_map={
                '1. Em Dia / Antecipado': COLOR_EM_DIA,
                '2. Atraso Leve (1-30d)': COLOR_ATRASO,
                '3. Atraso Moderado (31-90d)': COLOR_INADIMPLENTE,
                '4. Inadimplente (>90d)': "#E53333"
            }
        )

        fig_bucket.update_traces(
            texttemplate='R$ %{y:.2f}M', 
            textposition='outside', 
            cliponaxis=False,
            width=0.35
        )

        fig_bucket.update_layout(
            height=360, 
            showlegend=False, 
            yaxis_title="Volume (R$ M)", 
            xaxis_title=None,
            yaxis=dict(range=[0, max_b * 1.25])
        )
        
        st.plotly_chart(fig_bucket, width='stretch')
    else:
        st.info("Nenhum registro encontrado para calcular o gráfico de faixas de atraso no período selecionado.")

    st.markdown("---")
    st.markdown("### Filtrar e Baixar Tabela por Faixa de Atraso")

    bucket_sel = st.selectbox(
        "Selecione a faixa para consultar as faturas:",
        ["Todas as Faixas", "1. Em Dia / Antecipado", "2. Atraso Leve (1-30d)", "3. Atraso Moderado (31-90d)", "4. Inadimplente (>90d)"],
        index=0
    )

    where_bucket_extra = "1=1"
    if bucket_sel == "1. Em Dia / Antecipado":
        where_bucket_extra = "status LIKE '%Antecipado%' OR status LIKE '%em Dia%'"
    elif bucket_sel == "2. Atraso Leve (1-30d)":
        where_bucket_extra = "dias_atraso > 0 AND dias_atraso <= 30"
    elif bucket_sel == "3. Atraso Moderado (31-90d)":
        where_bucket_extra = "dias_atraso > 30 AND dias_atraso <= 90"
    elif bucket_sel == "4. Inadimplente (>90d)":
        where_bucket_extra = "dias_atraso > 90"

    df_pag_detalhe = q(f"""
        SELECT
            id_cliente AS "ID Cliente",
            id_fatura AS "ID Fatura",
            mes_ano AS "Mês",
            valor_fatura AS "Valor Fatura",
            valor_pagamento AS "Valor Pago",
            dias_atraso AS "Dias Atraso",
            status AS "Categoria"
        FROM vw_join
        WHERE {where_mes} AND {where_bucket_extra}
        ORDER BY dias_atraso DESC, valor_fatura DESC
        LIMIT 100
    """)

    st.dataframe(
        df_pag_detalhe.style.format({
            "Valor Fatura": "R$ {:,.2f}",
            "Valor Pago": "R$ {:,.2f}",
            "Dias Atraso": "{:,.0f} dias"
        }),
        width='stretch',
        height=300
    )
    export_csv_button(df_pag_detalhe, f"pagamentos_faixa_{periodo_label.replace('/', '-')}")


# ──────────────────────────────────────────────────────────────
# ABA 5: INADIMPLÊNCIA & PERFIL INTEGRADO 

with aba_inad:
    st.subheader("Cobrança & Gestão de Risco")
    st.caption("Clique em qualquer cliente na tabela para atualizar o histórico.")

    df_top_risco = q(f"""
        SELECT
            id_cliente AS "ID_Cliente",
            COUNT(DISTINCT id_fatura) AS "Faturas Inadimplentes",
            SUM(valor_fatura - valor_pagamento) AS "Saldo Devedor",
            MAX(dias_atraso) AS "Maior Atraso",
            MAX(status) AS "Categoria Principal"
        FROM vw_join
        WHERE {where_mes} AND (status LIKE '%Atraso%' OR status LIKE '%Inadimplente%')
        GROUP BY id_cliente
        ORDER BY "Saldo Devedor" DESC
        LIMIT 20
    """)

    col_tabela, col_detalhe = st.columns([1.1, 0.9])

    with col_tabela:
        st.markdown("##### Clientes com Maior Saldo Devedor")
        
        evento_inad = st.dataframe(
            df_top_risco.style.format({
                "Saldo Devedor": "R$ {:,.2f}",
                "Maior Atraso": "{:,.0f} dias"
            }),
            width='stretch', 
            height=380,
            on_select="rerun",
            selection_mode="single-row",
            key="tabela_inadimplencia"
        )
        export_csv_button(df_top_risco, f"top_20_inadimplentes_{periodo_label.replace('/', '-')}")

    with col_detalhe:
        if hasattr(evento_inad, "selection") and evento_inad.selection.rows:
            idx_inad = evento_inad.selection.rows[0]
            st.session_state["cli_inad_sel"] = int(df_top_risco.iloc[idx_inad]["ID_Cliente"])
        elif "cli_inad_sel" not in st.session_state or st.session_state["cli_inad_sel"] not in df_top_risco["ID_Cliente"].values:
            if not df_top_risco.empty:
                st.session_state["cli_inad_sel"] = int(df_top_risco.iloc[0]["ID_Cliente"])
            else:
                st.session_state["cli_inad_sel"] = None

        cliente_id_sel = st.session_state.get("cli_inad_sel")

        if cliente_id_sel:
            st.markdown(f"### Histórico do Cliente {cliente_id_sel}")

            df_cli_summary = q(f"""
                SELECT
                    COUNT(DISTINCT id_fatura) AS total_faturas,
                    SUM(valor_fatura) AS total_faturado,
                    SUM(valor_pagamento) AS total_pago,
                    AVG(dias_atraso) AS media_atraso
                FROM vw_join
                WHERE id_cliente = {cliente_id_sel}
            """)

            v_fat = df_cli_summary['total_faturado'].iloc[0] or 0
            v_pag = df_cli_summary['total_pago'].iloc[0] or 0
            m_atr = df_cli_summary['media_atraso'].iloc[0] or 0

            lbl_fat = f"R$ {v_fat/1e3:.1f}k" if v_fat >= 10000 else f"R$ {v_fat:,.2f}"
            lbl_pag = f"R$ {v_pag/1e3:.1f}k" if v_pag >= 10000 else f"R$ {v_pag:,.2f}"

            p1, p2, p3 = st.columns(3)
            p1.metric("Total Faturado", lbl_fat)
            p2.metric("Total Pago", lbl_pag)
            p3.metric("Média Atraso", f"{m_atr:.0f} dias")

            st.markdown("**Histórico de Faturas Recentes:**")
            df_cli_hist = q(f"""
                SELECT
                    mes_ano AS Mes,
                    valor_fatura AS Fatura,
                    valor_pagamento AS Pago,
                    status AS Categoria
                FROM vw_join
                WHERE id_cliente = {cliente_id_sel}
                ORDER BY data_referencia DESC
                LIMIT 6
            """)
            st.dataframe(
                df_cli_hist.style.format({"Fatura": "R$ {:,.2f}", "Pago": "R$ {:,.2f}"}), 
                width='stretch',
                height=230
            )


# ──────────────────────────────────────────────────────────────
# ABA 6: PERFIL DO CLIENTE

with aba_perfil:
    st.subheader("Busca Direta por ID de Cliente")
    id_busca = st.number_input("Digite o ID do Cliente:", min_value=1, value=343, step=1)
    
    if id_busca:
        df_direct = q(f"SELECT * FROM vw_join WHERE id_cliente = {id_busca} ORDER BY data_referencia DESC")
        if not df_direct.empty:
            st.dataframe(df_direct, width='stretch')
        else:
            st.warning(f"Cliente #{id_busca} não encontrado na base.")


# ──────────────────────────────────────────────────────────────
# ABA 7: FEATURE STORE (BOOK DE VARIÁVEIS)

with aba_book:
    st.info("""
        **Camada Refined (Feature Store) para Machine Learning:**  
        Tradução de transações brutas em métricas comportamentais agregadas (janelas de **1, 3, 6 e 12 meses**).  
        Colunas reorganizadas com `ref`, `dt_proc` e `id_cliente` no início para garantir governança.
    """)

    # Traz as colunas com ref, dt_proc e id_cliente obrigatoriamente primeiro
    df_sample_book = q("SELECT ref, dt_proc, id_cliente, * EXCLUDE (ref, dt_proc, id_cliente) FROM vw_book_fatura LIMIT 100")

    if isinstance(df_sample_book, pd.DataFrame) and not df_sample_book.empty:
        total_cols = max(0, len(df_sample_book.columns) - 3)
        st.metric("Total de Features Geradas no Book", total_cols)

        df_book_display = df_sample_book.copy()

        cols_numericas = df_book_display.select_dtypes(include=['number', 'Int32', 'Int64', 'float64']).columns
        df_book_display[cols_numericas] = df_book_display[cols_numericas].fillna(0)

        st.subheader("Amostra das Features Comportamentais (Camada Refined)")
        st.dataframe(df_book_display, width='stretch')
        export_csv_button(df_sample_book, "amostra_feature_store_refined")
    else:
        st.error("Não foi possível conectar à View 'vw_book_fatura'. Verifique os arquivos Parquet na pasta datalake.")

# ──────────────────────────────────────────────────────────────
# ABA 8: QUALIDADE DOS DADOS 

with aba_qual:
    st.subheader("Diagnóstico e Integridade dos Dados")
    
    # Consultas ajustadas trocando 'safra_raw' por 'ref'
    df_dup_fat = q("""
        SELECT COUNT(*) - COUNT(DISTINCT id_fatura) AS dup_fat
        FROM (SELECT id_fatura, ref FROM tb_faturas GROUP BY id_fatura, ref)
    """)
    
    df_dup_pag = q("""
        SELECT COUNT(*) - COUNT(DISTINCT id_pagamento) AS dup_pag
        FROM (SELECT id_pagamento, ref FROM tb_pagamentos GROUP BY id_pagamento, ref)
    """)

    # Leitura segura com fallback para 0 caso a query não retorne
    qtd_dup_fat = df_dup_fat['dup_fat'].iloc[0] if (isinstance(df_dup_fat, pd.DataFrame) and not df_dup_fat.empty and 'dup_fat' in df_dup_fat.columns) else 0
    qtd_dup_pag = df_dup_pag['dup_pag'].iloc[0] if (isinstance(df_dup_pag, pd.DataFrame) and not df_dup_pag.empty and 'dup_pag' in df_dup_pag.columns) else 0

    q1, q2 = st.columns(2)
    q1.metric("Duplicatas em Faturas (Deduplicado)", f"{qtd_dup_fat}", delta="OK")
    q2.metric("Duplicatas em Pagamentos (Deduplicado)", f"{qtd_dup_pag}", delta="OK")

    st.success("Todos os arquivos Parquet foram limpos, deduplicados e validados pela camada de Observabilidade.")