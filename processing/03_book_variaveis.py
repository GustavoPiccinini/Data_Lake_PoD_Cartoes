"""
Book de Variáveis (002_trusted -> 003_refined/book_fatura)
"""
import argparse
import os
import sys
from datetime import datetime

python_path = r"C:\Data_Lake_PoD_Cartoes\.venv\Scripts\python.exe"

os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop.home.dir"] = "C:\\hadoop"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.lake import layer_path, resolve_lake_root  # noqa: E402
from common.observability import check_integrity, record_lineage  # noqa: E402

FBCS = ["PAGO_EM_ATRASO", "NAO_PAGO", "PAGO_EM_DIA", "PAGAMENTO_ANTECIPADO"]
JANELAS = {"total": None, "u1m": "flg_u1m", "u3m": "flg_u3m", "u6m": "flg_u6m", "u12m": "flg_u12m"}

METRIC_SPECS = []
for fbc in FBCS:
    tag = fbc.lower()
    for janela_nome in JANELAS:
        METRIC_SPECS.append((f"qtd_transacoes_{tag}_{janela_nome}", "fvl_qtd_transacoes", "SUM", fbc, janela_nome))
        for agg in ("SUM", "AVG", "MIN", "MAX"):
            METRIC_SPECS.append((f"valor_fatura_{agg.lower()}_{tag}_{janela_nome}", "fvl_valor_fatura", agg, fbc, janela_nome))
        for agg in ("AVG", "MIN", "MAX"):
            METRIC_SPECS.append((f"dias_atraso_{agg.lower()}_{tag}_{janela_nome}", "fvl_numero_dias_atraso", agg, fbc, janela_nome))


def dedup_latest(df, key_cols: list):
    """Mantém só o registro mais recente (maior dt_proc) por chave."""
    w = Window.partitionBy(*key_cols).orderBy(col("dt_proc").desc())
    return (
        df.withColumn("_rn", row_number().over(w))
        .where(col("_rn") == 1)
        .drop("_rn")
    )


def build_book_query() -> str:
    select_parts = ["id_cliente", "ref"]
    for nome_metrica, coluna_valor, agg, fbc, janela_nome in METRIC_SPECS:
        janela_col = JANELAS[janela_nome]
        janela_cond = "1=1" if janela_col is None else f"{janela_col} = 1"
        
        # Para somas e contagens, substitui NULL por 0
        if agg in ("SUM", "COUNT"):
            metric_expr = (
                f"COALESCE({agg}(CASE WHEN fbc_classificacao_pagamento = '{fbc}' AND {janela_cond} "
                f"THEN {coluna_valor} END), 0) AS {nome_metrica}"
            )
        # Para Média, Mínimo e Máximo (ex: média de dias de atraso), mantém NULL quando não houve atraso
        else:
            metric_expr = (
                f"{agg}(CASE WHEN fbc_classificacao_pagamento = '{fbc}' AND {janela_cond} "
                f"THEN {coluna_valor} END) AS {nome_metrica}"
            )
            
        select_parts.append(metric_expr)
        
    select_sql = ",\n    ".join(select_parts)
    return f"""
        SELECT
            {select_sql}
        FROM janela_de_tempo
        GROUP BY id_cliente, ref
    """


def main(ref_date_str: str | None, lake_root: str | None):
    lake_root = resolve_lake_root(lake_root)
    print(f"LAKE_ROOT = {lake_root}")

    spark = (
        SparkSession.builder
        .appName("pod_cartoes_book_variaveis")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.file", "org.apache.hadoop.mapreduce.lib.output.FileOutputCommitterFactory")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.fs.permissions.umask-mode", "022")
        .getOrCreate()
    )

    dt_proc = datetime.now().strftime("%Y%m%d%H%M%S")

    fatura_path = layer_path(lake_root, "trusted", "tb_01_fatura")
    pagamento_path = layer_path(lake_root, "trusted", "tb_02_pagamento")

    tb_fatura = spark.read.parquet(fatura_path)

    if ref_date_str:
        ref_yyyymm = ref_date_str.replace("-", "")[:6]
        tb_fatura = tb_fatura.where(f"ref = '{ref_yyyymm}'")
        print(f"[fatura] Filtrando apenas a safra ref = {ref_yyyymm}")

    tb_fatura_dedup = dedup_latest(tb_fatura, ["id_cliente", "id_fatura"])
    tb_fatura_dedup.createOrReplaceTempView("tb_fatura_final")

    tb_pagamento = spark.read.parquet(pagamento_path)
    tb_pagamento_dedup = dedup_latest(tb_pagamento, ["id_cliente", "id_fatura"])
    tb_pagamento_dedup.createOrReplaceTempView("tb_pagamento_final")

    df_join = spark.sql("""
        select
            a.id_cliente,
            a.id_fatura,
            a.ref,
            a.data_emissao,
            a.data_vencimento,
            a.valor_fatura,
            a.valor_pagamento_minimo,
            b.id_pagamento,
            b.data_pagamento,
            b.valor_pagamento
        from tb_fatura_final a
        left join tb_pagamento_final b
        on a.id_fatura = b.id_fatura and a.id_cliente = b.id_cliente
    """)
    df_join.createOrReplaceTempView("df_join")

    tb_classificado = spark.sql("""
        select
            *,
            case
                when data_pagamento is null then 'NAO_PAGO'
                when data_pagamento = data_vencimento then 'PAGO_EM_DIA'
                when data_pagamento < data_vencimento then 'PAGAMENTO_ANTECIPADO'
                when data_pagamento > data_vencimento then 'PAGO_EM_ATRASO'
            end fbc_classificacao_pagamento
        from df_join
    """)
    tb_classificado.createOrReplaceTempView("tb_classificado")

    # Define a data limite para o cálculo de atraso
    ref_eval_date = f"'{ref_date_str}'" if ref_date_str else "coalesce(data_pagamento, data_vencimento)"

    tb_dias_atraso = spark.sql(f"""
        select
            *,
            case
                when fbc_classificacao_pagamento = 'NAO_PAGO' then datediff({ref_eval_date}, data_vencimento)
                when fbc_classificacao_pagamento = 'PAGO_EM_ATRASO' then datediff(data_pagamento, data_vencimento)
                when fbc_classificacao_pagamento = 'PAGO_EM_DIA' then 0
                when fbc_classificacao_pagamento = 'PAGAMENTO_ANTECIPADO' then 0
            end fvl_numero_dias_atraso
        from tb_classificado
    """)
    tb_dias_atraso.createOrReplaceTempView("tb_dias_atraso")

    stage = spark.sql(f"""
        select
            id_cliente,
            ref,
            "{dt_proc}" as dt_proc,
            fbc_classificacao_pagamento,
            data_emissao,
            valor_fatura as fvl_valor_fatura,
            fvl_numero_dias_atraso,
            1 as fvl_qtd_transacoes
        from tb_dias_atraso
    """)
    stage.cache()
    stage.createOrReplaceTempView("stage")

    stage_path = layer_path(lake_root, "refined", "stage_fatura")
    stage.write.mode("append").partitionBy("ref").parquet(stage_path)

    # Ajuste de Janelas Temporais: Calcula a janela relativa à data de emissão
    date_anchor = f"cast('{ref_date_str}' as date)" if ref_date_str else "max(data_emissao) over (partition by id_cliente)"

    janela_de_tempo = spark.sql(f"""
        select
            *,
            case when months_between({date_anchor}, data_emissao) <= 1 then 1 else 0 end flg_u1m,
            case when months_between({date_anchor}, data_emissao) <= 3 then 1 else 0 end flg_u3m,
            case when months_between({date_anchor}, data_emissao) <= 6 then 1 else 0 end flg_u6m,
            case when months_between({date_anchor}, data_emissao) <= 12 then 1 else 0 end flg_u12m
        from stage
    """)
    janela_de_tempo.createOrReplaceTempView("janela_de_tempo")

    book = spark.sql(build_book_query())
    book.cache()

    check_integrity(
        spark, book,
        tabela="book_fatura",
        lake_root=lake_root,
        dt_proc=dt_proc,
        key_cols=["id_cliente", "ref"],
        not_null_cols=["id_cliente", "ref"],
    )

    book_path = layer_path(lake_root, "refined", "book_fatura")
    book.write.mode("append").partitionBy("ref").parquet(book_path)
    print(f"[book] {book.count()} registros de clientes gravados em {book_path}")

    record_lineage(
        spark, lake_root,
        tabela="book_fatura",
        dt_proc=dt_proc,
        qtd_registros=book.count(),
        camada_origem="002_trusted/tb_01_fatura + tb_02_pagamento",
        arquivo_origem="todas_as_safras_e_anos",
    )

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-date", required=False, default=None, help="data de referência (opcional, formato YYYY-MM-DD)")
    parser.add_argument("--lake-root", default=None)
    args = parser.parse_args()
    main(args.ref_date, args.lake_root)