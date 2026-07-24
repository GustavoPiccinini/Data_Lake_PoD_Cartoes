"""
Monitoria, Linhagem e Integridade — camada de observabilidade do Data Lake.

- LINHAGEM (lineage): toda gravação registra DE ONDE veio o dado (camada e
  arquivo de origem), QUANDO rodou (dt_proc) e QUANTOS registros — grava em
  controle. Isso responde "de onde veio esse número na camada Refined?"
  sem precisar abrir o código.

- INTEGRIDADE (quality): checks executáveis rodados a cada carga — chave
  duplicada, nulos em coluna obrigatória, e reconciliação de contagem entre
  camadas. Resultado vai pra qualidade; se algum check falhar, a função levanta
  erro e interrompe o pipeline (fail-fast).
"""
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import count as spark_count
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import LongType, StringType, StructField, StructType

from common.lake import layer_path

LINEAGE_SCHEMA = StructType([
    StructField("tabela", StringType(), True),
    StructField("dt_proc", StringType(), True),
    StructField("qtd_registros", LongType(), True),
    StructField("camada_origem", StringType(), True),
    StructField("arquivo_origem", StringType(), True),
    StructField("executado_em", StringType(), True),
])

QUALITY_SCHEMA = StructType([
    StructField("tabela", StringType(), True),
    StructField("dt_proc", StringType(), True),
    StructField("status", StringType(), True),
    StructField("falhas", StringType(), True),
    StructField("total_registros", LongType(), True),
    StructField("chaves_duplicadas", LongType(), True),
])


def record_lineage(
    spark: SparkSession,
    lake_root: str,
    tabela: str,
    dt_proc: str,
    qtd_registros: int,
    camada_origem: str,
    arquivo_origem: str,
):
    """Grava uma linha de linhagem na camada de controle: essa tabela, gerada
    nesse dt_proc, com N registros, veio de tal camada/arquivo de origem."""
    ctl_path = layer_path(lake_root, "controle")

    # Método nativo e resiliente (sem serialização de objetos Python em RDDs)
    row = (
        spark.range(1)
        .select(
            lit(str(tabela)).cast("string").alias("tabela"),
            lit(str(dt_proc)).cast("string").alias("dt_proc"),
            lit(int(qtd_registros)).cast("long").alias("qtd_registros"),
            lit(str(camada_origem)).cast("string").alias("camada_origem"),
            lit(str(arquivo_origem)).cast("string").alias("arquivo_origem"),
            current_timestamp().cast("string").alias("executado_em"),
        )
    )

    row.write.mode("append").parquet(ctl_path)
    print(f"[linhagem] {tabela} <- {camada_origem}/{arquivo_origem} ({qtd_registros} registros, dt_proc={dt_proc})")


def check_integrity(
    spark: SparkSession,
    df: DataFrame,
    tabela: str,
    lake_root: str,
    dt_proc: str,
    key_cols: list,
    not_null_cols: list,
    reconcile_against: int | None = None,
    reconcile_tolerance: float = 0.0,
) -> dict:
    """Roda 3 famílias de checks e grava o resultado na camada de qualidade.
    Levanta RuntimeError se algum check crítico falhar (fail-fast).
    """
    resultados = {}
    falhas = []

    total = df.count()
    resultados["total_registros"] = total

    # 1) Unicidade de chave
    duplicados = (
        df.groupBy(*key_cols)
        .agg(spark_count(lit(1)).alias("qtd"))
        .where("qtd > 1")
        .count()
    )
    resultados["chaves_duplicadas"] = duplicados
    if duplicados > 0:
        falhas.append(f"{duplicados} chave(s) duplicada(s) em {key_cols}")

    # 2) Nulos em colunas obrigatórias
    nulos = {}
    for c in not_null_cols:
        qtd_nulos = df.where(df[c].isNull()).count()
        nulos[c] = qtd_nulos
        if qtd_nulos > 0:
            falhas.append(f"{qtd_nulos} nulo(s) na coluna obrigatória '{c}'")
    resultados["nulos_por_coluna"] = nulos

    # 3) Reconciliação de contagem com a camada anterior
    if reconcile_against is not None:
        diff = abs(total - reconcile_against)
        limite = reconcile_against * reconcile_tolerance
        resultados["reconciliacao"] = {
            "esperado": reconcile_against,
            "obtido": total,
            "diferenca": diff,
        }
        if diff > limite:
            falhas.append(
                f"reconciliação falhou: esperado ~{reconcile_against}, obtido {total} (diff={diff})"
            )

    resultados["status"] = "FALHOU" if falhas else "OK"
    resultados["falhas"] = falhas

    # Grava o resultado na camada de qualidade via API SQL nativa
    qual_path = layer_path(lake_root, "qualidade")
    
    row = (
        spark.range(1)
        .select(
            lit(str(tabela)).cast("string").alias("tabela"),
            lit(str(dt_proc)).cast("string").alias("dt_proc"),
            lit(str(resultados["status"])).cast("string").alias("status"),
            lit(str("; ".join(falhas))).cast("string").alias("falhas"),
            lit(int(total)).cast("long").alias("total_registros"),
            lit(int(duplicados)).cast("long").alias("chaves_duplicadas"),
        )
    )

    row.write.mode("append").parquet(qual_path)

    if falhas:
        print(f"[integridade] ✗ {tabela}: {resultados['status']} -> {falhas}")
        raise RuntimeError(f"Checks de integridade falharam para '{tabela}': {falhas}")

    print(f"[integridade] ✓ {tabela}: OK ({total} registros, chaves únicas, sem nulos obrigatórios)")
    return resultados