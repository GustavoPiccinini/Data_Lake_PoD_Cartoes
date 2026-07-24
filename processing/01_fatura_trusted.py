"""
Camada Trusted — Fatura (raw -> trusted/tb_01_fatura)
Suporte para fallback automatico em pastas de reserva/backup.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.lake import layer_path, resolve_lake_root  # noqa: E402
from common.observability import check_integrity, record_lineage  # noqa: E402


def get_raw_file_path(lake_root: str, dataset: str, raw_filename: str) -> str:
    """Busca o arquivo na pasta oficial raw. Se nao encontrar, busca nas pastas de backup."""
    raw_dir = Path(layer_path(lake_root, "raw", dataset))
    official_file = raw_dir / raw_filename

    if official_file.exists():
        return str(official_file)

    # Busca em repositorios de reserva dentro da camada raw
    raw_base_dir = Path(layer_path(lake_root, "raw"))
    backup_sources = [
        raw_base_dir / "backup_01" / dataset / raw_filename,
        raw_base_dir / "backup_02" / dataset / raw_filename,
    ]

    for backup_file in backup_sources:
        if backup_file.exists():
            print(f"[AVISO] Arquivo '{raw_filename}' nao encontrado na pasta oficial. Carregando da reserva: '{backup_file}'")
            return str(backup_file)

    raise FileNotFoundError(
        f"Arquivo '{raw_filename}' nao encontrado em '{official_file}' nem nos backups (backup_01, backup_02)."
    )


def main(raw_file: str, lake_root: str | None):
    lake_root = resolve_lake_root(lake_root)

    spark = (
        SparkSession.builder
        .appName("pod_cartoes_fatura_trusted")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.file", "org.apache.hadoop.mapreduce.lib.output.FileOutputCommitterFactory")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.fs.permissions.umask-mode", "022")
        .getOrCreate()
    )
    
    dt_proc = datetime.now().strftime("%Y%m%d%H%M%S")

    # Resolve o caminho com suporte a busca em pastas de reserva
    raw_path = get_raw_file_path(lake_root, "fatura", raw_file)

    df_fatura = spark.read.option("header", "true").csv(raw_path)
    qtd_raw = df_fatura.count()
    print(f"[fatura] {qtd_raw} registros lidos de {raw_path}")
    df_fatura.createOrReplaceTempView("df_fatura")

    df_fatura_format = spark.sql(f"""
        select
            cast(regexp_replace(cast(id_fatura as string), '[^0-9]', '') as bigint) as id_fatura,
            cast(id_cliente as bigint) as id_cliente,
            "{dt_proc}" as dt_proc,
            substring(replace(cast(dt_emissao_fatura as string), '-', ''), 1, 6) as ref,
            cast(dt_emissao_fatura as date) as data_emissao,
            cast(dt_vencimento_fatura as date) as data_vencimento,
            cast(valor_fatura as decimal(15,2)) as valor_fatura,
            cast(0 as decimal(15,2)) as valor_pagamento_minimo
        from df_fatura
    """)
    df_fatura_format.cache()

    check_integrity(
        spark, df_fatura_format,
        tabela="tb_01_fatura",
        lake_root=lake_root,
        dt_proc=dt_proc,
        key_cols=["id_cliente", "id_fatura", "dt_proc"],
        not_null_cols=["id_cliente", "id_fatura", "data_emissao", "valor_fatura"],
        reconcile_against=qtd_raw,
    )

    trusted_path = layer_path(lake_root, "trusted", "tb_01_fatura")
    df_fatura_format.write.mode("append").partitionBy("ref").parquet(trusted_path)
    print(f"[fatura] gravado em {trusted_path}")

    record_lineage(
        spark, lake_root,
        tabela="tb_01_fatura",
        dt_proc=dt_proc,
        qtd_registros=qtd_raw,
        camada_origem="001_raw/fatura",
        arquivo_origem=raw_file,
    )

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", required=True, help="nome do CSV dentro de raw/fatura/")
    parser.add_argument("--lake-root", default=None)
    args = parser.parse_args()
    main(args.raw_file, args.lake_root)