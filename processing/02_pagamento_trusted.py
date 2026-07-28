"""
Camada Trusted — Pagamento (raw -> 002_trusted/tb_02_pagamento)
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
        .appName("pod_cartoes_pagamento_trusted")
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
    raw_path = get_raw_file_path(lake_root, "pagamento", raw_file)

    df_pagamento = spark.read.option("header", "true").csv(raw_path)
    qtd_raw = df_pagamento.count()
    print(f"[pagamento] {qtd_raw} registros lidos de {raw_path}")
    df_pagamento.createOrReplaceTempView("df_pagamento")

    # SQL Ajustado: Chaves convertidas para INT e Safra com date_format seguro
    df_pagamento_format = spark.sql(f"""
        select
            cast(regexp_replace(cast(id_pagamento as string), '[^0-9]', '') as int) as id_pagamento,
            cast(regexp_replace(cast(id_fatura as string), '[^0-9]', '') as int) as id_fatura,
            cast(id_cliente as int) as id_cliente,
            "{dt_proc}" as dt_proc,
            date_format(cast(dt_pagamento as date), 'yyyyMM') as ref,
            cast(dt_pagamento as date) as data_pagamento,
            cast(valor_pagamento as decimal(15,2)) as valor_pagamento
        from df_pagamento
    """)
    df_pagamento_format.cache()

    check_integrity(
        spark, df_pagamento_format,
        tabela="tb_02_pagamento",
        lake_root=lake_root,
        dt_proc=dt_proc,
        key_cols=["id_cliente", "id_fatura", "id_pagamento", "dt_proc"],
        not_null_cols=["id_cliente", "id_fatura", "id_pagamento", "data_pagamento", "valor_pagamento"],
        reconcile_against=qtd_raw,
    )

    trusted_path = layer_path(lake_root, "trusted", "tb_02_pagamento")
    df_pagamento_format.write.mode("append").partitionBy("ref").parquet(trusted_path)
    print(f"[pagamento] gravado em {trusted_path}")

    # Libera o cache de memória alocado
    df_pagamento_format.unpersist()

    record_lineage(
        spark, lake_root,
        tabela="tb_02_pagamento",
        dt_proc=dt_proc,
        qtd_registros=qtd_raw,
        camada_origem="001_raw/pagamento",
        arquivo_origem=raw_file,
    )

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", required=True, help="nome do CSV dentro de raw/pagamento/")
    parser.add_argument("--lake-root", default=None)
    args = parser.parse_args()
    main(args.raw_file, args.lake_root)