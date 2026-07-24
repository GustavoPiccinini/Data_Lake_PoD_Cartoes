"""
Resolve o storage do Data Lake PoD Cartões: local (pastas no PC) ou S3.

LAKE_ROOT pode ser:
  - "./datalake" ou caminho absoluto -> modo local
  - "s3://nome-do-bucket"             -> modo AWS

Prioridade: argumento cli_value > variável de ambiente LAKE_ROOT > local por padrão.

Camadas:
    001_raw        -> CSVs brutos (fatura, pagamento)
    002_trusted    -> Parquet tipado (tb_01_fatura, tb_02_pagamento)
    003_refined    -> Stage + Book de Variáveis
    005_controle   -> Contagem de registros por execução (auditoria)
    006_qualidade  -> Resultado dos checks de integridade
"""
import os
import sys
from pathlib import Path
from typing import Union

from pyspark.sql import SparkSession

# Raiz do projeto (duas pastas acima deste arquivo: common/lake.py -> raiz do projeto)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_ROOT = str(PROJECT_ROOT / "datalake")

LAYERS = {
    "raw": "raw",
    "trusted": "trusted",
    "refined": "refined",
    "controle": "controle",
    "qualidade": "qualidade",
}


def get_spark_session(app_name: str = "DataLakePipeline") -> SparkSession:
    """
    Inicializa a SparkSession de forma dinâmica e totalmente portátil.
    O uso do 'sys.executable' garante que o Spark utilize o executável do Python correto,
    seja no Windows local (python.exe) ou no Container Docker Debian (python).
    """
    py_exec = sys.executable

    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.pyspark.python", py_exec)
        .config("spark.pyspark.driver.python", py_exec)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def resolve_lake_root(cli_value: Union[str, Path, None] = None) -> str:
    """Resolve o caminho raiz do Data Lake priorizando CLI > Env Var > Default Local."""
    if cli_value:
        return str(cli_value).rstrip("/\\")
    
    env_value = os.environ.get("LAKE_ROOT")
    if env_value:
        return env_value.rstrip("/\\")
        
    return DEFAULT_LOCAL_ROOT


def is_s3(lake_root: str) -> bool:
    """Verifica se o caminho do Data Lake aponta para um bucket S3."""
    return str(lake_root).startswith("s3://")


def layer_path(lake_root: Union[str, Path], layer: str, dataset: str = "") -> str:
    """
    Monta (e cria, se local) o caminho de uma camada/dataset do Data Lake.
    Garante suporte nativo tanto para caminhos S3 quanto para Windows/Linux locais.
    """
    lake_root_str = str(lake_root)

    if layer not in LAYERS:
        raise ValueError(f"Camada inválida: '{layer}'. Opções válidas: {list(LAYERS.keys())}")

    layer_folder = LAYERS[layer]

    # Tratamento para S3 (força o uso de barras normais /)
    if is_s3(lake_root_str):
        parts = [lake_root_str.rstrip("/"), layer_folder]
        if dataset:
            parts.append(dataset)
        return "/".join(parts)

    # Tratamento para Sistema de Arquivos Local (Windows/Linux)
    path_obj = Path(lake_root_str) / layer_folder
    if dataset:
        path_obj = path_obj / dataset

    full_path = str(path_obj.resolve())

    # Garante que as pastas locais existam
    os.makedirs(full_path, exist_ok=True)

    return full_path