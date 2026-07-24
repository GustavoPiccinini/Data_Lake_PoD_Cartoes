"""
Pipeline principal do Data Lake PoD Cartões.

Uso:
    uv run python main.py
"""

from pathlib import Path
import subprocess
import sys
import time

# 1. Raiz do projeto e do Data Lake
ROOT = Path(__file__).parent.resolve()
LAKE_ROOT = (ROOT / "datalake").resolve()

# 2. Caminhos das pastas brutas oficiais
PASTA_RAW_FATURA = LAKE_ROOT / "raw" / "fatura"
PASTA_RAW_PAGAMENTO = LAKE_ROOT / "raw" / "pagamento"


def executar(comando: list[str]) -> None:
    """Executa subprocessos do Python exibindo tempo e tratando erros."""
    print("=" * 70)
    print("Executando:")
    print(" ".join(comando))
    print("=" * 70)

    inicio = time.perf_counter()

    resultado = subprocess.run(
        comando,
        cwd=ROOT
    )

    tempo = time.perf_counter() - inicio

    if resultado.returncode != 0:
        print("\nErro durante a execução do comando!")
        sys.exit(resultado.returncode)

    print(f"Concluído em {tempo:.2f} segundos\n")


def ultimo_csv(pasta_oficial: Path, dataset_tipo: str) -> Path:
    """
    Retorna o arquivo CSV mais recente da pasta oficial.
    Se a pasta oficial não existir ou estiver vazia, busca nos backups.
    """
    # 1. Tenta buscar na pasta oficial se ela existir
    if pasta_oficial.exists():
        arquivos = list(pasta_oficial.glob("*.csv"))
        if arquivos:
            return max(arquivos, key=lambda arq: arq.stat().st_mtime)

    # 2. Fallback: Procura nos diretórios de backup_01 e backup_02
    raw_base = LAKE_ROOT / "raw"
    backups = [
        raw_base / "backup_01" / dataset_tipo,
        raw_base / "backup_02" / dataset_tipo,
    ]

    for b_path in backups:
        if b_path.exists():
            arquivos_backup = list(b_path.glob("*.csv"))
            if arquivos_backup:
                escolhido = max(arquivos_backup, key=lambda arq: arq.stat().st_mtime)
                print(f"[AVISO main.py] Pasta oficial '{dataset_tipo}' indisponível ou vazia. Carregando da reserva: '{escolhido.name}'")
                return escolhido

    raise FileNotFoundError(
        f"Nenhum arquivo CSV foi encontrado para '{dataset_tipo}' na pasta oficial nem em backup_01/backup_02."
    )


def main():
    inicio_total = time.perf_counter()

    print("=" * 70)
    print("PIPELINE DATA LAKE - PoD CARTÕES")
    print("=" * 70)

    # Descobre os arquivos mais recentes (com fallback automático para backup)
    fatura_path = ultimo_csv(PASTA_RAW_FATURA, "fatura")
    pagamento_path = ultimo_csv(PASTA_RAW_PAGAMENTO, "pagamento")

    arquivo_fatura = fatura_path.name
    arquivo_pagamento = pagamento_path.name

    print(f"LAKE_ROOT..........: {LAKE_ROOT}")
    print(f"Arquivo Fatura.....: {arquivo_fatura}")
    print(f"Arquivo Pagamento..: {arquivo_pagamento}\n")

    # -----------------------------------------------------------------
    # Etapa 1: Trusted Fatura
    # -----------------------------------------------------------------
    executar([
        sys.executable,
        str(ROOT / "processing" / "01_fatura_trusted.py"),
        "--raw-file", arquivo_fatura,
        "--lake-root", str(LAKE_ROOT)
    ])

    # -----------------------------------------------------------------
    # Etapa 2: Trusted Pagamento
    # -----------------------------------------------------------------
    executar([
        sys.executable,
        str(ROOT / "processing" / "02_pagamento_trusted.py"),
        "--raw-file", arquivo_pagamento,
        "--lake-root", str(LAKE_ROOT)
    ])

    # -----------------------------------------------------------------
    # Etapa 3: Book de Variáveis
    # -----------------------------------------------------------------
    executar([
        sys.executable,
        str(ROOT / "processing" / "03_book_variaveis.py"),
        "--lake-root", str(LAKE_ROOT)
    ])

    total = time.perf_counter() - inicio_total

    print("=" * 70)
    print("PIPELINE FINALIZADA COM SUCESSO")
    print(f"Tempo Total: {total:.2f} segundos")
    print("=" * 70)


if __name__ == "__main__":
    main()