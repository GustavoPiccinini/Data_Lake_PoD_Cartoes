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
LAKE_ROOT = ROOT / "datalake"

# 2. Caminhos das pastas brutas
PASTA_RAW_FATURA = LAKE_ROOT / "raw" / "fatura"
PASTA_RAW_PAGAMENTO = LAKE_ROOT / "raw" / "pagamento"


def executar(comando: list):
    """Executa subprocessos do Python exibindo tempo e tratando erros."""
    print("=" * 70)
    print("Executando:")
    print(" ".join(str(c) for c in comando))
    print("=" * 70)

    inicio = time.perf_counter()

    resultado = subprocess.run(
        comando,
        cwd=ROOT
    )

    tempo = time.perf_counter() - inicio

    if resultado.returncode != 0:
        print("\n Erro durante a execução do comando!")
        sys.exit(resultado.returncode)

    print(f" Concluído em {tempo:.2f} segundos\n")


def ultimo_csv(pasta: Path) -> Path:
    """Retorna o arquivo CSV mais recente dentro do diretório especificado."""
    arquivos = list(pasta.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {pasta}")
    return max(arquivos, key=lambda arq: arq.stat().st_mtime)


def main():
    inicio_total = time.perf_counter()

    print("=" * 70)
    print("PIPELINE DATA LAKE - PoD CARTÕES")
    print("=" * 70)

    # Descobre os arquivos mais recentes em cada subpasta de forma dinâmica
    arquivo_fatura = ultimo_csv(PASTA_RAW_FATURA).name
    arquivo_pagamento = ultimo_csv(PASTA_RAW_PAGAMENTO).name

    print(f"LAKE_ROOT..........: {LAKE_ROOT}")
    print(f"Arquivo Fatura.....: {arquivo_fatura}")
    print(f"Arquivo Pagamento..: {arquivo_pagamento}\n")

    # -----------------------------------------------------------------
    # Etapa 1: Trusted Fatura
    # -----------------------------------------------------------------
    executar([
        sys.executable,
        "processing/01_fatura_trusted.py",
        "--raw-file", arquivo_fatura,
        "--lake-root", str(LAKE_ROOT)
    ])

    # -----------------------------------------------------------------
    # Etapa 2: Trusted Pagamento
    # -----------------------------------------------------------------
    executar([
        sys.executable,
        "processing/02_pagamento_trusted.py",
        "--raw-file", arquivo_pagamento,
        "--lake-root", str(LAKE_ROOT)
    ])

    # -----------------------------------------------------------------
    # Etapa 3: Book de Variáveis
    # -----------------------------------------------------------------
    executar([
        sys.executable,
        "processing/03_book_variaveis.py",
        "--lake-root", str(LAKE_ROOT)
    ])

    total = time.perf_counter() - inicio_total

    print("=" * 70)
    print("PIPELINE FINALIZADA COM SUCESSO")
    print(f"Tempo Total: {total:.2f} segundos")
    print("=" * 70)


if __name__ == "__main__":
    main()