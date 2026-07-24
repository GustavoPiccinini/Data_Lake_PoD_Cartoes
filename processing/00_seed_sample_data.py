"""
Copia os CSVs de sample_data/ para 001_raw/, com o nome no padrão que os
scripts de trusted esperam (útil pra testar o pipeline sem os dados reais).

Uso:
    python processing/00_seed_sample_data.py
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.lake import layer_path, resolve_lake_root  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seed(lake_root: str | None = None):
    lake_root = resolve_lake_root(lake_root)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    mapping = {
        "fatura": ("tb_faturas_sample.csv", f"tb_faturas_{ts}.csv"),
        "pagamento": ("tb_pagamentos_sample.csv", f"tb_pagamentos_{ts}.csv"),
    }

    for dataset, (src_name, dst_name) in mapping.items():
        src = os.path.join(PROJECT_ROOT, "sample_data", src_name)
        dst_dir = layer_path(lake_root, "raw", dataset)
        dst = os.path.join(dst_dir, dst_name)
        shutil.copy(src, dst)
        print(f"[{dataset}] {src} -> {dst}")

    print("\nSeed concluído com sucesso!\n")

    return (
            mapping["fatura"][1],
            mapping["pagamento"][1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake-root", default=None)
    args = parser.parse_args()
    seed(args.lake_root)





