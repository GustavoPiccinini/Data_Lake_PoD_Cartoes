from pathlib import Path

# Raiz do Projeto
BASE_DIR = Path(__file__).resolve().parent

# Caminho do Data Lake (Qualquer pessoa só precisa alterar esta linha se mudar de pasta)
LAKE_ROOT = BASE_DIR / "datalake"

# Mapeamento Centralizado de Pastas Brutas (Raw)
RAW_DIR = LAKE_ROOT / "raw"
FATURA_DIR = RAW_DIR / "fatura"
PAGAMENTO_DIR = RAW_DIR / "pagamento"

# Pastas de Backup/Reserva
BACKUP_DIRS = [
    RAW_DIR / "backup_01",
    RAW_DIR / "backup_02",
]