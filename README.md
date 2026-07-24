# Data Lake — PoD Cartões

Data Lake e Plataforma Analytics para a PoD Cartões: consolidação de faturas e pagamentos, cálculo de inadimplência, Feature Store / Book de Variáveis (U1M/U3M/U6M/U12M) para risco de crédito e Dashboard Executivo Interativo.

O projeto foi construído com foco em:
- Arquitetura Medalhão (Raw, Trusted e Refined).
- Execução local ou na nuvem AWS (via `LAKE_ROOT`).
- Parametrização dinâmica de cargas.
- Observabilidade de dados (Monitoria, Linhagem e Integridade *fail-fast*).
- Dashboard executivo responsivo construído em Streamlit + DuckDB.

---

##  Monitoria, Linhagem e Integridade

Através do módulo `common/observability.py`, a esteira garante:

- **Linhagem (`record_lineage`):** Registra na camada `005_controle` a tabela de destino, timestamp de processamento (`dt_proc`), volumetria e arquivo/camada de origem. Responde "de onde veio esse dado?" de forma transparente.
- **Integridade (`check_integrity`):** Registra na camada `006_qualidade` 3 validações cruciais antes de persistir os dados:
  1. **Chave Única:** Valida duplicidade de `(id_cliente, id_fatura)`.
  2. **Nulos Obrigatórios:** Garante preenchimento de colunas críticas (ex: `valor_fatura`).
  3. **Reconciliação Volumétrica:** Contagem da camada `Trusted` deve bater com a `Raw`.

> **Mecanismo Fail-Fast:** Se qualquer check falhar, a pipeline interrompe a execução com um `RuntimeError` explicativo **antes** de gravar dados corrompidos na camada seguinte.

---

##  Arquitetura de Pastas e Dados

```text
DATA_LAKE_POD_CARTOES/
│
├── Arquitetura/                  <-- Diagramas e documentação (PoD_Cartoes.drawio)
├── common/                       <-- Módulos reutilizáveis
│   ├── lake.py                   <-- Resolução de caminhos local/S3
│   └── observability.py          <-- Registros de linhagem e qualidade
│
├── datalake/                     <-- Arquivos Parquet / Datalake camada Medalhão 
│   ├── 001_raw/                  <-- CSVs brutos de faturas e pagamentos
│   ├── 002_trusted/              <-- Parquet tipado, particionado por referência
│   ├── 003_refined/              <-- Stage de atraso e Book de Variáveis
│   ├── 005_controle/             <-- Registros de linhagem de dados
│   └── 006_qualidade/            <-- Histórico dos checks de integridade
│
├── notebooks/                    <-- Exploração e análises
│   └── analytics/                <-- EDA, estudos de Churn e relatórios
│
├── processing/                   <-- Scripts de ETL (PySpark)
│   ├── 00_seed_sample_data.py    <-- Gerador de dados sintéticos para testes locais
│   ├── 01_fatura_trusted.py      <-- Processa dados reais de fatura (Raw -> Trusted Parquet)
│   ├── 02_pagamento_trusted.py   <-- Processa dados reais de pagamento (Raw -> Trusted Parquet)
│   └── 03_book_variaveis.py      <-- Consolida faturas/pagamentos e cria o book de pagamentos(Refined)
│
├── Streamlit_Dashboard/          <-- Camada de Consumo e Analytics
│   └── dashboard_pod_cartoes.py  <-- Dashboard  (Streamlit + DuckDB)
│
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml / uv.lock
└── README.md