# Data Lake — PoD Cartões (v2, reconstruído)

Data Lake para a PoD Cartões: consolidação de faturas e pagamentos, cálculo
de inadimplência e Book de Variáveis (U1M/U3M/U6M/U12M) para modelagem de
risco de crédito.

Reconstrução da versão original (Colab + Google Drive) com foco em:
rodar localmente ou na AWS, ser parametrizável (sem editar código a cada
execução), e ter monitoria/linhagem/integridade — ver seção dedicada abaixo.

## O que mudou da v1 pra v2

| Problema na v1 | Correção na v2 |
|---|---|
| Trusted grava `tb_01fatura`/`tb_02pagamento`, Book lê `tb_01_fatura`/`tb_02_pagamento` (nunca se encontravam) | Nomes padronizados com underscore em toda a esteira |
| Copy-paste: `cast(id_cliente as bigint) as data_emissao` no preview do pagamento | Corrigido |
| Nome do CSV de origem e data de referência do Book hardcoded no código | `--raw-file` e `--ref-date` como argumentos |
| 100% acoplado a Google Colab/Drive (`drive.mount`, `!pip install`) | Roda local (`./datalake`) ou S3 via `LAKE_ROOT`, sem Colab |
| Dedup por concatenar `ref+dt_proc` como string e pegar o MAX | `ROW_NUMBER() OVER (PARTITION BY chave ORDER BY dt_proc DESC)` |
| Query do Book gerada com contador manual (`VAR_1, VAR_2, ...`) | Nomes de coluna descritivos (`valor_fatura_sum_pago_em_dia_u3m`), sem colisão |
| README citava Lambda/Athena/Power BI/"camada Gold", código não usava nada disso | README alinhado com o que o código realmente faz |
| Zero validação — dado ruim segue pra frente silenciosamente | Checks de integridade fail-fast (ver abaixo) |
| Sem lineage — não dá pra saber de onde veio um número no Refined | Toda gravação registra origem em `005_controle` |
| 200 partições de shuffle padrão do Spark rodando local com dataset pequeno (lento) | `spark.sql.shuffle.partitions=8` pra execução local |

Todas as correções acima foram **testadas de verdade** rodando o pipeline
localmente (não só revisão de código) — inclusive o cenário de reprocessar o
mesmo dia duas vezes (dedup continua correto) e o cenário de dado ruim
(pipeline trava com erro claro, em vez de deixar passar).

## Monitoria, Linhagem e Integridade

Isso é o `common/observability.py`, usado pelos 3 scripts de processing:

**Linhagem (`record_lineage`)** — toda gravação registra em `005_controle`:
qual tabela, quando rodou (`dt_proc`), quantos registros, **de qual camada e
arquivo de origem** vieram. Responde "de onde veio esse número no Refined?"
sem abrir o código.

**Integridade (`check_integrity`)** — a cada carga, 3 famílias de checks,
gravadas em `006_qualidade`:
1. **Chave única** — `(id_cliente, id_fatura)` não pode se repetir na mesma carga
2. **Nulo em coluna obrigatória** — ex.: `valor_fatura` não pode ser nulo
3. **Reconciliação** — a contagem da trusted deve bater com a do raw

Se qualquer check falhar, a função levanta `RuntimeError` e o pipeline para
**antes** de gravar o dado ruim — fail-fast, em vez de o problema só aparecer
3 camadas depois, no Book de Variáveis, sem pista de onde começou.

## Arquitetura

```
datalake/                       (local, git-ignored — ou s3://bucket via LAKE_ROOT)
├── 001_raw/
│   ├── fatura/                 # CSVs brutos de fatura
│   └── pagamento/               # CSVs brutos de pagamento
├── 002_trusted/
│   ├── tb_01_fatura/            # Parquet tipado, particionado por ref
│   └── tb_02_pagamento/
├── 003_refined/
│   ├── stage_fatura/            # classificação de pagamento + dias de atraso
│   └── book_fatura/             # Book de Variáveis (160 métricas/cliente)
├── 005_controle/                # linhagem: de onde veio cada tabela
└── 006_qualidade/               # resultado dos checks de integridade
```

O diagrama de arquitetura original (`PoD_Cartoes.drawio`) mostra o alvo em
produção (S3 + EMR); a v2 roda local por padrão e migra pra AWS trocando só
o `LAKE_ROOT`, sem mudar código — mesmo padrão usado no projeto Data Lake
E-commerce.

## Estrutura do repositório

```
├── pyproject.toml / uv.lock
├── common/
│   ├── lake.py                  # resolve local/S3, cria as pastas das camadas
│   └── observability.py         # linhagem + integridade
├── processing/
│   ├── 00_seed_sample_data.py   # popula 001_raw a partir de sample_data (teste sem dados reais)
│   ├── 01_fatura_trusted.py
│   ├── 02_pagamento_trusted.py
│   └── 03_book_variaveis.py
└── sample_data/                 # CSVs sintéticos (mesmo schema dos reais) pra testar
```

## Como rodar

```bash
uv sync
source .venv/bin/activate

# popula 001_raw com dados de exemplo (sem precisar dos arquivos reais)
python processing/00_seed_sample_data.py

# roda a esteira (troque os nomes de arquivo pelos que o seed imprimiu)
spark-submit processing/01_fatura_trusted.py --raw-file tb_faturas_XXXXXXXXXXXXXX.csv
spark-submit processing/02_pagamento_trusted.py --raw-file tb_pagamentos_XXXXXXXXXXXXXX.csv
spark-submit processing/03_book_variaveis.py --ref-date 2024-01-31
```

Com dados reais, é só colocar o CSV em `001_raw/fatura/` ou `001_raw/pagamento/`
manualmente (ou por um script de ingestão, como no projeto Data Lake
E-commerce) e passar o nome real em `--raw-file`.

### AWS

```bash
uv sync --extra aws
export LAKE_ROOT=s3://seu-bucket
spark-submit processing/01_fatura_trusted.py --raw-file tb_faturas_....csv
```

## Próximos passos possíveis

- Alertas de verdade (Slack/e-mail) quando `006_qualidade` registra `FALHOU`,
  em vez de só parar o pipeline
- Orquestração via Airflow (mesmo padrão do Data Lake E-commerce)
- Dashboard em Streamlit consumindo `book_fatura` (era o objetivo original do README)
