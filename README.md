# Data Lakehouse & Feature Store — Empresa de Cartões

> Projeto de Engenharia de Dados para processamento, governança e disponibilização de dados analíticos de cartões de crédito utilizando PySpark.

**Dashboard Online:** https://datalakecartoes.streamlit.app/

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Problema de Negócio](#2-problema-de-negócio)
3. [Objetivos](#3-objetivos)
4. [Tecnologias](#4-tecnologias)
5. [Arquitetura da Solução](#5-arquitetura-da-solução)
6. [Arquitetura Medalhão](#6-arquitetura-medalhão)
7. [Pipeline ETL](#7-pipeline-etl)
8. [Feature Store](#8-feature-store)
9. [Regras de Negócio](#9-regras-de-negócio)
10. [Governança e Qualidade](#10-governança-e-qualidade)
11. [Estrutura das Tabelas](#11-estrutura-das-tabelas)
12. [Estrutura do Projeto](#12-estrutura-do-projeto)
13. [Dashboard Analítico](#13-dashboard-analítico)
14. [Como Executar](#14-como-executar)
15. [Roadmap](#15-roadmap)
16. [Autor](#16-autor)

---

## 1. Visão Geral

Este projeto implementa uma arquitetura Data Lakehouse para consolidar dados de faturas e pagamentos de cartão de crédito.

Os dados percorrem uma pipeline em PySpark composta pelas camadas Raw, Trusted e Refined, gerando uma Feature Store utilizada por modelos de Machine Learning e um dashboard analítico desenvolvido em Streamlit.

---

## 2. Problema de Negócio

A PoD Cartões enfrentava desafios comuns em ecossistemas de grande volume transacional:
* **Dados Descentralizados**: Registros de faturas e pagamentos chegavam em formatos brutos (CSV), com risco de inconsistências e duplicatas.
* **Criar books de variáveis para  Área de Negócio**: 
* **Falta de Rastreabilidade**: Necessidade de um portal de observabilidade para auditar a linhagem do dado e checar duplicatas antes do consumo executivo.

---

## 3. Objetivos

* Centralizar dados financeiros em formato colunar de alta performance.
* Padronizar processos ETL.
* Garantir a qualidade dos dados .
* Registrar a linhagem completa da pipeline.
* Criar Books de Variáveis para modelos de crédito (Feature Store).
* Disponibilizar indicadores executivos em tempo real.

---

## 4. Tecnologias

| Categoria | Ferramenta |
| :--- | :--- |
| Linguagem | Python |
| Processamento | PySpark |
| Armazenamento | Parquet |
| Consultas | SQL |
| Dashboard | Streamlit |
| Analytics Engine | DuckDB |
| Gerenciador de Pacotes | uv |

---

## 5. Arquitetura da Solução

```text
[CSV Origem / Backup]
       │
       ▼
  [001_raw] ────────┐
       │             │
       ▼             ▼
 [002_trusted] ──> [005_controle / 006_qualidade]
       │
       ▼
 [003_refined]
       │
       ├────────────────────────┐
       ▼                        ▼
 [Feature Store]     [Dashboard Streamlit / DuckDB]
```


## 6. Arquitetura Medalhão

| Camada | Diretório | Finalidade |
| :--- | :--- | :--- |
| **001_raw** | `datalake/001_raw/` | Ingestão dos arquivos brutos recebidos dos sistemas legados e backups. |
| **002_trusted** | `datalake/002_trusted/` | Tratamento, tipagem, padronização e deduplicação (`ref`, `dt_proc`, `id_cliente`). |
| **003_refined** | `datalake/003_refined/` | Feature Store (Book de Variáveis) e agregações de negócio. |
| **005_controle** | `datalake/005_controle/` | Registro de linhagem e auditoria da pipeline (`record_lineage`). |
| **006_qualidade** | `datalake/006_qualidade/` | Validações e Quality Gates de integridade (`check_integrity`). |

---

## 7. Pipeline ETL

1. Ingestão dos arquivos CSV com mecanismo de fallback para diretórios de backup.
2. Validação estrutural e conversão de tipos de dados no PySpark.
3. Padronização e ordenação física das colunas primárias (`[ref, dt_proc, id_cliente]`).
4. Deduplicação lógica utilizando Window Functions (`dedup_latest`).
5. Escrita otimizada em Parquet particionado por safra (`ref`).
6. Cálculo de agregações e geração do Book de Variáveis (Refined).
7. Auditoria em paralelo gravando logs de linhagem e testes de qualidade.
8. Consumo para Análise e modelos de Machine Learning.

---

## 8. Feature Store

### Janelas Temporais Avaliadas
* **total**: Histórico completo acumulado do cliente.
* **u1m**: Último 1 mês a partir da data de referência.
* **u3m**: Últimos 3 meses.
* **u6m**: Últimos 6 meses.
* **u12m**: Últimos 12 meses.

### Métricas Calculadas (+200 Colunas)
* **Frequência**: Quantidade de faturas (`qtd_transacoes_*`).
* **Volume Financeiro**: Valor total, médio, mínimo e máximo (`valor_fatura_*`).
* **Comportamento de Atraso**: Dias médios, mínimos e máximos de atraso (`dias_atraso_*`).
* **Status e Percentuais**: Proporção de pagamentos em dia, parciais e inadimplentes.

---

## 9. Regras de Negócio

| Status | Descrição |
| :--- | :--- |
| **PAGO_EM_DIA** | Pagamento realizado exatamente na data de vencimento. |
| **PAGAMENTO_ANTECIPADO** | Pagamento efetuado antes da data de vencimento. |
| **PAGO_EM_ATRASO** | Pagamento realizado após a data de vencimento. |
| **PAGAMENTO_PARCIAL** | Valor pago inferior ao valor total da fatura. |
| **NAO_PAGO** | Sem registro de pagamento no período (Inadimplência). |

---

## 10. Governança e Qualidade

* **Linhagem da Pipeline**: Registro automático de volumetria e origem dos dados via `record_lineage`.
* **Quality**: O módulo `check_integrity` valida unicidade de chaves e ausência de nulos em campos obrigatórios antes da escrita.
* **Deduplicação Dinâmica**: Função `dedup_latest()` utilizando `Window.partitionBy()` ordenado por `dt_proc DESC` para selecionar apenas a versão mais recente.
* **Integridade de Ingestão**: Busca automática em diretórios de backup (`backup_01` / `backup_02`) caso os arquivos da `raw` não estejam disponíveis.

---

## 11. Estrutura das Tabelas

### `tb_01_fatura` (Trusted)
* `ref` (STRING): Safra da emissão (YYYYMM) - Chave de Partição.
* `dt_proc` (STRING): Timestamp de processamento.
* `id_cliente` (BIGINT): Identificador único do cliente.
* `id_fatura` (BIGINT): Identificador da fatura.
* `data_emissao` (DATE): Data de emissão.
* `data_vencimento` (DATE): Data de vencimento.
* `valor_fatura` (DOUBLE): Valor total da fatura.
* `valor_pagamento_minimo` (DOUBLE): Valor do pagamento mínimo.

### `tb_02_pagamento` (Trusted)
* `ref` (STRING): Safra do pagamento (YYYYMM) - Chave de Partição.
* `dt_proc` (STRING): Timestamp de processamento.
* `id_cliente` (BIGINT): Identificador único do cliente.
* `id_fatura` (BIGINT): Identificador da fatura.
* `id_pagamento` (BIGINT): Identificador do pagamento.
* `data_pagamento` (DATE): Data do pagamento.
* `valor_pagamento` (DOUBLE): Valor pago.

### `book_fatura` (Refined)
* `ref` (STRING): Safra do book - Chave de Partição.
* `dt_proc` (STRING): Timestamp de processamento.
* `id_cliente` (BIGINT): Identificador único do cliente.
* `+200 colunas de features`: Mapeamento comportamental em janelas móveis.


---

## 12. Estrutura do Projeto

```text
DATA_LAKE_POD_CARTOES/
│
├── Arquitetura/                  <-- Diagramas e especificações técnicas
├── common/                       <-- Módulos de infraestrutura e observabilidade
│   ├── lake.py                   <-- Resolução dinâmica de caminhos (Local / Cloud AWS S3)
│   └── observability.py          <-- Módulo de Linhagem e Check de Integridade
│
├── datalake/                     <-- Estrutura física das camadas em Parquet
│   ├── 001_raw/                  <-- Ingestão bruta (fatura, pagamento, backups)
│   ├── 002_trusted/              <-- Tabelas Parquet formatadas (tb_01_fatura, tb_02_pagamento)
│   ├── 003_refined/              <-- Stage e Feature Store (book_fatura)
│   ├── 005_controle/             <-- Auditoria de Linhagem
│   └── 006_qualidade/            <-- Logs de Quality Gates
│
├── docs/                         <-- Imagens e capturas de tela do dashboard
│
├── notebooks/                    <-- Análises exploratórias e prototipagem
│   └── analytics/                <-- Estudos de EDA e Inadimplência
│
├── processing/                   <-- Módulos de ETL em PySpark
│   ├── 01_fatura_trusted.py      <-- Carga e validação da Fatura
│   ├── 02_pagamento_trusted.py   <-- Carga e validação do Pagamento
│   └── 03_book_variaveis.py      <-- Consolidação e geração do Book de Crédito
│
├── Streamlit_Dashboard/          <-- Camada de Consumo Analytics
│   └── dashboard_pod_cartoes.py  <-- Aplicação Streamlit com Engine DuckDB
│
├── main.py                       <-- Orquestrador principal da pipeline
├── pyproject.toml / uv.lock      <-- Gerenciamento de dependências via UV
└── README.md                     <-- Documentação principal
```

---

## 13. Dashboard Analítico

Painéis disponíveis na aplicação Streamlit:
* **Visão Geral**: KPIs consolidados de faturamento, pagamentos e volume devedor.
* **Churn & Atividade**: Acompanhamento da retenção e inatividade da base.
* **Faturas**: Faixas de valor de fatura e comportamento de pagamento mínimo.
* **Pagamentos**: Distribuição por faixas de atraso (1-30d, 31-90d, >90d).
* **Inadimplência & Perfil**: Ranking de clientes de maior risco e histórico completo.
* **Consulta Direta**: Pesquisa individual de histórico por ID do Cliente.
* **Feature Store (Book)**: Inspecção das variáveis preparadas para Machine Learning.
* **Qualidade dos Dados**: Portal de observabilidade e auditoria de duplicatas.

---

## 14. Como Executar

### Clonar o repositório e sincronizar o ambiente
```bash
git clone [https://github.com/GustavoPiccinini/datalake-pod-cartoes.git](https://github.com/GustavoPiccinini/Data_Lake_PoD_Cartoes.git)

cd datalake-pod-cartoes
uv sync
```

### Executar a Pipeline ETL do Lakehouse
```bash
uv run python main.py
```

### Executar o Dashboard Analítico
```bash
uv run streamlit run Streamlit_Dashboard/dashboard_pod_cartoes.py
```

---

## 15. Roadmap

- [x] Ingestão e tratamento da Camada Trusted.
- [x] Construção da Feature Store com janelas móveis (Refined).
- [x] Integração de Observabilidade (Linhagem e Integridade).
- [x] Dashboard analítico em Streamlit + DuckDB.


---

## 16. Autor

**Gustavo Augusto Piccinini**

* **LinkedIn:** https://www.linkedin.com/in/gustavoapiccinini
* **GitHub:** https://github.com/GustavoPiccinini

---

## Licença

Projeto desenvolvido para estudos na PoD Academy, portfólio e demonstração de competências em Engenharia de Dados.
Os dados são fictícios, não contem informações que ferem lei da LGPD e podem ser reproduzidos para estudos.
