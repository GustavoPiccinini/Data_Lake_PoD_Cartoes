# Usa a imagem oficial leve do Python
FROM python:3.11-slim

# Evita criação de arquivos .pyc e força saída instantânea dos logs no terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala o Java OpenJDK (necessário para o Apache Spark) e utilitários básicos
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configura a variável de ambiente do Java dinamicamente no Linux
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Instala o 'uv' para gerenciamento de dependências
RUN pip install --no-cache-dir uv

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia arquivos de configuração de dependências
COPY pyproject.toml uv.lock ./

# Instala as dependências do projeto no ambiente do container
RUN uv sync --frozen || uv pip install --system .

# Copia todo o código do projeto
COPY . .

# Comando padrão ao iniciar o container
CMD ["uv", "run", "python", "main.py"]