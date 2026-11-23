# Ambiente do projeto: Python 3.12 e as dependências fixadas em uv.lock.
#   docker build -t desafio-03-hpc .
#   docker run --rm --shm-size=2g desafio-03-hpc                            tudo (~30 min)
#   docker run --rm --shm-size=2g desafio-03-hpc bash reproduzir.sh rapido  demonstração (~1 min)
# --shm-size: a inferência paralela guarda X e Y em memória compartilhada (/dev/shm). Com o
# padrão do Docker (64 MB), os blocos grandes iriam para arquivos em disco e os tempos mudariam.
FROM ghcr.io/astral-sh/uv:0.11.32-python3.12-trixie-slim

# Usa o Python 3.12 da imagem, sem baixar outro.
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /projeto

# Dependências primeiro: a camada só é refeita quando uv.lock muda.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen

COPY . .
CMD ["bash", "reproduzir.sh"]
