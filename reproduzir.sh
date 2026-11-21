#!/usr/bin/env bash
# Reproduz todos os resultados do desafio, a partir da raiz do projeto: bash reproduzir.sh
# Tempo total aproximado: 15 minutos (a referência escalar no conjunto grande leva uns 6).
set -euo pipefail

uv sync --frozen

# 1. Conjuntos do benchmark: mesma semente para todos. O "enorme" (1 milhão de linhas,
#    512 MB) não é pedido no enunciado: serve para conferir a tendência fora da faixa medida.
uv run --frozen python 03-hpc/gerar_entradas.py --rows 1000 --seed 2027 --output 03-hpc/dados/pequeno
uv run --frozen python 03-hpc/gerar_entradas.py --rows 10000 --seed 2027 --output 03-hpc/dados/medio
uv run --frozen python 03-hpc/gerar_entradas.py --rows 100000 --seed 2027 --output 03-hpc/dados/grande
uv run --frozen python 03-hpc/gerar_entradas.py --rows 1000000 --seed 2027 --output 03-hpc/dados/enorme

# 2. Correção: testes do material fornecido e da solução.
uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v

# 3. Custo da referência escalar, linha por linha.
uv run --frozen python 03-hpc/analise_escalar.py

# 4. Benchmark: todas as estratégias, em rodadas pareadas (grava 03-hpc/resultados/).
uv run --frozen python 03-hpc/benchmark.py

# 5. Tabelas e gráfico usados no relatório.
uv run --frozen python 03-hpc/resumo.py
