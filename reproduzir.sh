#!/usr/bin/env bash
# Executa o projeto de ponta a ponta, a partir da raiz:
#   bash reproduzir.sh          tudo: testes, análise, benchmark e resumo (~30 minutos)
#   bash reproduzir.sh rapido   demonstração: testes e benchmark com 1 mil e 10 mil linhas (~1 minuto)
# A saída de cada passo da execução completa também fica em 03-hpc/resultados/logs/.
set -euo pipefail

uv sync --frozen

gerar() {  # gerar <linhas> <pasta> [modelo]: conjunto sintético com a semente 2027
    uv run --frozen python 03-hpc/gerar_entradas.py --rows "$1" --seed 2027 --output "03-hpc/dados/$2" \
        --model "${3:-03-hpc/modelo/modelo.npz}"
}

if [ "${1:-}" = "rapido" ]; then
    gerar 1000 pequeno
    gerar 10000 medio
    uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v
    uv run --frozen python 03-hpc/benchmark.py --conjuntos pequeno medio --repeticoes 5 \
        --saida 03-hpc/resultados/demo
    exit 0
fi

LOGS=03-hpc/resultados/logs
mkdir -p "$LOGS"

# 1. Conjuntos do benchmark; o de 1 milhão de linhas confere a tendência além de 100 mil.
gerar 1000 pequeno
gerar 10000 medio
gerar 100000 grande
gerar 1000000 enorme

# 2. Correção: todos os testes.
uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v 2>&1 | tee "$LOGS/testes.log"

# 3. Custo da referência escalar, parte por parte.
uv run --frozen python 03-hpc/analise_escalar.py 2>&1 | tee "$LOGS/analise_escalar.log"

# 4. Benchmark: todas as estratégias, em rodadas pareadas.
uv run --frozen python 03-hpc/benchmark.py 2>&1 | tee "$LOGS/benchmark.log"

# 5. Experimento adicional: rede mais larga (64 → 1024 → 1), com 32 vezes mais conta por linha
#    e a mesma quantidade de dados por linha. A referência escalar fica de fora (lenta demais).
LARGO=03-hpc/modelo_largo/modelo.npz
uv run --frozen python 03-hpc/gerar_modelo.py --hidden 1024 --output "$LARGO"
gerar 1000 largo_pequeno "$LARGO"
gerar 10000 largo_medio "$LARGO"
gerar 100000 largo_grande "$LARGO"
uv run --frozen python 03-hpc/benchmark.py --modelo "$LARGO" \
    --conjuntos largo_pequeno largo_medio largo_grande --estrategias vetorizado fundido paralelo \
    --saida 03-hpc/resultados/largo 2>&1 | tee "$LOGS/largo.log"

# 6. Tabelas e gráfico do relatório.
uv run --frozen python 03-hpc/resumo.py
