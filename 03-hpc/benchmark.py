"""Benchmark das estratégias de inferência em CPU: escalar, vetorizada, paralela e fundida.

Uso (a partir da raiz do projeto, depois de gerar os conjuntos; veja reproduzir.sh):
    uv run --frozen python 03-hpc/benchmark.py
    uv run --frozen python 03-hpc/benchmark.py --conjuntos pequeno medio --repeticoes 5
    OPENBLAS_NUM_THREADS=4 uv run --frozen python 03-hpc/benchmark.py \\
        --estrategias vetorizado fundido --saida 03-hpc/resultados/blas_4_threads

Protocolo:
- Cada configuração tem 1 execução de aquecimento e `--repeticoes` execuções medidas com
  time.perf_counter. Todas as linhas do resultado são conferidas com as saídas esperadas.
- Dentro do intervalo medido: a chamada da estratégia, incluindo a validação das entradas.
  No cenário "completo" da paralela, isso inclui criar os processos, alocar a memória
  compartilhada, copiar X, calcular, coletar Y e encerrar os processos.
- Fora do intervalo: leitura do disco (X já está residente na RAM), carregamento do modelo
  e a conferência com as saídas esperadas. Vale igualmente para todas as estratégias.

Grava <saida>.csv (uma linha por configuração) e <saida>.json (ambiente + as mesmas linhas).
"""

import os

# Uma thread nativa por processo, salvo indicação contrária no ambiente. Isto precisa vir
# antes de importar o NumPy; os processos filhos (spawn) herdam estas variáveis.
VARIAVEIS_DE_THREADS = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for variavel in VARIAVEIS_DE_THREADS:
    os.environ.setdefault(variavel, "1")

import argparse
import csv
import json
import math
import multiprocessing as mp
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

from inferencia import InferenciaParalela, prever_fundido, prever_paralelo, prever_vetorizado
from mlp import ATOL, BASE, RTOL, carregar_modelo, prever_referencia


CONJUNTOS = ("pequeno", "medio", "grande", "enorme")
ESTRATEGIAS = ("escalar", "vetorizado", "fundido", "paralelo")
LIMITE_ESCALAR = 100_000  # a referência leva ~0,3 ms por linha; acima disso fica proibitiva
LOTES_EXTRAS = (4096, 512)  # extensão: lotes menores que um por processo
PROCESSOS = (2, 4)  # 2 é a entrega mínima; 4 é extensão


def medir(funcao, repeticoes):
    """1 aquecimento + `repeticoes` execuções cronometradas. Devolve os tempos e o resultado."""
    funcao()
    tempos = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        resultado = funcao()
        tempos.append(time.perf_counter() - inicio)
    return tempos, resultado


def conferir(resultado, esperado):
    """Correção antes de desempenho: forma, tipo, finitos e tolerância em todas as linhas."""
    assert resultado.shape == esperado.shape, resultado.shape
    assert resultado.dtype == np.float64, resultado.dtype
    assert np.isfinite(resultado).all()
    np.testing.assert_allclose(resultado, esperado, rtol=RTOL, atol=ATOL)
    return float(np.max(np.abs(resultado - esperado)))


def medir_conjunto(nome, modelo, estrategias, repeticoes):
    """Mede todas as configurações pedidas em um conjunto e devolve uma linha por configuração."""
    pasta = BASE / "dados" / nome
    X = np.load(pasta / "entradas.npy", allow_pickle=False)  # residente na RAM, sem mmap
    esperado = np.load(pasta / "saidas_esperadas.npy", allow_pickle=False)
    n = len(X)
    linhas = []

    def registrar(estrategia, cenario, processos, lote, funcao):
        tempos, resultado = medir(funcao, repeticoes)
        mediana = statistics.median(tempos)
        linhas.append({
            "conjunto": nome,
            "linhas": n,
            "estrategia": estrategia,
            "cenario": cenario,
            "processos": processos,
            "threads_solicitadas": os.environ["OPENBLAS_NUM_THREADS"],
            "lote": lote,
            "tarefas": math.ceil(n / lote),
            "repeticoes": repeticoes,
            "tempos_s": ";".join(f"{t:.6e}" for t in tempos),
            "mediana_s": mediana,
            "min_s": min(tempos),
            "max_s": max(tempos),
            "throughput_obs_s": n / mediana,
            "max_dif_abs": conferir(resultado, esperado),
        })
        print(f"{nome:8s} {estrategia:10s} {cenario:22s} p={processos} lote={lote:<7d} "
              f"mediana={1e3 * mediana:10.3f} ms", flush=True)

    if "escalar" in estrategias and n <= LIMITE_ESCALAR:
        registrar("escalar", "completo", 1, n, lambda: prever_referencia(modelo, X))
    if "vetorizado" in estrategias:
        for lote in (n, *LOTES_EXTRAS):
            registrar("vetorizado", "completo", 1, lote, lambda: prever_vetorizado(modelo, X, lote))
    if "fundido" in estrategias:
        w, b = modelo.W_efetivo, modelo.b_efetivo
        registrar("fundido", "completo", 1, n, lambda: prever_fundido(modelo, X, w, b))
    if "paralelo" in estrategias:
        for p in PROCESSOS:
            um_por_processo = math.ceil(n / p)
            registrar("paralelo", "completo", p, um_por_processo,
                      lambda: prever_paralelo(modelo, X, p, um_por_processo))
            # Extensão: pool criado uma vez e reutilizado (processos já prontos).
            with InferenciaParalela(modelo, n, p) as paralela:
                for lote in (um_por_processo, *LOTES_EXTRAS):
                    registrar("paralelo", "pool pronto", p, lote, lambda: paralela.prever(X, lote))
                # X já está na memória compartilhada: mede o cálculo sem validar nem copiar.
                registrar("paralelo", "pool pronto, sem cópia", p, um_por_processo,
                          lambda: paralela.calcular(n, um_por_processo))

    adicionar_speedups(linhas)
    return linhas


def adicionar_speedups(linhas):
    """Speedup = tempo_base / tempo_estratégia, contra duas bases do mesmo conjunto."""
    def speedup(base, linha):
        for candidata in linhas:  # a base processa o conjunto inteiro em um lote só
            if candidata["estrategia"] == base and candidata["lote"] == candidata["linhas"]:
                return candidata["mediana_s"] / linha["mediana_s"]
        return None  # base não medida neste conjunto (ex.: escalar no conjunto enorme)

    for linha in linhas:
        linha["speedup_vs_vetorizado"] = speedup("vetorizado", linha)
        linha["speedup_vs_escalar"] = speedup("escalar", linha)


def threads_do_processo():
    """Threads do sistema operacional no processo atual, depois de um produto de matrizes.

    O OpenBLAS cria as suas threads ao ser carregado; com 1 thread pedida, não cria nenhuma.
    """
    np.ones((256, 256)) @ np.ones((256, 256))
    return psutil.Process().num_threads()


def nome_da_cpu():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for linha in cpuinfo.read_text().splitlines():
            if linha.startswith("model name"):
                return linha.split(":", 1)[1].strip()
    return platform.processor()


def descrever_ambiente(modelo, repeticoes):
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    with mp.get_context("spawn").Pool(1) as pool:
        threads_filho = pool.apply(threads_do_processo)
    return {
        "sistema": platform.platform(),
        "cpu": nome_da_cpu(),
        "cpus_logicas": os.cpu_count(),
        "memoria_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": {chave: blas.get(chave) for chave in ("name", "version", "openblas configuration")},
        "threads_solicitadas": {variavel: os.environ[variavel] for variavel in VARIAVEIS_DE_THREADS},
        "threads_do_so_observadas": {"pai": threads_do_processo(), "filho_spawn": threads_filho},
        "metodo_de_inicio": "spawn",
        "relogio": "time.perf_counter",
        "aquecimentos": 1,
        "repeticoes": repeticoes,
        "dados": "X residente na RAM (np.load sem mmap); leitura do disco fora do intervalo",
        "preparo_fundido_s": statistics.median(
            medir(lambda: (modelo.W_efetivo, modelo.b_efetivo), repeticoes)[0]
        ),
        "rtol": RTOL,
        "atol": ATOL,
    }


def salvar(linhas, ambiente, saida):
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida.with_suffix(".csv"), "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    conteudo = {"ambiente": ambiente, "resultados": linhas}
    saida.with_suffix(".json").write_text(
        json.dumps(conteudo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Resultados em {saida.with_suffix('.csv')} e {saida.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conjuntos", nargs="+", default=CONJUNTOS,
                        help="Pastas em 03-hpc/dados/ (padrão: %(default)s).")
    parser.add_argument("--estrategias", nargs="+", choices=ESTRATEGIAS, default=ESTRATEGIAS)
    parser.add_argument("--repeticoes", type=int, default=10,
                        help="Execuções medidas por configuração (padrão: 10; o mínimo pedido é 5).")
    parser.add_argument("--saida", type=Path, default=BASE / "resultados" / "benchmark",
                        help="Prefixo dos arquivos .csv e .json de saída.")
    args = parser.parse_args()

    modelo = carregar_modelo()
    ambiente = descrever_ambiente(modelo, args.repeticoes)
    print(json.dumps(ambiente, indent=2, ensure_ascii=False))
    linhas = []
    for nome in args.conjuntos:
        linhas += medir_conjunto(nome, modelo, args.estrategias, args.repeticoes)
    salvar(linhas, ambiente, args.saida)


if __name__ == "__main__":
    main()
