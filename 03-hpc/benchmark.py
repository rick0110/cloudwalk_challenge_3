"""Benchmark das estratégias de inferência em CPU: escalar, vetorizada, paralela e fundida.

Uso (a partir da raiz do projeto, depois de gerar os conjuntos; veja reproduzir.sh):
    uv run --frozen python 03-hpc/benchmark.py
    uv run --frozen python 03-hpc/benchmark.py --conjuntos pequeno medio --repeticoes 5
    uv run --frozen python 03-hpc/benchmark.py --modelo 03-hpc/modelo_largo/modelo.npz \
        --conjuntos largo_medio --estrategias vetorizado paralelo --saida 03-hpc/resultados/largo

Protocolo:
- Correção antes de desempenho: cada configuração roda uma vez (aquecimento) e esse
  resultado é conferido linha a linha com as saídas esperadas.
- Desenho pareado: depois do aquecimento, cada rodada mede todas as configurações do
  conjunto, uma após a outra (10 rodadas por padrão). A velocidade da máquina varia ao
  longo dos minutos; intercalar faz essa variação atingir todas as estratégias por igual.
  O speedup é a mediana, entre as rodadas, de tempo_base / tempo_estratégia.
- Ordem sorteada: a cada rodada, a ordem das configurações é sorteada (semente fixa). Dentro
  da rodada, cada configuração roda duas vezes seguidas e só a segunda é medida, para não
  herdar o estado de cache deixado pela anterior (exceto a escalar, em Python puro, que leva
  ~30 s por chamada no conjunto grande).
- Dentro do intervalo medido: a chamada da estratégia, incluindo a validação das entradas.
  Na paralela "completa", isso inclui criar os processos, alocar a memória compartilhada,
  copiar X, calcular, coletar Y e encerrar os processos.
- Fora do intervalo: leitura do disco (X já está residente na RAM), carregamento do modelo,
  conferência dos resultados e ajuste do limite de threads. Igual para todas as estratégias.
- Memória: depois das rodadas, cada configuração roda mais uma vez, fora do tempo medido,
  com a RSS do pai e dos filhos amostrada a cada 1 ms (base, pico e número de amostras).
- Threads: 1 thread nativa por processo (variáveis de ambiente definidas antes de importar o
  NumPy). Só as configurações "threads do BLAS" sobem esse limite, no processo pai, com
  threadpoolctl.

Grava <saida>.csv (uma linha por configuração) e <saida>.json (ambiente + as mesmas linhas).
"""

import os

# Uma thread nativa por processo. Isto precisa vir antes de importar o NumPy; os processos
# filhos (spawn) herdam estas variáveis.
VARIAVEIS_DE_THREADS = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for variavel in VARIAVEIS_DE_THREADS:
    os.environ[variavel] = "1"

import argparse
import csv
import json
import math
import multiprocessing as mp
import platform
import random
import statistics
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from inferencia import InferenciaParalela, prever_fundido, prever_paralelo, prever_vetorizado
from mlp import ATOL, BASE, MODELO_PADRAO, RTOL, carregar_modelo, prever_referencia


CONJUNTOS = ("pequeno", "medio", "grande", "enorme")
ESTRATEGIAS = ("escalar", "vetorizado", "fundido", "paralelo")
LIMITE_ESCALAR = 100_000  # a referência leva ~0,3 ms por linha; acima disso fica proibitiva
LOTES_EXTRAS = (4096, 512)  # extensão: lotes menores que "um por processo"
PROCESSOS = (2, 4)  # 2 é a entrega mínima; 4 é extensão
THREADS_BLAS = (2, 4)  # extensão: paralelismo dentro de um processo, pelas threads do BLAS
SEMENTE_DA_ORDEM = 2027


# --------------------------------------------------------------------------- configurações


def configuracao(estrategia, cenario, funcao, lote, processos=1, threads=1):
    return {"estrategia": estrategia, "cenario": cenario, "funcao": funcao, "lote": lote,
            "processos": processos, "threads": threads}


def montar_configuracoes(modelo, X, estrategias, pools):
    """Todas as configurações medidas em um conjunto. A primeira vetorizada é a base."""
    n = len(X)
    lista = []
    if "escalar" in estrategias and n <= LIMITE_ESCALAR:
        lista.append(configuracao("escalar", "completo", partial(prever_referencia, modelo, X), n))
    if "vetorizado" in estrategias:
        for lote in (n, *LOTES_EXTRAS):
            lista.append(configuracao("vetorizado", "completo",
                                      partial(prever_vetorizado, modelo, X, lote), lote))
        for threads in THREADS_BLAS:
            lista.append(configuracao("vetorizado", "threads do BLAS",
                                      partial(prever_vetorizado, modelo, X), n, threads=threads))
    if "fundido" in estrategias:
        w, b = modelo.W_efetivo, modelo.b_efetivo
        lista.append(configuracao("fundido", "completo", partial(prever_fundido, modelo, X, w, b), n))
    if "paralelo" in estrategias:
        for p, paralela in pools.items():
            um_por_processo = math.ceil(n / p)
            lista.append(configuracao("paralelo", "completo",
                                      partial(prever_paralelo, modelo, X, p, um_por_processo),
                                      um_por_processo, processos=p))
            for lote in (um_por_processo, *LOTES_EXTRAS):
                lista.append(configuracao("paralelo", "pool pronto",
                                          partial(paralela.prever, X, lote), lote, processos=p))
            # X já está na memória compartilhada do pool: sem validar e sem copiar.
            lista.append(configuracao("paralelo", "pool pronto, sem cópia",
                                      partial(paralela.calcular, n, um_por_processo),
                                      um_por_processo, processos=p))
            lista.append(configuracao("paralelo", "pool pronto, por pickle",
                                      partial(paralela.prever_por_pickle, X, um_por_processo),
                                      um_por_processo, processos=p))
    return lista


# --------------------------------------------------------------------------- medição


def conferir(resultado, esperado):
    """Forma, tipo, valores finitos e tolerância em todas as linhas, na mesma ordem."""
    assert resultado.shape == esperado.shape, resultado.shape
    assert resultado.dtype == np.float64, resultado.dtype
    assert np.isfinite(resultado).all()
    np.testing.assert_allclose(resultado, esperado, rtol=RTOL, atol=ATOL)
    return float(np.max(np.abs(resultado - esperado)))


def medir_em_rodadas(configuracoes, esperado, repeticoes):
    """Aquecimento e conferência de cada configuração; depois, rodadas em ordem sorteada."""
    sorteio = random.Random(SEMENTE_DA_ORDEM)
    for config in configuracoes:
        with threadpool_limits(limits=config["threads"], user_api="blas"):
            config["max_dif_abs"] = conferir(config["funcao"](), esperado)
        config["tempos"] = []
    for _ in range(repeticoes):
        for config in sorteio.sample(configuracoes, len(configuracoes)):
            with threadpool_limits(limits=config["threads"], user_api="blas"):
                if config["estrategia"] != "escalar":
                    config["funcao"]()  # reaquece depois da configuração anterior; não medido
                inicio = time.perf_counter()
                config["funcao"]()
                config["tempos"].append(time.perf_counter() - inicio)


def medir_memoria(funcao, intervalo_s=0.001):
    """Executa `funcao` uma vez e amostra a RSS do pai somada à dos filhos a cada 1 ms.

    Páginas compartilhadas (como a memória compartilhada de X) entram uma vez em cada
    processo que as toca, e picos mais curtos que o intervalo podem escapar.
    """
    processo = psutil.Process()

    def rss_total():
        total = processo.memory_info().rss
        for filho in processo.children(recursive=True):
            try:
                total += filho.memory_info().rss
            except psutil.NoSuchProcess:  # filho que terminou entre a listagem e a leitura
                pass
        return total

    base, amostras, parar = rss_total(), [], threading.Event()

    def amostrar():
        while not parar.is_set():
            amostras.append(rss_total())
            time.sleep(intervalo_s)

    amostrador = threading.Thread(target=amostrar)
    amostrador.start()
    funcao()
    parar.set()
    amostrador.join()
    return base / 1e6, max(amostras + [base]) / 1e6, len(amostras)


def razao_pareada(base, config):
    """Mediana, entre as rodadas, de tempo_base / tempo_estratégia na mesma rodada."""
    return statistics.median(b / t for b, t in zip(base["tempos"], config["tempos"]))


def medir_conjunto(nome, modelo, estrategias, repeticoes):
    pasta = BASE / "dados" / nome
    X = np.load(pasta / "entradas.npy", allow_pickle=False)  # residente na RAM, sem mmap
    esperado = np.load(pasta / "saidas_esperadas.npy", allow_pickle=False)
    n = len(X)
    pools = {p: InferenciaParalela(modelo, n, p) for p in PROCESSOS} if "paralelo" in estrategias else {}
    configuracoes = montar_configuracoes(modelo, X, estrategias, pools)
    medir_em_rodadas(configuracoes, esperado, repeticoes)
    for config in configuracoes:  # memória: uma execução extra, fora do tempo medido
        config["memoria"] = (None, None, None)
        if config["estrategia"] != "escalar":
            with threadpool_limits(limits=config["threads"], user_api="blas"):
                config["memoria"] = medir_memoria(config["funcao"])
    for paralela in pools.values():
        paralela.fechar()

    base = next(c for c in configuracoes if c["estrategia"] == "vetorizado")
    escalar = next((c for c in configuracoes if c["estrategia"] == "escalar"), None)
    linhas = []
    for config in configuracoes:
        tempos = config["tempos"]
        mediana = statistics.median(tempos)
        linhas.append({
            "conjunto": nome,
            "linhas": n,
            "estrategia": config["estrategia"],
            "cenario": config["cenario"],
            "processos": config["processos"],
            "threads_solicitadas": config["threads"],
            "lote": config["lote"],
            "tarefas": math.ceil(n / config["lote"]),
            "repeticoes": len(tempos),
            "tempos_s": ";".join(f"{t:.6e}" for t in tempos),
            "mediana_s": mediana,
            "min_s": min(tempos),
            "max_s": max(tempos),
            "throughput_obs_s": n / mediana,
            "speedup_vs_vetorizado": razao_pareada(base, config),
            "speedup_vs_escalar": razao_pareada(escalar, config) if escalar else None,
            "max_dif_abs": config["max_dif_abs"],
            "rss_base_mb": config["memoria"][0],
            "rss_pico_mb": config["memoria"][1],
            "rss_amostras": config["memoria"][2],
        })
        print(f"{nome:8s} {config['estrategia']:10s} {config['cenario']:24s} "
              f"p={config['processos']} t={config['threads']} lote={config['lote']:<7d} "
              f"mediana={1e3 * mediana:10.3f} ms  speedup={linhas[-1]['speedup_vs_vetorizado']:7.3f}",
              flush=True)
    return linhas


# --------------------------------------------------------------------------- ambiente


def threads_do_processo():
    """Threads do sistema operacional no processo atual, depois de um produto de matrizes."""
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
    observadas = {}
    for threads in (1, *THREADS_BLAS):  # em ordem crescente: o OpenBLAS não destrói threads
        with threadpool_limits(limits=threads, user_api="blas"):
            observadas[f"pai, limite de {threads}"] = threads_do_processo()
    with mp.get_context("spawn").Pool(1) as pool:
        observadas["filho spawn"] = pool.apply(threads_do_processo)
    preparo = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        modelo.W_efetivo, modelo.b_efetivo
        preparo.append(time.perf_counter() - inicio)
    return {
        "sistema": platform.platform(),
        "cpu": nome_da_cpu(),
        "cpus_logicas": os.cpu_count(),
        "memoria_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": {chave: blas.get(chave) for chave in ("name", "version", "openblas configuration")},
        "threads_solicitadas": {variavel: os.environ[variavel] for variavel in VARIAVEIS_DE_THREADS},
        "threads_do_so_observadas": observadas,
        "metodo_de_inicio": "spawn",
        "relogio": "time.perf_counter",
        "aquecimentos": 1,
        "rodadas": repeticoes,
        "dados": "X residente na RAM (np.load sem mmap); leitura do disco fora do intervalo",
        "preparo_fundido_s": statistics.median(preparo),
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--conjuntos", nargs="+", default=CONJUNTOS,
                        help="Pastas em 03-hpc/dados/ (padrão: %(default)s).")
    parser.add_argument("--estrategias", nargs="+", choices=ESTRATEGIAS, default=ESTRATEGIAS)
    parser.add_argument("--repeticoes", type=int, default=10,
                        help="Rodadas medidas (padrão: 10; o mínimo pedido é 5).")
    parser.add_argument("--modelo", type=Path, default=MODELO_PADRAO,
                        help="Arquivo .npz do modelo (padrão: o modelo fornecido).")
    parser.add_argument("--saida", type=Path, default=BASE / "resultados" / "benchmark",
                        help="Prefixo dos arquivos .csv e .json de saída.")
    args = parser.parse_args()

    modelo = carregar_modelo(args.modelo)
    ambiente = descrever_ambiente(modelo, args.repeticoes)
    ambiente["modelo"] = {"arquivo": str(args.modelo), "arquitetura": [
        modelo.features, modelo.hidden, modelo.outputs]}
    print(json.dumps(ambiente, indent=2, ensure_ascii=False))
    linhas = []
    for nome in args.conjuntos:
        linhas += medir_conjunto(nome, modelo, args.estrategias, args.repeticoes)
    salvar(linhas, ambiente, args.saida)


if __name__ == "__main__":
    main()
