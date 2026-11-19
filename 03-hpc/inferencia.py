"""Inferência em CPU do MLP linear 64 → 32 → 1.

Todas as estratégias usam o mesmo núcleo de cálculo (`calcular_camadas`) e a mesma
divisão das linhas em lotes (`fatias`). A versão paralela é a vetorizada com o laço
de lotes distribuído entre processos.
"""

import math
import multiprocessing as mp

import numpy as np

from mlp import MLP, validar_entradas


def fatias(n: int, lote: int) -> list[tuple[int, int]]:
    """Divide as linhas [0, n) em ceil(n / lote) intervalos contíguos [início, fim)."""
    return [(inicio, min(inicio + lote, n)) for inicio in range(0, n, lote)]


def calcular_camadas(modelo: MLP, X: np.ndarray) -> np.ndarray:
    """Núcleo: H = X W1 + b1 e y = H W2 + b2, com produtos de matrizes (BLAS).

    Os biases são somados in-place: `X @ W1 + b1` alocaria uma segunda matriz (n, 32).
    """
    H = X @ modelo.W1
    H += modelo.b1
    y = H @ modelo.W2
    y += modelo.b2
    return y


def prever_vetorizado(modelo: MLP, X: np.ndarray, lote: int | None = None) -> np.ndarray:
    """Inferência vetorizada em um processo; sem `lote`, calcula todas as linhas de uma vez."""
    X = validar_entradas(modelo, X)
    Y = np.empty((len(X), modelo.outputs))
    for inicio, fim in fatias(len(X), lote or max(len(X), 1)):
        Y[inicio:fim] = calcular_camadas(modelo, X[inicio:fim])
    return Y


# ---------------------------------------------------------------------------------------
# Paralela: processos criados com spawn, entradas e saídas em memória compartilhada.
#
# - Entradas X e saídas Y ficam em dois blocos de memória compartilhada (RawArray).
#   O pai copia X para lá uma vez; cada filho lê só a sua fatia. Nenhuma linha de X
#   passa por pipe ou pickle.
# - Pesos: enviados uma vez a cada filho, na criação do processo (pickle de ~17 KB).
# - Tarefas: só o par (início, fim) de cada lote. Cada filho escreve em Y[início:fim],
#   então a ordem das linhas é preservada por construção.
# ---------------------------------------------------------------------------------------

_filho = {}  # estado de cada processo filho, preenchido uma única vez por _iniciar_filho


def _iniciar_filho(modelo, x_compartilhado, y_compartilhado, capacidade):
    """Roda uma vez em cada filho: guarda os pesos e enxerga a memória compartilhada."""
    _filho["modelo"] = modelo
    _filho["X"] = np.frombuffer(x_compartilhado).reshape(capacidade, modelo.features)
    _filho["Y"] = np.frombuffer(y_compartilhado).reshape(capacidade, modelo.outputs)


def _calcular_fatia(fatia):
    """Tarefa de um filho: calcula as linhas [início, fim) e grava direto em Y."""
    inicio, fim = fatia
    _filho["Y"][inicio:fim] = calcular_camadas(_filho["modelo"], _filho["X"][inicio:fim])


class InferenciaParalela:
    """Pool de processos (spawn) reutilizável, para até `capacidade` linhas por chamada."""

    def __init__(self, modelo: MLP, capacidade: int, processos: int = 2):
        contexto = mp.get_context("spawn")
        x_compartilhado = contexto.RawArray("d", capacidade * modelo.features)
        y_compartilhado = contexto.RawArray("d", capacidade * modelo.outputs)
        self.modelo = modelo
        self.processos = processos
        self.X = np.frombuffer(x_compartilhado).reshape(capacidade, modelo.features)
        self.Y = np.frombuffer(y_compartilhado).reshape(capacidade, modelo.outputs)
        self.pool = contexto.Pool(
            processos, _iniciar_filho, (modelo, x_compartilhado, y_compartilhado, capacidade)
        )

    def calcular(self, n: int, lote: int | None = None) -> np.ndarray:
        """Infere as n primeiras linhas que já estão na memória compartilhada (self.X).

        Sem `lote`, usa ceil(n / processos): uma tarefa por processo. chunksize=1 faz
        cada lote virar exatamente uma tarefa, logo ceil(n / lote) tarefas.
        """
        lote = lote or max(math.ceil(n / self.processos), 1)
        self.pool.map(_calcular_fatia, fatias(n, lote), chunksize=1)
        return self.Y[:n].copy()

    def prever(self, X: np.ndarray, lote: int | None = None) -> np.ndarray:
        """Valida X, copia para a memória compartilhada (a única cópia) e infere."""
        X = validar_entradas(self.modelo, X)
        self.X[: len(X)] = X
        return self.calcular(len(X), lote)

    def fechar(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *excecao):
        self.fechar()


def prever_paralelo(
    modelo: MLP, X: np.ndarray, processos: int = 2, lote: int | None = None
) -> np.ndarray:
    """Execução completa: cria os processos, compartilha os dados, calcula, coleta e encerra."""
    with InferenciaParalela(modelo, len(X), processos) as paralela:
        return paralela.prever(X, lote)
