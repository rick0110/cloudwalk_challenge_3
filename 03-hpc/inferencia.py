"""Inferência em CPU do MLP linear 64 → 32 → 1.

Todas as estratégias usam o mesmo núcleo de cálculo (`calcular_camadas`) e a mesma
divisão das linhas em lotes (`fatias`).
"""

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
