"""Testes de correção da solução (inferencia.py).

Execute a partir da raiz do projeto:
    uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v
"""

import math
import unittest

import numpy as np

from inferencia import InferenciaParalela, fatias, prever_paralelo, prever_vetorizado
from mlp import ATOL, BASE, RTOL, carregar_modelo


MODELO = carregar_modelo()
X = np.load(BASE / "dados" / "exemplo" / "entradas.npy", allow_pickle=False)
ESPERADO = np.load(BASE / "dados" / "exemplo" / "saidas_esperadas.npy", allow_pickle=False)

# Cada estratégia recebe as entradas e devolve as saídas. Com lote 7, as 128 linhas
# viram 18 lotes completos e um último lote incompleto, de 2 linhas.
ESTRATEGIAS = {
    "vetorizado": lambda X: prever_vetorizado(MODELO, X),
    "vetorizado, lote 7": lambda X: prever_vetorizado(MODELO, X, lote=7),
    "paralelo, 2 processos": lambda X: prever_paralelo(MODELO, X, processos=2),
    "paralelo, 2 processos, lote 7": lambda X: prever_paralelo(MODELO, X, processos=2, lote=7),
    # Lote 1: cada linha é uma tarefa, a forma mais fácil de embaralhar a ordem de execução.
    "paralelo, 4 processos, lote 1": lambda X: prever_paralelo(MODELO, X, processos=4, lote=1),
}


class TestInferencia(unittest.TestCase):
    def conferir(self, resultado, esperado):
        """Forma, tipo, valores finitos e tolerância em todas as linhas, na mesma ordem."""
        self.assertEqual(resultado.shape, esperado.shape)
        self.assertEqual(resultado.dtype, np.float64)
        self.assertTrue(np.isfinite(resultado).all())
        np.testing.assert_allclose(resultado, esperado, rtol=RTOL, atol=ATOL)

    def test_todas_as_linhas_conferem_com_as_saidas_esperadas(self):
        for nome, prever in ESTRATEGIAS.items():
            with self.subTest(nome):
                self.conferir(prever(X), ESPERADO)

    def test_uma_unica_observacao_devolve_forma_1_por_1(self):
        for nome, prever in ESTRATEGIAS.items():
            with self.subTest(nome):
                self.conferir(prever(X[:1]), ESPERADO[:1])

    def test_ordem_das_linhas_e_preservada(self):
        ordem = np.random.default_rng(0).permutation(len(X))
        for nome, prever in ESTRATEGIAS.items():
            with self.subTest(nome):
                self.conferir(prever(X[ordem]), ESPERADO[ordem])

    def test_numero_de_features_incompativel_e_rejeitado(self):
        for nome, prever in ESTRATEGIAS.items():
            with self.subTest(nome), self.assertRaises(ValueError):
                prever(X[:, :63])

    def test_pool_reutilizado_nao_mistura_chamadas(self):
        # A segunda chamada tem menos linhas que a capacidade: nada da primeira pode vazar.
        with InferenciaParalela(MODELO, capacidade=len(X), processos=2) as paralela:
            self.conferir(paralela.prever(X), ESPERADO)
            self.conferir(paralela.prever(X[:10][::-1]), ESPERADO[:10][::-1])

    def test_lote_final_incompleto(self):
        self.assertEqual(fatias(128, 7)[-1], (126, 128))
        self.assertEqual(fatias(10, 4), [(0, 4), (4, 8), (8, 10)])

    def test_lotes_cobrem_todas_as_linhas_em_ceil_n_sobre_lote_tarefas(self):
        for n, lote in [(1, 1), (10, 4), (128, 7), (100, 100), (100, 1000)]:
            with self.subTest(n=n, lote=lote):
                intervalos = fatias(n, lote)
                self.assertEqual(len(intervalos), math.ceil(n / lote))
                self.assertEqual(intervalos[0][0], 0)
                self.assertEqual(intervalos[-1][1], n)
                for anterior, seguinte in zip(intervalos, intervalos[1:]):
                    self.assertEqual(anterior[1], seguinte[0])


if __name__ == "__main__":
    unittest.main()
