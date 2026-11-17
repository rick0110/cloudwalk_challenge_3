"""Testes do material fornecido; a solução do desafio deve ter seus próprios testes."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


BASE = Path(__file__).resolve().parent


class TestMaterialHPC(unittest.TestCase):
    def executar(self, script, *argumentos, esperado=0):
        resultado = subprocess.run(
            [sys.executable, str(BASE / script), *map(str, argumentos)],
            cwd=BASE.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(resultado.returncode, esperado, resultado.stderr)
        return resultado

    def test_cli_gera_modelo_dados_e_executa_exemplo(self):
        # Detecta artefatos faltantes, caminhos dependentes do cwd e CLI desconectada.
        with tempfile.TemporaryDirectory() as temporario:
            pasta = Path(temporario)
            modelo = pasta / "modelo.npz"
            dados = pasta / "dados"
            self.executar("gerar_modelo.py", "--output", modelo)
            self.executar(
                "gerar_entradas.py", "--model", modelo, "--rows", 9,
                "--output", dados, "--batch-size", 4,
            )
            self.executar("exemplo.py", "--model", modelo, "--input", dados)
            entradas = np.load(dados / "entradas.npy", allow_pickle=False)
            saidas = np.load(dados / "saidas_esperadas.npy", allow_pickle=False)
            self.assertEqual(entradas.shape, (9, 64))
            self.assertEqual(saidas.shape, (9, 1))
            with np.load(modelo, allow_pickle=False) as pesos:
                esperado = np.einsum("ij,jk->ik", entradas, pesos["W_efetivo"]) + pesos["b_efetivo"]
            np.testing.assert_allclose(saidas, esperado, rtol=1e-10, atol=1e-10)
            metadata = json.loads((dados / "metadados.json").read_text())
            self.assertEqual(metadata["rows"], 9)
            self.assertEqual(metadata["dtype"], "float64")

    def test_inferencia_tem_resultado_manual_e_preserva_ordem(self):
        # Detecta transposição, bias omitido, ativação indevida e linhas reordenadas.
        from mlp import MLP, prever_afim, prever_oraculo, prever_referencia

        modelo = MLP(
            W1=np.array([[1., 2.], [3., 4.]]),
            b1=np.array([5., 6.]),
            W2=np.array([[7.], [8.]]),
            b2=np.array([9.]),
        )
        entradas = np.array([[1., 0.], [0., 1.], [-1., -2.]])
        esperado = np.array([[115.], [145.], [-37.]])
        for inferir in (prever_referencia, prever_oraculo, prever_afim):
            with self.subTest(inferir=inferir.__name__):
                np.testing.assert_allclose(inferir(modelo, entradas), esperado)

    def test_pesos_reproduziveis_e_seed_tem_efeito(self):
        from gerar_modelo import construir_modelo

        primeiro = construir_modelo(seed=21)
        segundo = construir_modelo(seed=21)
        outro = construir_modelo(seed=22)
        for nome in ("W1", "b1", "W2", "b2"):
            np.testing.assert_array_equal(getattr(primeiro, nome), getattr(segundo, nome))
        self.assertFalse(np.array_equal(primeiro.W1, outro.W1))

    def test_geracao_independe_do_tamanho_do_lote(self):
        # A semente define os dados, mesmo quando a memória disponível muda.
        from gerar_entradas import gerar_entradas
        from gerar_modelo import construir_modelo

        modelo = construir_modelo(seed=4)
        with tempfile.TemporaryDirectory() as temporario:
            base = Path(temporario)
            for nome, lote in (("a", 2), ("b", 7)):
                gerar_entradas(modelo, rows=11, seed=42, output=base / nome, batch_size=lote)
            a = np.load(base / "a" / "entradas.npy", allow_pickle=False)
            b = np.load(base / "b" / "entradas.npy", allow_pickle=False)
            np.testing.assert_array_equal(a, b)
            ya = np.load(base / "a" / "saidas_esperadas.npy", allow_pickle=False)
            yb = np.load(base / "b" / "saidas_esperadas.npy", allow_pickle=False)
            np.testing.assert_allclose(ya, yb, rtol=1e-10, atol=1e-10)

    def test_entradas_com_dimensao_dtype_ou_valor_invalido_sao_rejeitadas(self):
        from gerar_modelo import construir_modelo
        from mlp import prever_oraculo, prever_referencia

        modelo = construir_modelo()
        invalidas = (
            np.ones(64),
            np.ones((2, 63)),
            np.ones((2, 64), dtype=np.float32),
            np.full((1, 64), np.nan),
        )
        for entradas in invalidas:
            for inferir in (prever_oraculo, prever_referencia):
                with self.subTest(shape=entradas.shape, inferir=inferir.__name__):
                    with self.assertRaises(ValueError):
                        inferir(modelo, entradas)

    def test_pesos_incompativeis_sao_rejeitados(self):
        from mlp import MLP

        with self.assertRaises(ValueError):
            MLP(np.ones((3, 2)), np.ones(3), np.ones((2, 1)), np.ones(1))

    def test_modelo_salvo_com_equivalente_afim_corrompido_e_rejeitado(self):
        from gerar_modelo import construir_modelo, salvar_modelo
        from mlp import carregar_modelo

        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "modelo.npz"
            salvar_modelo(construir_modelo(), caminho, seed=2026)
            with np.load(caminho, allow_pickle=False) as arquivo:
                pesos = {nome: arquivo[nome] for nome in arquivo.files}
            pesos["W_efetivo"] = pesos["W_efetivo"] + 1
            np.savez(caminho, **pesos)
            with self.assertRaises(ValueError):
                carregar_modelo(caminho)

    def test_gerador_recusa_argumentos_invalidos_sem_criar_dados(self):
        from gerar_entradas import gerar_entradas
        from gerar_modelo import construir_modelo

        with tempfile.TemporaryDirectory() as temporario:
            destino = Path(temporario) / "nao_deve_existir"
            for rows, batch_size in ((0, 10), (-1, 10), (2, 0)):
                with self.subTest(rows=rows, batch_size=batch_size):
                    with self.assertRaises(ValueError):
                        gerar_entradas(
                            construir_modelo(), rows=rows, batch_size=batch_size,
                            seed=1, output=destino,
                        )
                    self.assertFalse(destino.exists())

    def test_cli_recusa_numero_de_features_diferente_do_modelo(self):
        with tempfile.TemporaryDirectory() as temporario:
            pasta = Path(temporario)
            modelo = pasta / "modelo.npz"
            self.executar("gerar_modelo.py", "--output", modelo)
            self.executar(
                "gerar_entradas.py", "--model", modelo, "--features", 63,
                "--output", pasta / "dados", esperado=2,
            )
            self.assertFalse((pasta / "dados").exists())


if __name__ == "__main__":
    unittest.main()
