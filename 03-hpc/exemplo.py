"""Carrega o modelo e confere um pequeno lote; não mede desempenho."""

import argparse
import json
from pathlib import Path

import numpy as np

from mlp import (
    ATOL, RTOL, BASE, MODELO_PADRAO, carregar_modelo, identificador_modelo,
    prever_afim, prever_oraculo, prever_referencia,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELO_PADRAO)
    parser.add_argument("--input", type=Path, default=BASE / "dados" / "exemplo", help="Pasta dos arrays NPY.")
    parser.add_argument("--rows", type=int, default=16, help="Linhas verificadas também pela referência escalar.")
    args = parser.parse_args()
    try:
        if args.rows <= 0:
            raise ValueError("rows deve ser positivo.")
        modelo = carregar_modelo(args.model)
        metadados = json.loads((args.input / "metadados.json").read_text(encoding="utf-8"))
        if metadados["model_parameters_sha256"] != identificador_modelo(modelo):
            raise ValueError("Os dados foram gerados com parâmetros de outro modelo.")
        entradas = np.load(args.input / "entradas.npy", mmap_mode="r", allow_pickle=False)
        esperado = np.load(args.input / "saidas_esperadas.npy", mmap_mode="r", allow_pickle=False)
        if entradas.ndim != 2 or esperado.shape != (len(entradas), modelo.outputs):
            raise ValueError("Formas incompatíveis entre entradas e saídas esperadas.")
        quantidade = min(args.rows, len(entradas))
        if quantidade == 0:
            raise ValueError("Os dados devem conter ao menos uma linha.")
        lote = entradas[:quantidade]
        referencia = prever_referencia(modelo, lote)
        oraculo = prever_oraculo(modelo, lote)
        afim = prever_afim(modelo, lote)
        for resultado in (referencia, oraculo, afim):
            np.testing.assert_allclose(resultado, esperado[:quantidade], rtol=RTOL, atol=ATOL)
    except (ValueError, OSError, KeyError, AssertionError) as erro:
        parser.error(str(erro))
    print(f"Correto: {quantidade} de {len(entradas)} linhas verificadas em CPU.")
    print(f"Referência, oráculo e y = Xw + b coincidem com as saídas salvas (rtol={RTOL}, atol={ATOL}).")
    print(f"Primeiras saídas: {oraculo[:3].ravel().tolist()}")
    print("Este exemplo verifica correção; o benchmark faz parte do desafio.")


if __name__ == "__main__":
    main()
