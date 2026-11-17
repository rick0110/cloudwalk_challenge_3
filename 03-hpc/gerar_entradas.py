"""Gera entradas sintéticas e respostas do oráculo em lotes, usando arquivos NPY."""

import argparse
import json
from pathlib import Path

import numpy as np

from mlp import (
    ATOL, RTOL, BASE, MODELO_PADRAO, MLP, carregar_modelo,
    identificador_modelo, prever_oraculo,
)


def gerar_entradas(
    modelo: MLP, *, rows: int, seed: int, output: str | Path, batch_size: int = 4096
) -> dict:
    if rows <= 0 or batch_size <= 0:
        raise ValueError("rows e batch-size devem ser positivos.")
    if seed < 0:
        raise ValueError("seed deve ser não negativa.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A entrada inteira reside em disco. Só um lote temporário é gerado por vez.
    entradas = np.lib.format.open_memmap(
        output / "entradas.npy", mode="w+", dtype=np.float64, shape=(rows, modelo.features)
    )
    saidas = np.lib.format.open_memmap(
        output / "saidas_esperadas.npy", mode="w+", dtype=np.float64, shape=(rows, modelo.outputs)
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    for inicio in range(0, rows, batch_size):
        fim = min(inicio + batch_size, rows)
        lote = rng.standard_normal((fim - inicio, modelo.features))
        entradas[inicio:fim] = lote
        saidas[inicio:fim] = prever_oraculo(modelo, lote)
    entradas.flush()
    saidas.flush()
    del entradas, saidas
    metadados = {
        "format_version": 1,
        "rows": rows,
        "features": modelo.features,
        "outputs": modelo.outputs,
        "dtype": "float64",
        "seed": seed,
        "rng": "PCG64",
        "numpy_version": np.__version__,
        "distribution": "Features independentes com distribuição normal padrão N(0, 1).",
        "batch_size": batch_size,
        "model_parameters_sha256": identificador_modelo(modelo),
        "inputs_file": "entradas.npy",
        "expected_outputs_file": "saidas_esperadas.npy",
        "array_bytes": rows * (modelo.features + modelo.outputs) * 8,
        "rtol": RTOL,
        "atol": ATOL,
    }
    (output / "metadados.json").write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadados


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELO_PADRAO)
    parser.add_argument("--output", type=Path, default=BASE / "dados" / "exemplo", help="Pasta de destino.")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--features", type=int, help="Valida a quantidade de features; deve coincidir com o modelo.")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    try:
        modelo = carregar_modelo(args.model)
        if args.features is not None and args.features != modelo.features:
            raise ValueError(f"O modelo exige {modelo.features} features; recebido {args.features}.")
        metadados = gerar_entradas(
            modelo, rows=args.rows, seed=args.seed, output=args.output, batch_size=args.batch_size
        )
    except (ValueError, OSError) as erro:
        parser.error(str(erro))
    print(f"Dados salvos em: {args.output}")
    print(f"X: ({metadados['rows']}, {metadados['features']}), y: ({metadados['rows']}, {metadados['outputs']})")
    print(f"Arrays: {metadados['array_bytes']:,} bytes, sem contar cabeçalhos NPY.")


if __name__ == "__main__":
    main()
