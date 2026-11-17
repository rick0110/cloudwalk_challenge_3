"""Constrói e salva os pesos do modelo; não realiza treinamento."""

import argparse
import json
from pathlib import Path

import numpy as np

from mlp import ATOL, RTOL, MLP, MODELO_PADRAO, identificador_modelo


def construir_modelo(*, seed=2026, features=64, hidden=32, outputs=1) -> MLP:
    if min(features, hidden, outputs) <= 0:
        raise ValueError("features, hidden e outputs devem ser positivos.")
    if seed < 0:
        raise ValueError("seed deve ser não negativa.")
    rng = np.random.Generator(np.random.PCG64(seed))
    return MLP(
        W1=rng.standard_normal((features, hidden)) / np.sqrt(features),
        b1=rng.standard_normal(hidden) * 0.1,
        W2=rng.standard_normal((hidden, outputs)) / np.sqrt(hidden),
        b2=rng.standard_normal(outputs) * 0.1,
    )


def salvar_modelo(modelo: MLP, output: str | Path, *, seed: int) -> None:
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("O caminho de saída do modelo deve terminar em .npz.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        W1=modelo.W1,
        b1=modelo.b1,
        W2=modelo.W2,
        b2=modelo.b2,
        W_efetivo=modelo.W_efetivo,
        b_efetivo=modelo.b_efetivo,
    )
    metadados = {
        "format_version": 1,
        "architecture": [modelo.features, modelo.hidden, modelo.outputs],
        "activation": "identity",
        "dtype": "float64",
        "seed": seed,
        "rng": "PCG64",
        "numpy_version": np.__version__,
        "trained": False,
        "purpose": "Exercício de inferência e desempenho em CPU; sem validade preditiva de crédito.",
        "formula": "H = X @ W1 + b1; y = H @ W2 + b2 = X @ W_efetivo + b_efetivo",
        "model_parameters_sha256": identificador_modelo(modelo),
        "rtol": RTOL,
        "atol": ATOL,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODELO_PADRAO, help="Arquivo .npz de destino.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--outputs", type=int, default=1)
    args = parser.parse_args()
    try:
        modelo = construir_modelo(
            seed=args.seed, features=args.features, hidden=args.hidden, outputs=args.outputs
        )
        salvar_modelo(modelo, args.output, seed=args.seed)
    except (ValueError, OSError) as erro:
        parser.error(str(erro))
    print(f"Modelo salvo: {args.output}")
    print(f"Arquitetura: {modelo.features} → {modelo.hidden} → {modelo.outputs}; float64; identidade.")
    print("Pesos inicializados deterministicamente; nenhum treinamento realizado.")


if __name__ == "__main__":
    main()
