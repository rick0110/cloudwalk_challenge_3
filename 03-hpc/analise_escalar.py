"""Decompõe o custo da referência escalar (`prever_referencia`) com micro-medições (timeit).

A linha mais executada da referência é a do laço interno da primeira camada:

    acumulado += float(entradas[linha, feature]) * float(modelo.W1[feature, neuronio])

Ela roda 64 x 32 = 2.048 vezes por observação (mais 32 vezes na segunda camada, que é
parecida). Este script mede quanto custa a linha inteira e cada uma de suas partes.

Uso (a partir da raiz do projeto):
    uv run --frozen python 03-hpc/analise_escalar.py
"""

import json
import platform
import timeit

import numpy as np

from mlp import BASE, carregar_modelo


SAIDA = BASE / "resultados" / "custo_escalar.json"
EXECUCOES = 1_000_000  # execuções por medição
REPETICOES = 5  # medições; usamos a menor, a menos afetada por interrupções do sistema

# nome: (código medido, quantas vezes aparece na linha inteira)
PARTES = {
    "linha inteira do laço interno": ("a += float(X[i, j]) * float(m.W1[j, k])", 1),
    "ler um elemento do array e converter": ("float(X[i, j])", 2),
    "  só a leitura (cria um numpy.float64)": ("X[i, j]", 2),
    "buscar o atributo m.W1 do dataclass": ("m.W1", 1),
    "multiplicar e acumular floats": ("a += b * c", 1),
}


def nanossegundos(codigo: str, contexto: dict) -> float:
    tempos = timeit.repeat(
        codigo, setup="a = 0.0", globals=contexto, number=EXECUCOES, repeat=REPETICOES
    )
    return min(tempos) / EXECUCOES * 1e9


def custo_liquido(codigo: str, contexto: dict) -> float:
    """Desconta o custo do próprio laço do timeit, medido com um comando vazio."""
    return nanossegundos(codigo, contexto) - nanossegundos("pass", contexto)


def main():
    contexto = {
        "X": np.load(BASE / "dados" / "exemplo" / "entradas.npy", allow_pickle=False),
        "m": carregar_modelo(),
        "b": 1.5,
        "c": 2.5,
        "i": 3,
        "j": 5,
        "k": 7,
    }
    custos = {nome: custo_liquido(codigo, contexto) for nome, (codigo, _) in PARTES.items()}

    linha = custos["linha inteira do laço interno"]
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print("    ns  vezes  % da linha  parte")
    for nome, ns in custos.items():
        vezes = PARTES[nome][1]
        print(f"{ns:6.1f}  {vezes:5d}  {100 * vezes * ns / linha:9.0f}%  {nome}")
    print(f"Estimativa por observação: 2.080 x {linha:.0f} ns = {2080 * linha / 1e6:.2f} ms")

    SAIDA.parent.mkdir(exist_ok=True)
    resultado = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "metodo": f"timeit: menor de {REPETICOES} medições de {EXECUCOES} execuções, "
        "descontado o laço vazio (pass)",
        "codigo": PARTES,
        "nanossegundos": custos,
    }
    SAIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
