"""MLP linear fornecido: referência escalar e oráculos de correção, sem benchmark."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


RTOL = 1e-10
ATOL = 1e-10
BASE = Path(__file__).resolve().parent
MODELO_PADRAO = BASE / "modelo" / "modelo.npz"


@dataclass(frozen=True)
class MLP:
    """Duas camadas afins com ativação identidade e parâmetros float64."""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def __post_init__(self):
        for nome in ("W1", "b1", "W2", "b2"):
            valor = getattr(self, nome)
            if not isinstance(valor, np.ndarray) or valor.dtype != np.float64:
                raise ValueError(f"{nome} deve ser um array float64.")
            if not np.isfinite(valor).all():
                raise ValueError(f"{nome} contém valores não finitos.")
        if self.W1.ndim != 2 or self.W2.ndim != 2:
            raise ValueError("W1 e W2 devem ter duas dimensões.")
        if min(*self.W1.shape, *self.W2.shape) <= 0:
            raise ValueError("As dimensões do modelo devem ser positivas.")
        if self.W1.shape[1] != self.W2.shape[0]:
            raise ValueError("A dimensão oculta de W1 e W2 deve coincidir.")
        if self.b1.shape != (self.hidden,) or self.b2.shape != (self.outputs,):
            raise ValueError("Os biases devem ter as dimensões das respectivas saídas.")

    @property
    def features(self):
        return self.W1.shape[0]

    @property
    def hidden(self):
        return self.W1.shape[1]

    @property
    def outputs(self):
        return self.W2.shape[1]

    @property
    def W_efetivo(self):
        return np.dot(self.W1, self.W2)

    @property
    def b_efetivo(self):
        return np.dot(self.b1, self.W2) + self.b2


def identificador_modelo(modelo: MLP) -> str:
    """Identifica os parâmetros, independentemente do caminho do arquivo NPZ."""
    digest = hashlib.sha256()
    for nome in ("W1", "b1", "W2", "b2"):
        array = getattr(modelo, nome)
        digest.update(nome.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.asarray(array, dtype="<f8", order="C").tobytes())
    return digest.hexdigest()


def carregar_modelo(caminho: str | Path = MODELO_PADRAO) -> MLP:
    """Carrega somente arrays numéricos; o arquivo não contém código nem pickle."""
    campos = {"W1", "b1", "W2", "b2", "W_efetivo", "b_efetivo"}
    with np.load(caminho, allow_pickle=False) as arquivo:
        if set(arquivo.files) != campos:
            raise ValueError(f"O modelo deve conter exatamente os campos {sorted(campos)}.")
        modelo = MLP(*(arquivo[nome] for nome in ("W1", "b1", "W2", "b2")))
        for nome in ("W_efetivo", "b_efetivo"):
            salvo = arquivo[nome]
            calculado = getattr(modelo, nome)
            if (
                salvo.dtype != np.float64
                or salvo.shape != calculado.shape
                or not np.isfinite(salvo).all()
                or not np.allclose(salvo, calculado, rtol=RTOL, atol=ATOL)
            ):
                raise ValueError(f"{nome} não corresponde às duas camadas salvas.")
    return modelo


def validar_entradas(modelo: MLP, entradas: np.ndarray) -> np.ndarray:
    entradas = np.asarray(entradas)
    if entradas.ndim != 2 or entradas.shape[1] != modelo.features:
        raise ValueError(f"As entradas devem ter forma (n, {modelo.features}).")
    if entradas.dtype != np.float64:
        raise ValueError("As entradas devem usar float64.")
    if not np.isfinite(entradas).all():
        raise ValueError("As entradas devem conter somente números finitos.")
    return entradas


def prever_referencia(modelo: MLP, entradas: np.ndarray) -> np.ndarray:
    """Ponto de partida propositalmente lento: loops Python nas duas camadas."""
    entradas = validar_entradas(modelo, entradas)
    saidas = np.empty((len(entradas), modelo.outputs), dtype=np.float64)
    for linha in range(len(entradas)):
        ocultos = [0.0] * modelo.hidden
        for neuronio in range(modelo.hidden):
            acumulado = float(modelo.b1[neuronio])
            for feature in range(modelo.features):
                acumulado += float(entradas[linha, feature]) * float(modelo.W1[feature, neuronio])
            ocultos[neuronio] = acumulado  # Ativação identidade.
        for saida in range(modelo.outputs):
            acumulado = float(modelo.b2[saida])
            for neuronio in range(modelo.hidden):
                acumulado += ocultos[neuronio] * float(modelo.W2[neuronio, saida])
            saidas[linha, saida] = acumulado
    return saidas


def prever_oraculo(modelo: MLP, entradas: np.ndarray) -> np.ndarray:
    """Oráculo NumPy para validar a solução; não implementa um benchmark."""
    entradas = validar_entradas(modelo, entradas)
    ocultos = np.dot(entradas, modelo.W1) + modelo.b1
    return np.dot(ocultos, modelo.W2) + modelo.b2


def prever_afim(modelo: MLP, entradas: np.ndarray) -> np.ndarray:
    """Verifica a identidade matemática y = Xw + b usando parâmetros efetivos."""
    entradas = validar_entradas(modelo, entradas)
    return np.dot(entradas, modelo.W_efetivo) + modelo.b_efetivo
