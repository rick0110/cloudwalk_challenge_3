"""Gera as tabelas e o gráfico do relatório a partir dos resultados do benchmark.

Uso (a partir da raiz do projeto, depois de benchmark.py e analise_escalar.py):
    uv run --frozen python 03-hpc/resumo.py

Lê 03-hpc/resultados/benchmark.csv, benchmark.json, largo.csv (rede 64 → 1024 → 1) e
custo_escalar.json.
Escreve 03-hpc/resultados/resumo.md e os gráficos speedup_por_linhas.png e
speedup_por_camada_oculta.png.
Nenhum número do relatório é digitado à mão: todos saem daqui.
"""

import csv
import json
import math
import statistics

import matplotlib

matplotlib.use("Agg")  # só grava o arquivo, sem abrir janela
import matplotlib.pyplot as plt

from mlp import BASE


PASTA = BASE / "resultados"
CONJUNTOS = ("pequeno", "medio", "grande", "enorme")

# Configurações comparadas: rótulo -> filtros de `buscar`.
VETORIZADA = dict(estrategia="vetorizado")
CONFIGURACOES = {
    "paralela completa, 2 processos": dict(estrategia="paralelo", processos=2),
    "paralela completa, 4 processos": dict(estrategia="paralelo", processos=4),
    "pool pronto, 2 processos": dict(estrategia="paralelo", cenario="pool pronto", processos=2),
    "pool pronto, 4 processos": dict(estrategia="paralelo", cenario="pool pronto", processos=4),
    "pool pronto sem cópia, 2 processos":
        dict(estrategia="paralelo", cenario="pool pronto, sem cópia", processos=2),
    "pool pronto sem cópia, 4 processos":
        dict(estrategia="paralelo", cenario="pool pronto, sem cópia", processos=4),
    "pool pronto por pickle, 2 processos":
        dict(estrategia="paralelo", cenario="pool pronto, por pickle", processos=2),
    "pool pronto por pickle, 4 processos":
        dict(estrategia="paralelo", cenario="pool pronto, por pickle", processos=4),
    "vetorizada, 2 threads do BLAS": dict(estrategia="vetorizado", cenario="threads do BLAS", threads=2),
    "vetorizada, 4 threads do BLAS": dict(estrategia="vetorizado", cenario="threads do BLAS", threads=4),
    "vetorizada, lote 4.096": dict(estrategia="vetorizado", lote=4096),
    "vetorizada, lote 512": dict(estrategia="vetorizado", lote=512),
    "fundida y = Xw + b": dict(estrategia="fundido"),
}


# --------------------------------------------------------------------------- leitura


def ler_resultados(nome="benchmark"):
    with open(PASTA / f"{nome}.csv", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    for linha in linhas:
        for chave in ("linhas", "processos", "threads_solicitadas", "lote", "tarefas"):
            linha[chave] = int(linha[chave])
        for chave in ("mediana_s", "min_s", "max_s", "throughput_obs_s", "speedup_vs_vetorizado"):
            linha[chave] = float(linha[chave])
        linha["speedup_vs_escalar"] = float(linha["speedup_vs_escalar"] or "nan")
        linha["tempos"] = [float(t) for t in linha["tempos_s"].split(";")]
        linha["rss_acrescimo_mb"] = (float(linha["rss_pico_mb"]) - float(linha["rss_base_mb"])
                                     if linha["rss_pico_mb"] else None)
    return linhas


def buscar(r, conjunto, estrategia, cenario="completo", processos=1, threads=1, lote=None):
    """Uma configuração. Sem `lote`, a padrão: uma tarefa por processo, ceil(n / processos)."""
    for linha in r:
        if (linha["conjunto"], linha["estrategia"], linha["cenario"], linha["processos"],
                linha["threads_solicitadas"]) == (conjunto, estrategia, cenario, processos, threads) \
                and linha["lote"] == (lote or math.ceil(linha["linhas"] / processos)):
            return linha
    return None


def diferenca_pareada(a, b):
    """Mediana, entre as rodadas, de tempo_a − tempo_b na mesma rodada."""
    return statistics.median(ta - tb for ta, tb in zip(a["tempos"], b["tempos"]))


# --------------------------------------------------------------------------- formatação


def ms(segundos):
    """Milissegundos com 3 algarismos significativos e vírgula decimal."""
    valor = 1e3 * segundos
    if valor >= 1000:
        return milhar(valor)
    casas = max(0, 2 - math.floor(math.log10(abs(valor)))) if valor else 0
    return f"{valor:.{casas}f}".replace(".", ",")


def faixa(linha):
    return f"{ms(linha['mediana_s'])} [{ms(linha['min_s'])} – {ms(linha['max_s'])}]"


def milhar(valor):
    return f"{valor:,.0f}".replace(",", ".")


def vezes(valor):
    """Speedup legível: abaixo de 0,1, escreve 1/k (k vezes mais lenta que a base)."""
    if valor < 0.1:
        return "1/" + milhar(1 / valor)
    return f"{valor:.3g}×".replace(".", ",")


def tabela(cabecalho, linhas):
    texto = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    texto += ["| " + " | ".join(map(str, linha)) + " |" for linha in linhas]
    return "\n".join(texto)


# --------------------------------------------------------------------------- tabelas


def tabela_minima(r, conjuntos):
    linhas = []
    for c in conjuntos:
        for rotulo, config in (
            ("escalar (referência)", dict(estrategia="escalar")),
            ("vetorizada, 1 processo", VETORIZADA),
            ("paralela completa, 2 processos", CONFIGURACOES["paralela completa, 2 processos"]),
        ):
            linha = buscar(r, c, **config)
            if linha:
                linhas.append([f"{c} ({milhar(linha['linhas'])})", rotulo,
                               f"{milhar(linha['lote'])} ({linha['tarefas']})", faixa(linha),
                               milhar(linha["throughput_obs_s"]),
                               vezes(linha["speedup_vs_vetorizado"])])
    return tabela(["Conjunto (linhas)", "Estratégia", "Lote (tarefas)", "Mediana [mín – máx] ms",
                   "Throughput (obs/s)", "Speedup vs vetorizada"], linhas)


def tabela_speedups(r, conjuntos):
    linhas = []
    for rotulo, config in CONFIGURACOES.items():
        medidas = [buscar(r, c, **config) for c in conjuntos]
        linhas.append([rotulo] + [vezes(m["speedup_vs_vetorizado"]) for m in medidas])
    escalar = [buscar(r, c, estrategia="escalar") for c in conjuntos]
    linhas.append(["escalar (referência)"] + [vezes(m["speedup_vs_vetorizado"]) if m else "—"
                                              for m in escalar])
    return tabela(["Speedup vs vetorizada (mediana pareada)"] + list(conjuntos), linhas)


def tabela_decomposicao(r, conjuntos):
    """Diferenças pareadas entre cenários medidos: estimativas agregadas, não parcelas isoladas."""
    linhas = []
    for c in conjuntos:
        vetorizada = buscar(r, c, **VETORIZADA)
        for p in (2, 4):
            sem_copia, pronto, completo = (
                buscar(r, c, "paralelo", cenario, p)
                for cenario in ("pool pronto, sem cópia", "pool pronto", "completo")
            )
            linhas.append([c, p, ms(vetorizada["mediana_s"]), ms(sem_copia["mediana_s"]),
                           ms(pronto["mediana_s"]), ms(completo["mediana_s"]),
                           ms(diferenca_pareada(pronto, sem_copia)),
                           ms(diferenca_pareada(completo, pronto))])
    return tabela(["Conjunto", "Processos", "Vetorizada", "Pool pronto, sem cópia", "Pool pronto",
                   "Completa", "Validação + cópia (pronto − sem cópia)",
                   "Criar e encerrar processos + alocar memória compartilhada (completa − pronto)"], linhas)


def tabela_custo_marginal(r):
    """Quanto o tempo sobe por linha adicional, entre o conjunto grande e o enorme.

    Se uma estratégia gasta mais por linha adicional do que a vetorizada, nenhum n faz ela
    alcançar a vetorizada: o speedup tende a a_vetorizada / a_estratégia (o "teto").
    """
    def custo_por_linha(config):
        grande, enorme = buscar(r, "grande", **config), buscar(r, "enorme", **config)
        return (enorme["mediana_s"] - grande["mediana_s"]) / (enorme["linhas"] - grande["linhas"])

    base = custo_por_linha(VETORIZADA)
    linhas = [["vetorizada, 1 processo (base)", f"{1e9 * base:.0f}", "1×"]]
    for rotulo, config in CONFIGURACOES.items():
        a = custo_por_linha(config)
        linhas.append([rotulo, f"{1e9 * a:.0f}", vezes(base / a)])
    return tabela(["Estratégia", "Custo por linha adicional (ns), grande → enorme",
                   "Teto do speedup (n → ∞)"], linhas)


def tabela_lotes(r, conjuntos):
    linhas = []
    for c in conjuntos:
        for p in (2, 4):
            for linha in r:
                if (linha["conjunto"], linha["cenario"], linha["processos"]) == (c, "pool pronto", p):
                    linhas.append([c, p, milhar(linha["lote"]), linha["tarefas"], faixa(linha),
                                   vezes(linha["speedup_vs_vetorizado"])])
        for lote in (4096, 512):
            linha = buscar(r, c, "vetorizado", lote=lote)
            linhas.append([c, 1, milhar(lote), linha["tarefas"], faixa(linha),
                           vezes(linha["speedup_vs_vetorizado"])])
    return tabela(["Conjunto", "Processos", "Lote", "Tarefas", "Mediana [mín – máx] ms",
                   "Speedup vs vetorizada (lote único)"], linhas)


def amortizacao_da_fusao(r, conjuntos, preparo_s):
    linhas = []
    for c in conjuntos:
        fundida = buscar(r, c, "fundido")
        economia = diferenca_pareada(buscar(r, c, **VETORIZADA), fundida)
        linhas.append([c, f"{1e6 * preparo_s:.1f} µs".replace(".", ","), ms(economia),
                       math.ceil(preparo_s / economia), vezes(fundida["speedup_vs_vetorizado"])])
    return tabela(["Conjunto", "Preparo de w e b (uma vez)", "Economia por chamada (ms)",
                   "Chamadas para amortizar", "Speedup vs vetorizada"], linhas)


def tabela_escalar(r):
    custo = json.loads((PASTA / "custo_escalar.json").read_text(encoding="utf-8"))
    nanos = custo["nanossegundos"]
    linha_inteira = nanos["linha inteira do laço interno"]
    linhas = [[nome.strip(), f"`{codigo}`", f"{nanos[nome]:.0f}", vezes_na_linha,
               f"{100 * vezes_na_linha * nanos[nome] / linha_inteira:.0f}%"]
              for nome, (codigo, vezes_na_linha) in custo["codigo"].items()]
    medida = buscar(r, "grande", "escalar")
    texto = tabela(["Parte", "Código", "ns", "Vezes na linha", "% da linha"], linhas)
    texto += (f"\n\nEstimativa: 2.080 execuções por observação × {linha_inteira:.0f} ns = "
              f"{2080 * linha_inteira / 1e3:.0f} µs por observação. "
              f"Medido no conjunto grande: {1e6 * medida['mediana_s'] / medida['linhas']:.0f} µs.")
    return texto


# --------------------------------------------------------------------------- gráfico


def grafico(paineis, caminho):
    """Speedup pareado contra a vetorizada, por tamanho, um painel por rede.

    `paineis` é uma lista de (título, resultados, conjuntos). Acima da linha de 1× compensa.
    """
    series = ("paralela completa, 2 processos", "pool pronto, 2 processos",
              "pool pronto sem cópia, 2 processos", "vetorizada, 4 threads do BLAS")
    cores = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # azul, laranja, verde-água, amarelo
    tracos = ("-", "-", ":", "-")  # pontilhado: na rede larga, coincide com o pool pronto
    figura, eixos = plt.subplots(1, len(paineis), figsize=(4.2 * len(paineis), 3.6), dpi=200,
                                 sharey=True, squeeze=False)
    figura.set_facecolor("#fcfcfb")
    for eixo, (titulo, r, conjuntos) in zip(eixos[0], paineis):
        eixo.set_facecolor("#fcfcfb")
        eixo.axhline(1, color="#52514e", linewidth=1, linestyle="--")
        for rotulo, cor, traco in zip(series, cores, tracos):
            medidas = [buscar(r, c, **CONFIGURACOES[rotulo]) for c in conjuntos]
            eixo.plot([m["linhas"] for m in medidas], [m["speedup_vs_vetorizado"] for m in medidas],
                      color=cor, linewidth=2, linestyle=traco, marker="o", markersize=5,
                      label=rotulo)
        eixo.set_xscale("log")
        eixo.set_yscale("log")
        eixo.set_xlim(6e2, 1.6e6)
        eixo.set_xticks([1e3, 1e4, 1e5, 1e6], ["1 mil", "10 mil", "100 mil", "1 milhão"])
        eixo.set_yticks([1e-3, 1e-2, 1e-1, 1, 2, 4], ["1/1000", "1/100", "1/10", "1×", "2×", "4×"])
        eixo.minorticks_off()
        eixo.set_title(titulo, fontsize=9, color="#0b0b0b", loc="left")
        eixo.set_xlabel("linhas por chamada", color="#52514e", fontsize=8)
        eixo.grid(True, color="#e4e3df", linewidth=0.8)
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eixo.spines[lado].set_color("#c3c2b7")
        eixo.tick_params(colors="#52514e", labelsize=7)
    eixos[0][0].set_ylabel("speedup vs vetorizada em 1 processo\n(acima da linha tracejada, compensa)",
                           color="#52514e", fontsize=8)
    figura.legend(*eixos[0][0].get_legend_handles_labels(), loc="lower center", ncol=4,
                  fontsize=7, frameon=False, labelcolor="#0b0b0b")
    figura.tight_layout(rect=(0, 0.08, 1, 1))
    figura.savefig(caminho, facecolor=figura.get_facecolor())


def tabela_rede_larga(r, conjuntos):
    """Mesmas estratégias, rede 64 → 1024 → 1: 32 vezes mais conta por linha, mesmos dados."""
    rotulos = ("paralela completa, 2 processos", "paralela completa, 4 processos",
               "pool pronto, 2 processos", "pool pronto, 4 processos",
               "pool pronto por pickle, 2 processos", "vetorizada, 4 threads do BLAS")
    linhas = [["vetorizada, 1 processo (ms)"] + [ms(buscar(r, c, **VETORIZADA)["mediana_s"])
                                                 for c in conjuntos]]
    for rotulo in rotulos:
        linhas.append([rotulo] + [vezes(buscar(r, c, **CONFIGURACOES[rotulo])["speedup_vs_vetorizado"])
                                  for c in conjuntos])
    return tabela(["Rede 64 → 1024 → 1: speedup vs vetorizada"] + list(conjuntos), linhas)


def tabela_memoria(r, conjuntos):
    """Acréscimo de RSS (pai + filhos) durante uma chamada, sobre a base antes dela."""
    rotulos = ("paralela completa, 2 processos", "paralela completa, 4 processos",
               "pool pronto, 2 processos", "pool pronto, 4 processos",
               "pool pronto por pickle, 2 processos")
    linhas = []
    for rotulo, config in [("vetorizada, 1 processo", VETORIZADA)] + [
            (rotulo, CONFIGURACOES[rotulo]) for rotulo in rotulos]:
        medidas = [buscar(r, c, **config) for c in conjuntos]
        linhas.append([rotulo] + [f"{m['rss_acrescimo_mb']:.0f} ({m['rss_amostras']})"
                                  for m in medidas])
    return tabela(["Acréscimo de RSS em MB (amostras)"] + list(conjuntos), linhas)


def ler_varredura():
    """Resultados da varredura da camada oculta: {tamanho da camada: linhas do CSV}."""
    varredura = {}
    for arquivo in sorted((PASTA / "varredura").glob("oculta_*.csv")):
        varredura[int(arquivo.stem.split("_")[1])] = ler_resultados(f"varredura/{arquivo.stem}")
    return dict(sorted(varredura.items()))


VARREDURA = {  # rótulo: (filtros de `buscar`, cor, traço)
    "paralela completa, 2 processos": (CONFIGURACOES["paralela completa, 2 processos"], "#2a78d6", "-"),
    "paralela completa, 4 processos": (CONFIGURACOES["paralela completa, 4 processos"], "#2a78d6", "--"),
    "pool pronto, 2 processos": (CONFIGURACOES["pool pronto, 2 processos"], "#eb6834", "-"),
    "pool pronto, 4 processos": (CONFIGURACOES["pool pronto, 4 processos"], "#eb6834", "--"),
    "vetorizada, 2 threads do BLAS": (CONFIGURACOES["vetorizada, 2 threads do BLAS"], "#eda100", "-"),
    "vetorizada, 4 threads do BLAS": (CONFIGURACOES["vetorizada, 4 threads do BLAS"], "#eda100", "--"),
}


def tabela_varredura(varredura):
    ocultas = list(varredura)
    linhas = [["vetorizada, 1 processo (ms)"] + [
        ms(buscar(varredura[h], f"oculta_{h}", **VETORIZADA)["mediana_s"]) for h in ocultas]]
    for rotulo, (config, _, _) in VARREDURA.items():
        linhas.append([rotulo] + [vezes(buscar(varredura[h], f"oculta_{h}", **config)["speedup_vs_vetorizado"])
                                  for h in ocultas])
    return tabela(["10 mil linhas; camada oculta →"] + [str(h) for h in ocultas], linhas)


def grafico_varredura(varredura, caminho):
    """Speedup contra a vetorizada, com 10 mil linhas, em função da camada oculta."""
    ocultas = list(varredura)
    figura, eixo = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    for fundo in (figura, eixo):
        fundo.set_facecolor("#fcfcfb")
    eixo.axhline(1, color="#52514e", linewidth=1, linestyle="--")
    for ideal in (2, 4):
        eixo.axhline(ideal, color="#c3c2b7", linewidth=1, linestyle=":")
        eixo.annotate(f"ideal com {ideal}", (ocultas[0], ideal), xytext=(0, 3),
                      textcoords="offset points", fontsize=7, color="#52514e")
    for rotulo, (config, cor, traco) in VARREDURA.items():
        s = [buscar(varredura[h], f"oculta_{h}", **config)["speedup_vs_vetorizado"] for h in ocultas]
        eixo.plot(ocultas, s, color=cor, linestyle=traco, linewidth=2, marker="o", markersize=5,
                  label=rotulo)
    eixo.set_xscale("log", base=2)
    eixo.set_yscale("log")
    eixo.set_xticks(ocultas, [str(h) for h in ocultas])
    eixo.set_yticks([1 / 256, 1 / 64, 1 / 16, 1 / 4, 1, 2, 4],
                    ["1/256", "1/64", "1/16", "1/4", "1×", "2×", "4×"])
    eixo.minorticks_off()
    eixo.set_xlabel("neurônios na camada oculta (64 → oculta → 1)", color="#52514e", fontsize=8)
    eixo.set_ylabel("speedup vs vetorizada em 1 processo", color="#52514e", fontsize=8)
    eixo.set_title("Com 10 mil linhas: mais conta por linha, mais o paralelismo compensa, até cerca de 2×",
                   fontsize=9, color="#0b0b0b", loc="left")
    eixo.grid(True, color="#e4e3df", linewidth=0.8)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color("#c3c2b7")
    eixo.tick_params(colors="#52514e", labelsize=7)
    eixo.legend(fontsize=7, frameon=False, loc="lower right", ncol=2, labelcolor="#0b0b0b")
    figura.tight_layout()
    figura.savefig(caminho, facecolor=figura.get_facecolor())


# --------------------------------------------------------------------------- principal


def main():
    r = ler_resultados()
    ambiente = json.loads((PASTA / "benchmark.json").read_text(encoding="utf-8"))["ambiente"]
    conjuntos = [c for c in CONJUNTOS if any(linha["conjunto"] == c for linha in r)]
    grandes = [c for c in conjuntos if c in ("grande", "enorme")]
    secoes = [
        ("Ambiente", f"```json\n{json.dumps(ambiente, indent=2, ensure_ascii=False)}\n```"),
        ("Tabela 1. Entrega mínima", tabela_minima(r, conjuntos)),
        ("Tabela 2. Quando cada estratégia compensa", tabela_speedups(r, conjuntos)),
        ("Tabela 3. Para onde vai o tempo da execução paralela", tabela_decomposicao(r, grandes)),
        ("Tabela 4. Custo por linha adicional", tabela_custo_marginal(r)),
        ("Tabela 5. Tamanho do lote", tabela_lotes(r, grandes)),
        ("Tabela 6. Amortização do preparo da fusão",
         amortizacao_da_fusao(r, conjuntos, ambiente["preparo_fundido_s"])),
        ("Tabela 7. Custo da referência escalar", tabela_escalar(r)),
        ("Tabela 8. Memória", tabela_memoria(r, grandes)),
    ]
    if (PASTA / "largo.csv").exists():
        largo = ler_resultados("largo")
        secoes.append(("Tabela 9. Rede mais larga, 64 → 1024 → 1", tabela_rede_larga(
            largo, list(dict.fromkeys(linha["conjunto"] for linha in largo)))))
    varredura = ler_varredura()
    if varredura:
        secoes.append(("Tabela 10. Varredura da camada oculta, 10 mil linhas",
                       tabela_varredura(varredura)))
    texto = "# Resumo dos resultados\n\nGerado por `03-hpc/resumo.py`.\n\n" + "\n\n".join(
        f"## {titulo}\n\n{conteudo}" for titulo, conteudo in secoes) + "\n"
    (PASTA / "resumo.md").write_text(texto, encoding="utf-8")
    paineis = [("Rede fornecida, 64 → 32 → 1", r, conjuntos)]
    if (PASTA / "largo.csv").exists():
        paineis.append(("Rede larga, 64 → 1024 → 1", largo,
                        list(dict.fromkeys(linha["conjunto"] for linha in largo))))
    grafico(paineis, PASTA / "speedup_por_linhas.png")
    if varredura:
        grafico_varredura(varredura, PASTA / "speedup_por_camada_oculta.png")
    print(texto)


if __name__ == "__main__":
    main()
