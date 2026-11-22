# Inferência e paralelização em CPU

Desafio 03 (HPC) · CloudWalk × IMPA Tech · Richard Elias Soares Viana

Inferência vetorizada e paralela de um MLP 64 → 32 → 1 em CPU, com um benchmark
reproduzível que mostra quando o paralelismo compensa, quanto custa e por quê. O relatório
técnico, de duas páginas, está em [RELATORIO.md](RELATORIO.md).

## Visão geral

O projeto mede o custo da implementação escalar de referência, implementa a inferência
vetorizada em um processo e a inferência paralela com processos `spawn`, confere todas as
linhas de todas as estratégias contra as saídas esperadas (`rtol = atol = 1e-10`) e mede
latência, throughput e speedup com 1 mil, 10 mil, 100 mil e 1 milhão de linhas.

Neste modelo, paralelizar com processos não compensou: a execução paralela completa com 2
processos foi mais lenta que a vetorizada em todos os tamanhos, de 1/1.046 a 0,46×. Criar os
processos custa cerca de 120 ms, e validar e copiar os dados custa mais que o cálculo que se
economiza. Para ganhar vazão, as threads do próprio BLAS chegam a 1,76× sem criar processos.

## Arquitetura

| Arquivo | Função |
| --- | --- |
| [03-hpc/inferencia.py](03-hpc/inferencia.py) | Núcleo de cálculo e as inferências vetorizada, paralela e fundida |
| [03-hpc/benchmark.py](03-hpc/benchmark.py) | Mede todas as estratégias e grava os resultados em CSV e JSON |
| [03-hpc/analise_escalar.py](03-hpc/analise_escalar.py) | Decompõe o custo da referência escalar |
| [03-hpc/resumo.py](03-hpc/resumo.py) | Gera as tabelas e o gráfico a partir dos resultados |
| [03-hpc/test_inferencia.py](03-hpc/test_inferencia.py) | Testes de correção |
| [03-hpc/mlp.py](03-hpc/mlp.py) | Modelo, referência escalar e oráculo |
| [03-hpc/gerar_entradas.py](03-hpc/gerar_entradas.py) | Gerador dos conjuntos sintéticos |
| [reproduzir.sh](reproduzir.sh) e [Dockerfile](Dockerfile) | Execução de ponta a ponta |

Todas as estratégias usam o mesmo núcleo, `calcular_camadas`, que faz
`H = X W1 + b1; y = H W2 + b2` com produtos de matrizes, e a mesma divisão das linhas em
lotes, `fatias`. A versão vetorizada aplica o núcleo em um único processo. A paralela
distribui os lotes entre processos:

```text
 processo pai                     memória compartilhada (RawArray)      processos filhos (spawn)
 ------------                     --------------------------------      ------------------------
 1. valida X                                                            recebem os pesos (17 KB)
 2. copia X  ------------------>  X (n x 64)  --- lê X[início:fim] -->  uma vez, na criação
 3. envia as tarefas: só o par (início, fim), pela fila do pool ----->  4. aplicam o núcleo
 5. copia Y[:n]  <--------------  Y (n x 1)   <-- grava Y[início:fim] -    ao seu lote
```

Entradas e saídas ficam em memória compartilhada, então nenhuma linha de X passa por pipe
ou pickle. Cada filho grava na faixa de Y correspondente às suas linhas, o que preserva a
ordem. Os pesos são enviados uma única vez a cada filho, na criação do processo. O lote
padrão é ceil(n / processos), uma tarefa por processo; com `chunksize=1`, são sempre
ceil(n / lote) tarefas. `prever_paralelo` cria, usa e encerra os processos em uma chamada,
e `InferenciaParalela` mantém o pool aberto entre chamadas.

## Como executar

Com Docker:

```bash
docker build -t desafio-03-hpc .
```

```bash
docker run --rm --shm-size=2g -v "$PWD/03-hpc/resultados:/projeto/03-hpc/resultados" desafio-03-hpc
```

Com [uv](https://docs.astral.sh/uv/), que instala o Python 3.12 e as dependências fixadas
em `uv.lock`:

```bash
uv sync --frozen
```

```bash
bash reproduzir.sh
```

A execução completa leva cerca de 15 minutos e precisa de uns 4 GB de RAM livres. Ela gera
os conjuntos de dados (semente 2027), roda os testes, a análise da referência escalar, o
benchmark e o resumo, e grava tudo em `03-hpc/resultados/`. O benchmark fixa uma thread
nativa por processo (`OPENBLAS_NUM_THREADS` e afins) antes de importar o NumPy, então
nenhuma variável de ambiente precisa ser definida.

Os passos também podem ser executados separadamente:

```bash
uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v
```

```bash
uv run --frozen python 03-hpc/benchmark.py --conjuntos grande --repeticoes 5
```

O benchmark aceita `--conjuntos` (pastas em `03-hpc/dados/`), `--estrategias` (`escalar`,
`vetorizado`, `fundido`, `paralelo`), `--repeticoes` (padrão 10) e `--saida` (prefixo dos
arquivos de resultado).

Para usar as funções diretamente:

```python
from inferencia import InferenciaParalela, prever_paralelo, prever_vetorizado
from mlp import carregar_modelo

modelo = carregar_modelo()
y = prever_vetorizado(modelo, X)                  # X de forma (n, 64), float64
y = prever_paralelo(modelo, X, processos=2)       # cria, usa e encerra 2 processos

with InferenciaParalela(modelo, capacidade=len(X), processos=2) as paralela:
    y = paralela.prever(X)                        # pool reutilizado entre chamadas
```

Todas devolvem `float64` com forma `(n, 1)`, na ordem das linhas de `X`, e levantam
`ValueError` se o número de features for diferente do esperado pelo modelo.

## Demonstração

Para uma demonstração de cerca de 30 segundos, com os conjuntos de 1 mil e 10 mil linhas:

```bash
bash reproduzir.sh rapido
```

Ou com Docker:

```bash
docker run --rm --shm-size=2g desafio-03-hpc bash reproduzir.sh rapido
```

Os 17 testes devem terminar com `OK`. Em seguida, o benchmark mostra o ambiente (CPU,
versões, BLAS e threads) e uma linha por configuração, com a mediana do tempo e o speedup
em relação à vetorizada. Antes de medir, cada configuração tem todas as linhas conferidas
contra as saídas esperadas. Com 10 mil linhas, a paralela completa com 2 processos aparece
com speedup próximo de 0,01, e a vetorizada com 4 threads do BLAS, acima de 1.

## Resultados

Medidos em uma VM Linux com 4 vCPUs Intel Xeon 2,3 GHz, Python 3.12.3, NumPy 2.2.6 e
OpenBLAS 0.3.29, com uma thread nativa por processo. Cada configuração teve 1 aquecimento e
10 rodadas; em cada rodada, todas as configurações foram medidas em ordem sorteada, e o
speedup é a mediana das razões `tempo_vetorizada / tempo_estratégia` dentro da mesma rodada.

Tempo mediano em ms, com mínimo e máximo; speedup em relação à vetorizada (1/k significa k
vezes mais lenta):

| Linhas | Escalar | Vetorizada | Paralela completa, 2 processos | Speedup da paralela |
|---:|---:|---:|---:|---:|
| 1.000 | 331 [315 – 388] | 0,120 [0,110 – 0,189] | 123 [117 – 158] | 1/1.046 |
| 10.000 | 3.304 [3.203 – 3.530] | 1,67 [1,30 – 2,13] | 125 [119 – 139] | 1/73 |
| 100.000 | 33.799 [33.119 – 34.624] | 21,8 [19,4 – 24,9] | 172 [163 – 189] | 0,13× |
| 1.000.000 | — | 279 [265 – 314] | 605 [568 – 682] | 0,46× |

Throughput, em observações por segundo:

| Estratégia | 1.000 | 10.000 | 100.000 | 1.000.000 |
|---|---:|---:|---:|---:|
| Escalar | 3.020 | 3.026 | 2.959 | — |
| Vetorizada | 8.315.663 | 5.978.614 | 4.594.657 | 3.580.313 |
| Paralela completa, 2 processos | 8.162 | 79.864 | 582.139 | 1.654.079 |

![Speedup em relação à vetorizada, por número de linhas](03-hpc/resultados/speedup_por_linhas.png)

Speedup de cada estratégia em relação à vetorizada em um processo:

| Estratégia | 1.000 | 10.000 | 100.000 | 1.000.000 |
|---|---:|---:|---:|---:|
| Paralela completa, 2 processos | 1/1.046 | 1/73 | 0,13× | 0,46× |
| Paralela completa, 4 processos | 1/1.071 | 1/78 | 0,13× | 0,50× |
| Pool pronto, 2 processos | 0,19× | 0,71× | 0,82× | 1,24× |
| Pool pronto, 4 processos | 0,14× | 0,79× | 1,12× | 1,57× |
| Pool pronto, X já na memória compartilhada, 2 processos | 0,19× | 1,12× | 2,10× | 2,25× |
| Pool pronto, X já na memória compartilhada, 4 processos | 0,20× | 1,31× | 3,17× | 4,04× |
| Pool pronto, dados enviados por pickle, 2 processos | 1/12 | 0,12× | 0,16× | 0,19× |
| Vetorizada, 2 threads do BLAS | 0,99× | 1,29× | 1,39× | 1,36× |
| Vetorizada, 4 threads do BLAS | 1,09× | 1,47× | 1,70× | 1,76× |
| Vetorizada em lotes de 4.096 linhas | 0,96× | 1,03× | 1,00× | 1,37× |
| Fundida, y = Xw + b | 5,60× | 2,92× | 2,89× | 2,88× |

Na referência escalar, a linha mais executada (2.080 vezes por observação) custa cerca de
151 ns, dos quais 85% vão na leitura de elementos do array, um a um, e só 7% na
multiplicação e soma; a vetorizada é de 1.544 a 2.758 vezes mais rápida. Na paralela, criar
e encerrar os processos custa cerca de 120 ms fixos, o equivalente a calcular 550 mil linhas
na vetorizada. Com 100 mil linhas e 2 processos, o cálculo paralelo leva 9,6 ms, enquanto
validar e copiar X para a memória compartilhada leva 13,9 ms. Entre 100 mil e 1 milhão de
linhas, cada linha adicional custou 481 ns na paralela completa e 286 ns na vetorizada, de
modo que o speedup tende a cerca de 0,6× e não chega a 1. Enviar os dados por pickle, em vez
de usar memória compartilhada, foi de 5 a 7 vezes mais lento.

A estratégia recomendada é a vetorizada em um processo, em lotes de alguns milhares de
linhas para entradas grandes. Para mais vazão, as threads do BLAS. Processos só compensam
com o pool reutilizado e os dados já em memória compartilhada. Enquanto o modelo for linear,
a fusão das camadas dá cerca de 2,9× com uma única thread.

Os 17 testes cobrem a conferência de todas as linhas com as saídas esperadas, uma única
observação, lote final incompleto, preservação da ordem com tarefas de uma linha, número de
features incompatível, reutilização do pool e uma rede com outra arquitetura (8 → 5 → 3).
No benchmark, as 75 configurações foram conferidas antes de medir: a maior diferença foi de
2,7 × 10⁻¹⁵, e 64 delas produziram resultados idênticos bit a bit.

Os dados completos estão em [03-hpc/resultados/](03-hpc/resultados/): tempos individuais
em `benchmark.csv` e `benchmark.json`, tabelas em `resumo.md`, o gráfico e os logs de
execução.
