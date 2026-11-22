# Desafio 03-hpc: inferência e paralelização em CPU

Solução do **desafio 3 (HPC)** do estágio CloudWalk × IMPA Tech: inferência vetorizada e
paralela (2 processos `spawn`) do MLP fornecido, e um benchmark reproduzível que mostra
quando o paralelismo ajuda e quanto custa.

- Enunciado: [03-hpc/README.md](03-hpc/README.md)
- Relatório de duas páginas: [RELATORIO.md](RELATORIO.md)
- Todas as medições, tabelas, gráfico e logs: [03-hpc/resultados/](03-hpc/resultados/)

Seções: 1. Visão geral · 2. Arquitetura · 3. Como rodar · 4. Guia de demonstração ·
5. Resultados · 6. Decisões e por quê · 7. Testes · 8. Logs e evidências · 9. Estrutura

## 1. Visão geral (Overview)

O pacote traz um MLP linear 64 → 32 → 1 já pronto, um gerador de entradas sintéticas, uma
referência escalar lenta e um oráculo. Este projeto:

1. mede e explica o custo da referência escalar;
2. implementa a inferência **vetorizada** em um processo e a **paralela** com processos
   `spawn`, com entradas e saídas em memória compartilhada;
3. confere todas as linhas de todas as estratégias com as saídas esperadas
   (`rtol = atol = 1e-10`);
4. mede latência, throughput e speedup em 1 mil, 10 mil e 100 mil linhas (mais 1 milhão,
   para conferir a tendência), e explica para onde vai o tempo.

**Resultado em uma frase:** neste modelo, paralelizar com processos não compensou. A
paralela completa com 2 processos foi mais lenta que a vetorizada em todos os tamanhos
(de 1/1.046 a 0,46×), porque criar os processos custa ~120 ms e copiar os dados custa mais
que o cálculo que se economiza. Detalhes na seção 5 e no relatório.

## 2. Arquitetura (Architecture)

| Módulo | Papel |
| --- | --- |
| [03-hpc/inferencia.py](03-hpc/inferencia.py) | Funções de inferência: núcleo, vetorizada, paralela e fundida |
| [03-hpc/benchmark.py](03-hpc/benchmark.py) | Mede todas as estratégias; grava `resultados/benchmark.csv` e `.json` |
| [03-hpc/analise_escalar.py](03-hpc/analise_escalar.py) | Decompõe o custo da referência escalar, linha a linha |
| [03-hpc/resumo.py](03-hpc/resumo.py) | Gera as tabelas e o gráfico do relatório a partir dos resultados |
| [03-hpc/test_inferencia.py](03-hpc/test_inferencia.py) | Testes de correção da solução |
| [reproduzir.sh](reproduzir.sh) | Roda tudo, do zero, na ordem certa |

O fluxo dos scripts é linear:
`gerar_entradas.py` (material fornecido) → testes → `analise_escalar.py` → `benchmark.py`
→ `resumo.py` → `RELATORIO.md`.

Em `inferencia.py`, todas as estratégias usam o mesmo núcleo, `calcular_camadas`
(`H = X W1 + b1; y = H W2 + b2`), e a mesma divisão das linhas em lotes, `fatias`. A
paralela é a vetorizada com o laço de lotes distribuído entre processos:

```text
 processo pai                     memória compartilhada (RawArray)      processos filhos (spawn)
 ------------                     --------------------------------      ------------------------
 1. valida X                                                            recebem os pesos (17 KB)
 2. copia X  ------------------>  X (n x 64)  --- lê X[início:fim] -->  uma vez, na criação
 3. envia as tarefas: só o par (início, fim), pela fila do pool ----->  4. aplicam o núcleo
 5. copia Y[:n]  <--------------  Y (n x 1)   <-- grava Y[início:fim] -    ao seu lote
```

Nenhuma linha de X passa por pipe ou pickle, e cada filho grava na faixa de Y que
corresponde às suas linhas, então a ordem sai certa por construção. `InferenciaParalela`
mantém o pool aberto entre chamadas; `prever_paralelo` cria, usa e encerra os processos em
uma só chamada.

## 3. Como rodar (How to Run)

**Requisitos:** [uv](https://docs.astral.sh/uv/), que instala o Python 3.12 e as versões
fixadas em `uv.lock`; `bash` para o `reproduzir.sh` (no Windows, WSL ou Git Bash); cerca de
4 GB de RAM livres para o conjunto de 1 milhão de linhas.

**Instalação:**

```bash
uv sync --frozen
```

**Variáveis de ambiente:** nenhuma obrigatória. O `benchmark.py` fixa
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS` e `VECLIB_MAXIMUM_THREADS` em 1
antes de importar o NumPy, como pede o enunciado. Os processos filhos herdam esse valor.

**Tudo, do zero** (~15 minutos, dos quais ~6 são da referência escalar com 100 mil linhas):

```bash
bash reproduzir.sh
```

O script gera os quatro conjuntos com a semente 2027 e roda, nesta ordem, os testes, a
análise da referência escalar, o benchmark e o `resumo.py`. A saída de cada passo fica em
`03-hpc/resultados/logs/`.

**Passo a passo:**

```bash
uv run --frozen python 03-hpc/gerar_entradas.py --rows 100000 --seed 2027 --output 03-hpc/dados/grande
```

```bash
uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v
```

```bash
uv run --frozen python 03-hpc/benchmark.py --help
```

| Opção do benchmark | Padrão | Uso |
| --- | --- | --- |
| `--conjuntos` | `pequeno medio grande enorme` | Pastas em `03-hpc/dados/` |
| `--estrategias` | `escalar vetorizado fundido paralelo` | Estratégias medidas |
| `--repeticoes` | `10` | Rodadas medidas (o mínimo pedido é 5) |
| `--saida` | `03-hpc/resultados/benchmark` | Prefixo dos arquivos `.csv` e `.json` |

**Usar as funções:**

```python
from inferencia import InferenciaParalela, prever_paralelo, prever_vetorizado
from mlp import carregar_modelo

modelo = carregar_modelo()
y = prever_vetorizado(modelo, X)                  # um processo; X de forma (n, 64), float64
y = prever_paralelo(modelo, X, processos=2)       # cria, usa e encerra 2 processos (spawn)

with InferenciaParalela(modelo, capacidade=len(X), processos=2) as paralela:
    y = paralela.prever(X)                        # pool reutilizável entre chamadas
```

Todas devolvem um array `float64` de forma `(n, 1)`, na ordem das linhas de `X`, e rejeitam
número de features diferente do modelo (`ValueError`). Como o pool usa `spawn`, o programa
que chama `prever_paralelo` precisa estar protegido por `if __name__ == "__main__":`.

## 4. Guia de demonstração (Demo Guide)

**Demonstração rápida (~30 segundos):** gera os conjuntos de 1 mil e 10 mil linhas, roda os
testes e mede todas as estratégias neles.

```bash
uv sync --frozen
```

```bash
uv run --frozen python 03-hpc/gerar_entradas.py --rows 1000 --seed 2027 --output 03-hpc/dados/pequeno
```

```bash
uv run --frozen python 03-hpc/gerar_entradas.py --rows 10000 --seed 2027 --output 03-hpc/dados/medio
```

```bash
uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v
```

```bash
uv run --frozen python 03-hpc/benchmark.py --conjuntos pequeno medio --repeticoes 5 --saida 03-hpc/resultados/demo
```

O que observar:

- os testes terminam com `OK` (17 testes: 9 do material e 8 da solução);
- o benchmark imprime o ambiente e, depois, uma linha por configuração, com mediana e
  speedup contra a vetorizada. Antes de medir, ele confere todas as linhas de cada
  configuração e para com `AssertionError` se alguma estiver errada;
- no conjunto médio, a `paralelo completo p=2` aparece com speedup perto de 0,01 (70 a 80
  vezes mais lenta), e a `vetorizado threads do BLAS t=4` acima de 1.

**Avaliação completa:** `bash reproduzir.sh` regenera `03-hpc/resultados/` (CSV, JSON,
`resumo.md`, gráfico e logs). Os tempos absolutos mudam de máquina para máquina; as
conclusões do relatório devem se manter.

## 5. Resultados

Medido numa VM Linux com 4 vCPUs (Intel Xeon 2,3 GHz), Python 3.12.3, NumPy 2.2.6 e
OpenBLAS 0.3.29, com 1 thread nativa por processo e 10 rodadas pareadas. Mediana em ms;
speedup pareado contra a vetorizada (1/k = k vezes mais lenta).

| Linhas | Escalar | Vetorizada | Paralela completa, 2 processos | Speedup da paralela |
|---:|---:|---:|---:|---:|
| 1.000 | 331 | 0,120 | 123 | 1/1.046 |
| 10.000 | 3.304 | 1,67 | 125 | 1/73 |
| 100.000 | 33.799 | 21,8 | 172 | 0,13× |
| 1.000.000 | não medida | 279 | 605 | 0,46× |

![Speedup contra a vetorizada, por número de linhas](03-hpc/resultados/speedup_por_linhas.png)

- **Referência escalar:** 85% do tempo vai em ler elementos do array um a um; a
  multiplicação-soma é só 7%. A vetorizada é de 1.544 a 2.758× mais rápida.
- **Por que a paralela perde:** criar e encerrar os processos custa ~120 ms fixos (o tempo de
  calcular ~550 mil linhas na vetorizada). Validar e copiar X para a memória compartilhada
  custa mais que o cálculo dividido, e cada linha adicional custa 481 ns na paralela
  completa contra 286 ns na vetorizada. Pela tendência, ela não passa de ~0,6×.
- **Quando processos ajudam:** só com o pool reutilizado e X já na memória compartilhada,
  com 2,1 a 4,0× a partir de 100 mil linhas. Enviar os dados por pickle foi 5 a 7× mais
  lento que usar memória compartilhada.
- **Alternativas em um processo:** as threads do BLAS dão até 1,76× (4 threads), sem cópia
  nem criação de processos. Fundir as camadas (`y = Xw + b`) dá ~2,9×, enquanto o modelo
  for linear.

As tabelas completas estão em [03-hpc/resultados/resumo.md](03-hpc/resultados/resumo.md),
e a análise passo a passo em [RELATORIO.md](RELATORIO.md).

## 6. Decisões e por quê

"Tabela N" se refere a [03-hpc/resultados/resumo.md](03-hpc/resultados/resumo.md).
"Teste exploratório" é uma medição feita durante o desenvolvimento, fora do benchmark
versionado.

1. **Um núcleo de cálculo para todas as estratégias.** `calcular_camadas` é o único lugar
   com a conta; a vetorizada e a paralela só mudam como as linhas são distribuídas. Assim,
   diferenças de tempo vêm da organização (lotes, processos), não de implementações
   diferentes.
2. **Bias somado in-place.** `X @ W1 + b1` cria uma segunda matriz n × 32 (25,6 MB com 100
   mil linhas). Num teste exploratório, o núcleo caiu de 18,3 para 12,2 ms no conjunto
   grande. As operações são as mesmas, então o resultado numérico também.
3. **Validação dentro do intervalo, igual para todas.** Reaproveita `validar_entradas` do
   material (forma, tipo, finitos). Na paralela, ela roda no processo pai e é parte serial
   do tempo: validar e copiar X somam 13,9 ms com 100 mil linhas (Tabela 3).
4. **Memória compartilhada para X e Y, em vez de enviar os dados aos filhos.** Cada tarefa
   vira só o par `(início, fim)`. No mesmo pool, mandar os lotes por pickle foi 5 a 7× mais
   lento (Tabela 2).
5. **`RawArray` em vez de `shared_memory.SharedMemory`.** O bloco vai para os filhos na
   criação do pool (`initializer`) e é liberado sozinho quando deixa de ser usado;
   `SharedMemory` exige `unlink()` manual. O custo é parecido: alocar e copiar 51 MB levou
   34 contra 29 ms num teste exploratório.
6. **Lote padrão ceil(n / processos), com `chunksize=1`.** Toda linha custa o mesmo, então
   partes iguais já ficam balanceadas, e menos tarefas significam menos mensagens. Lotes de
   512 pioraram (cada tarefa a mais custou de 10 a 17 µs). Lotes de 4.096 ajudaram com 1
   milhão de linhas, mas a vetorizada com o mesmo lote ganha o mesmo; é efeito de memória,
   não de paralelismo (Tabela 5).
7. **Uma thread nativa por processo, fixada antes de importar o NumPy.** Assim, 2 processos
   ocupam 2 núcleos sem disputar CPU com threads do BLAS. Conferido com `psutil`: 1 thread
   no pai e 1 no filho.
8. **Dados residentes na RAM.** Com `mmap`, a primeira leitura das páginas cairia dentro do
   intervalo medido de uma estratégia e não de outra.
9. **Rodadas pareadas, ordem sorteada e reaquecimento.**
   - A velocidade da VM variou ~30% entre execuções (núcleo de 100 mil linhas: 12,2 e
     depois 16,2 ms, num teste exploratório). Medir cada estratégia num momento diferente
     misturaria essa variação com o efeito real; em rodadas, todas sofrem a mesma.
   - Na ordem fixa, a primeira configuração de cada rodada saía mais lenta: 0,22 contra
     0,13 ms para o mesmo código. A ordem sorteada espalha esse efeito.
   - Com o sorteio, toda medida vinha logo depois de outra estratégia, com o cache "frio".
     Rodar duas vezes e medir só a segunda devolveu o tempo de chamadas repetidas (0,12 ms).
10. **10 rodadas**, quando o mínimo pedido é 5: com a VM ruidosa, a mediana de 10 é mais
    estável. Mínimo, máximo e todos os tempos individuais ficam no CSV.
11. **Referência escalar medida inteira nos três conjuntos pedidos.** São ~0,34 ms por
    linha, ou 34 s por chamada com 100 mil linhas: lento, mas viável, então não foi preciso
    extrapolar. Ela só fica de fora do conjunto de 1 milhão.
12. **Conjunto extra de 1 milhão de linhas.** A dúvida principal era se a paralela
    alcançaria a vetorizada com n maior. Em vez de extrapolar a partir de 100 mil, medi.
13. **Custo por linha adicional entre 100 mil e 1 milhão de linhas.** Se uma estratégia gasta
    mais por linha adicional que a vetorizada, nenhum n a faz alcançar a vetorizada, e a
    razão entre os custos é o teto do speedup. Paralela completa: 481 contra 286 ns, teto de
    ~0,6× (Tabela 4).
14. **Threads do BLAS com `threadpoolctl`.** A biblioteca já está no `uv.lock` (é dependência
    do scikit-learn). Ela muda o limite só no processo pai e só durante aquela configuração,
    o que permite medir na mesma execução, pareada com as demais.
15. **Nada de C++ ou de bibliotecas extras.** O cálculo já roda em código nativo (OpenBLAS);
    o tempo perdido estava em criar processos e mover dados, não na conta.

## 7. Testes

`test_inferencia.py` roda cada estratégia (vetorizada com e sem lotes, paralela com 2 e 4
processos, paralela por pickle e fundida) contra as saídas esperadas do conjunto de exemplo,
com `rtol = atol = 1e-10`, conferindo forma, tipo `float64`, valores finitos e ordem.

| Teste | O que pegaria |
| --- | --- |
| todas as linhas conferem com as saídas esperadas | erro de conta, transposição, bias esquecido |
| uma única observação devolve forma (1, 1) | saída achatada para `(n,)` ou erro com n = 1 |
| ordem das linhas é preservada | linhas trocadas entre processos (com lote 1, cada linha é uma tarefa) |
| número de features incompatível é rejeitado | aceitar `(n, 63)` em silêncio |
| lote final incompleto | perder ou duplicar as últimas linhas quando n não é múltiplo do lote |
| lotes cobrem todas as linhas em ceil(n / lote) tarefas | buracos ou sobreposição entre lotes |
| pool reutilizado não mistura chamadas | sobra de uma chamada maior vazando para uma menor |
| funciona com outra arquitetura (8 → 5 → 3) | código preso ao formato 64 → 32 → 1 |

O benchmark também confere todas as linhas das 75 configurações, nos quatro conjuntos,
antes de medir: a maior diferença foi 2,7 × 10⁻¹⁵, e 64 configurações deram resultados
idênticos bit a bit (coluna `max_dif_abs` do CSV).

## 8. Logs e evidências

Tudo em [03-hpc/resultados/](03-hpc/resultados/), gerado pela mesma execução:

| Arquivo | Conteúdo |
| --- | --- |
| `benchmark.csv`, `benchmark.json` | Uma linha por configuração: conjunto, estratégia, cenário, processos, threads, lote, tarefas, tempos individuais, mediana, mínimo, máximo, throughput, speedups e diferença máxima; o JSON inclui o ambiente |
| `custo_escalar.json` | Custo de cada parte da linha mais executada da referência escalar |
| `resumo.md` | As sete tabelas usadas no relatório, geradas pelo `resumo.py` |
| `speedup_por_linhas.png` | Speedup contra a vetorizada, por tamanho |
| `logs/benchmark.log`, `logs/analise_escalar.log` | Saída original da execução que gerou os arquivos acima |
| `logs/testes.log` | Saída dos 17 testes |

## 9. Estrutura do repositório

```text
.
├── README.md                  este guia
├── RELATORIO.md               relatório de duas páginas
├── reproduzir.sh              roda tudo, do zero
├── pyproject.toml, uv.lock    ambiente fixado (material fornecido)
├── MANIFESTO.json, docs/      material fornecido
└── 03-hpc/
    ├── README.md              enunciado (material fornecido)
    ├── mlp.py, exemplo.py, gerar_*.py, test_hpc.py, modelo/, dados/exemplo/
    │                          material fornecido
    ├── inferencia.py          solução: funções de inferência
    ├── test_inferencia.py     solução: testes
    ├── analise_escalar.py     solução: custo da referência escalar
    ├── benchmark.py           solução: benchmark
    ├── resumo.py              solução: tabelas e gráfico
    └── resultados/            medições, tabelas, gráfico e logs
```

O material fornecido está como veio, com uma exceção: este `README.md` substituiu o README
de quatro linhas do pacote, porque as orientações gerais do processo pedem um `README.md`
com visão geral, arquitetura, como rodar e guia de demonstração. O original está no
primeiro commit, e as instruções dele continuam valendo (seção 3). Para conferir os demais
arquivos contra o `MANIFESTO.json` (a saída esperada é `['README.md']`):

```bash
uv run --frozen python -c "import hashlib, json, pathlib; m = json.loads(pathlib.Path('MANIFESTO.json').read_text()); print([f for f, h in m.items() if hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() != h])"
```
