# Solução do desafio 03-hpc: inferência e paralelização em CPU

Este guia explica o que foi feito, onde está cada parte, como reproduzir tudo e por que
cada decisão foi tomada. O relatório de duas páginas está em [RELATORIO.md](RELATORIO.md);
os dados completos das medições estão em [03-hpc/resultados/](03-hpc/resultados/).

## Estrutura

O material fornecido continua intacto: o primeiro commit importa o pacote como veio
(SHA-256 conferidos com `MANIFESTO.json`) e nenhum commit posterior altera esses arquivos.
A solução está toda em arquivos novos.

| Arquivo | Papel |
| --- | --- |
| [03-hpc/inferencia.py](03-hpc/inferencia.py) | Funções de inferência: núcleo, vetorizada, paralela e fundida |
| [03-hpc/test_inferencia.py](03-hpc/test_inferencia.py) | Testes de correção da solução |
| [03-hpc/benchmark.py](03-hpc/benchmark.py) | Benchmark; grava `resultados/*.csv` e `resultados/*.json` |
| [03-hpc/analise_escalar.py](03-hpc/analise_escalar.py) | Decomposição do custo da referência escalar |
| [03-hpc/resumo.py](03-hpc/resumo.py) | Gera as tabelas e o gráfico do relatório a partir dos resultados |
| [03-hpc/resultados/](03-hpc/resultados/) | Medições completas, resumo em Markdown e gráfico |
| [reproduzir.sh](reproduzir.sh) | Roda tudo, do zero, na ordem certa |
| [RELATORIO.md](RELATORIO.md) | Relatório de até duas páginas |

## Como reproduzir

Pré-requisito: [uv](https://docs.astral.sh/uv/). Na raiz do projeto:

```bash
bash reproduzir.sh
```

O script instala o ambiente fixado (`uv sync --frozen`), gera os quatro conjuntos com a
semente 2027, roda os testes, a análise da referência escalar, o benchmark principal, as
duas execuções com threads do BLAS e, por fim, o `resumo.py`. Leva uns 15 minutos; a maior
parte é a referência escalar no conjunto grande. Cada passo também pode ser rodado sozinho:

```bash
uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v
```

```bash
uv run --frozen python 03-hpc/benchmark.py --conjuntos pequeno medio grande --repeticoes 5
```

```bash
uv run --frozen python 03-hpc/resumo.py
```

## Como usar as funções

```python
from inferencia import InferenciaParalela, prever_paralelo, prever_vetorizado
from mlp import carregar_modelo

modelo = carregar_modelo()
y = prever_vetorizado(modelo, X)                  # um processo; X de forma (n, 64), float64
y = prever_paralelo(modelo, X, processos=2)       # cria, usa e encerra 2 processos (spawn)

with InferenciaParalela(modelo, capacidade=len(X), processos=2) as paralela:
    y = paralela.prever(X)                        # pool reutilizável entre chamadas
```

Todas devolvem um array `float64` de forma `(n, 1)`, na mesma ordem das linhas de `X`, e
rejeitam entradas com número de features diferente de 64 (`ValueError`). Como o pool usa
`spawn`, o programa que chama `prever_paralelo` precisa estar protegido por
`if __name__ == "__main__":`.

## Decisões e por quê

"Tabela N" se refere a [03-hpc/resultados/resumo.md](03-hpc/resultados/resumo.md), gerado
a partir do benchmark. "Teste exploratório" é uma medição feita durante o desenvolvimento,
fora do benchmark versionado.

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

## Testes

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

O benchmark também confere todas as linhas de todas as configurações, nos quatro
conjuntos, antes de medir (coluna `max_dif_abs` do CSV).
