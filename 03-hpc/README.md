# Desafio 03 — Inferência e paralelização em CPU

Implemente uma função de inferência em CPU e um script para comparar estratégias de execução. O trabalho envolve álgebra linear, vetorização, paralelismo e medição de desempenho. A entrega deve mostrar quando uma estratégia ajuda, quanto custa e como você verificou que ela preserva os resultados.

**Dedicação sugerida: 4 a 6 horas, após preparar o ambiente.** A instalação inicial das ferramentas e dependências fica fora desse período. Esperamos familiaridade com Python, arrays NumPy e multiplicação de matrizes. Você pode consultar a documentação de processos durante o exercício. Priorize a entrega mínima descrita abaixo; as extensões são opcionais e não são necessárias para uma boa avaliação. Não é necessário criar ou publicar um pacote Python.

Você recebe um modelo pronto, um gerador de entradas sintéticas, uma implementação escalar lenta e um oráculo numérico. **Não é necessário treinar um modelo.** O objetivo é implementar e avaliar as estratégias de execução.

Neste texto, **oráculo** é a implementação usada como referência de correção; **BLAS** é a biblioteca nativa que executa operações de álgebra linear; **pool** é um conjunto de processos que pode ser reutilizado; e **RSS** é a quantidade de memória residente de um processo, isto é, suas páginas presentes na RAM.

## O modelo fornecido

O arquivo [`modelo/modelo.npz`](modelo/modelo.npz) contém os pesos de uma rede totalmente conectada com arquitetura **64 → 32 → 1**, duas camadas afins e ativações identidade:

```text
H = X W1 + b1
y = H W2 + b2
```

Portanto, em aritmética exata:

```text
w = W1 W2
b = b1 W2 + b2
y = X w + b
```

O bias é somado a cada linha. A saída tem forma `(n, 1)`, inclusive quando `n = 1`.

Trata-se de um MLP linear, um caso particular sem ativação não linear. A escolha preserva exatamente a função afim solicitada, `y = Xw + b`. Adicionar ReLU, sigmoid ou outra não linearidade não preserva, em geral, essa identidade. Em ponto flutuante, diferentes ordens de operações podem produzir pequenas diferenças de arredondamento.

Os 2.113 parâmetros das duas camadas foram inicializados de forma determinística com a semente `2026`; eles **não foram treinados**. As entradas são sintéticas e independentes de qualquer cliente. A saída é um número real, pode ser negativa e não representa uma probabilidade, decisão de crédito ou estimativa validada para um problema real.

### Dicionário dos arquivos e dados

Todos os arrays usam `float64`. Cada linha de `X` é uma observação; cada coluna é uma feature numérica sem significado de negócio.

| Arquivo / campo | Forma padrão | Significado |
| --- | --- | --- |
| `modelo/modelo.npz`: `W1` | `(64, 32)` | Pesos da primeira camada |
| `modelo/modelo.npz`: `b1` | `(32,)` | Bias da primeira camada |
| `modelo/modelo.npz`: `W2` | `(32, 1)` | Pesos da segunda camada |
| `modelo/modelo.npz`: `b2` | `(1,)` | Bias da segunda camada |
| `modelo/modelo.npz`: `W_efetivo` | `(64, 1)` | `W1 W2`, correspondente a `w` |
| `modelo/modelo.npz`: `b_efetivo` | `(1,)` | `b1 W2 + b2`, correspondente a `b` |
| `dados/exemplo/entradas.npy` | `(128, 64)` | Features amostradas de uma normal padrão `N(0, 1)` |
| `dados/exemplo/saidas_esperadas.npy` | `(128, 1)` | Respostas das duas camadas calculadas pelo oráculo; não são rótulos de treinamento |
| `modelo/modelo.json` | JSON | Arquitetura, semente, tipo numérico, origem dos pesos e tolerâncias |
| `dados/exemplo/metadados.json` | JSON | Dimensões, distribuição, semente, tamanho do lote e identificação dos parâmetros |

O identificador `model_parameters_sha256` permite detectar o uso de entradas/saídas esperadas com pesos diferentes. Os arquivos NumPy são carregados com `allow_pickle=False`.

## Comece por aqui

Execute os comandos a partir da raiz deste pacote. O ambiente compartilhado usa Python 3.12 e dependências fixadas em `uv.lock`; veja também o [README principal](../README.md).

```bash
uv sync --frozen
uv run --frozen python 03-hpc/exemplo.py
uv run --frozen python -m unittest discover -s 03-hpc -p 'test_*.py' -v
```

O exemplo carrega os arquivos já fornecidos e compara as primeiras 16 linhas usando a referência escalar, o oráculo NumPy, a expressão afim e as saídas salvas. A mensagem informa quantas linhas foram verificadas. **Esse comando é uma verificação pequena de correção, não um benchmark.**

Os caminhos padrão são relativos aos próprios scripts, não ao diretório atual. Caminhos passados por argumento seguem o diretório de onde o comando é executado.

### Recrie os materiais

Os comandos abaixo recriam os arquivos padrão. Ao repetir um destino, seus arquivos de modelo ou dados são substituídos.

```bash
uv run --frozen python 03-hpc/gerar_modelo.py
uv run --frozen python 03-hpc/gerar_entradas.py --rows 128 --features 64 --seed 2027
uv run --frozen python 03-hpc/exemplo.py --rows 128
```

O gerador de modelo também aceita `--features`, `--hidden`, `--outputs`, `--seed` e `--output caminho/modelo.npz`. Na comparação principal, mantenha o modelo fornecido; alterações de arquitetura podem ser experimentos adicionais. No gerador de entradas, `--features` é opcional e serve para conferir a compatibilidade com o modelo carregado. Ele não altera os pesos nem adapta dados de dimensão incorreta.

### Gere os conjuntos do benchmark

Use pelo menos estes três tamanhos e mantenha a mesma semente para todas as estratégias:

```bash
uv run --frozen python 03-hpc/gerar_entradas.py --rows 1000 --seed 2027 --output 03-hpc/dados/pequeno
uv run --frozen python 03-hpc/gerar_entradas.py --rows 10000 --seed 2027 --output 03-hpc/dados/medio
uv run --frozen python 03-hpc/gerar_entradas.py --rows 100000 --seed 2027 --output 03-hpc/dados/grande
```

| Linhas | Tamanho de `X`, sem cabeçalho | Tamanho das saídas |
| ---: | ---: | ---: |
| 1.000 | 0,512 MB | 0,008 MB |
| 10.000 | 5,12 MB | 0,08 MB |
| 100.000 | 51,2 MB | 0,8 MB |

MB corresponde a 1.000.000 de bytes nessa tabela. Reserve também memória para resultados intermediários, cópias e processos adicionais.

O gerador aceita `--model caminho/modelo.npz` e `--batch-size`, cujo padrão é `4096`. Ele gera um lote por vez e escreve arrays mapeados em arquivos `.npy`, evitando alocar toda a entrada em um único array temporário. O sistema operacional controla as páginas residentes do mapeamento; isso não constitui um limite rígido de RSS. Reduza o lote quando necessário. Mudar somente o lote mantém as mesmas entradas para a mesma semente e versão do NumPy; saídas continuam equivalentes dentro da tolerância.

Você pode ler conjuntos grandes com `np.load(caminho, mmap_mode="r", allow_pickle=False)`. Documente se mede dados já residentes na memória ou acessos a arquivos mapeados: a primeira leitura e o cache de páginas podem afetar os tempos.

## Entrega mínima: o que implementar

Crie sua solução em arquivos próprios, preservando o material fornecido. Exponha uma função reutilizável de inferência e um script executável para os experimentos, com parâmetros documentados. Uma função e um script simples são suficientes; não exigimos estrutura de pacote nem uma CLI elaborada.

1. **Referência sequencial:** use `prever_referencia` de [`mlp.py`](mlp.py) como ponto inicial. Ela implementa as duas camadas com loops Python. Meça seu desempenho e identifique os principais custos.
2. **Inferência vetorizada em um processo:** implemente as duas camadas com operações sobre arrays. Preserve `float64`, a forma da saída e a ordem das observações.
3. **Inferência paralela em CPU:** particione as observações e execute a inferência com **2 processos**, usando o método de criação `spawn`, que funciona também no macOS. Compare com a versão vetorizada de um processo. Escolha um tamanho de lote e explique como entradas, pesos e saídas são compartilhados ou transferidos. Informe quantas tarefas são geradas: `ceil(n / lote)`. Se houver só uma tarefa, o cálculo não será dividido entre os processos disponíveis.
4. **Avaliação reproduzível:** implemente o benchmark e gere uma tabela de resultados para os três tamanhos de entrada. Explique os resultados e os limites da comparação.

`prever_oraculo` mostra a expressão NumPy das duas camadas como apoio intencional: você pode reutilizá-la como núcleo de cálculo. Sua contribuição deve organizar a inferência em lotes e processos, fazer medições reproduzíveis e analisar os resultados. A referência escalar, o oráculo e os arquivos esperados ajudam a verificar a correção.

GPU, clusters e serviços externos não são necessários. Uma implementação correta em uma máquina pessoal é suficiente.

## Correção e protocolo de medição

### Correção antes de desempenho

Compare todas as linhas produzidas por cada estratégia com `saidas_esperadas.npy` usando **`rtol=1e-10` e `atol=1e-10`**. Verifique também forma, tipo, valores finitos e ordem das linhas. Inclua testes para uma única observação, lotes finais incompletos e número de features incompatível. A comparação numérica é por elemento:

```text
abs(resultado - esperado) <= atol + rtol * abs(esperado)
```

Valide a referência escalar ao menos no conjunto pequeno. Se o custo de executá-la repetidamente nos conjuntos maiores for proibitivo, declare a limitação, registre a amostra efetivamente medida e não apresente uma extrapolação como medição. As versões vetorizada e paralela devem ser verificadas integralmente nos três conjuntos.

### Tempos e recursos da entrega mínima

Use um relógio monotônico de alta resolução, como `time.perf_counter`. Para cada configuração, faça pelo menos **uma execução de aquecimento e cinco repetições medidas**. Se a máquina não permitir esses valores para a referência escalar, explique a adaptação. Guarde os tempos individuais e informe mediana e dispersão, como mínimo e máximo.

Meça e diferencie:

- **Latência do lote completo**, em segundos ou milissegundos, com o mesmo número de linhas em cada comparação. Isso não é a latência de uma requisição de uma única linha.
- **Throughput**, em observações por segundo, calculado como `n / tempo`.
- **Speedup**, como `tempo_base / tempo_estratégia`, sempre identificando a base. A comparação entre processos deve incluir a versão vetorizada de um processo como base.

Na entrega mínima, meça a **execução paralela completa**: criação e inicialização dos processos, envio/compartilhamento dos dados, cálculo, coleta dos resultados e encerramento. Declare exatamente o que está dentro do intervalo cronometrado. Se excluir leitura do disco, carregamento do modelo ou validação, registre essa decisão e aplique-a igualmente às estratégias comparadas. Para usar `spawn`, defina a função executada pelos filhos no nível do módulo e proteja o início do programa com `if __name__ == "__main__":`.

Registre sistema operacional, CPU, versões de Python/NumPy, processos, tamanho dos lotes e configuração de threads nativas solicitada. Os comandos `np.show_config()` ou `np.show_runtime()` ajudam a identificar a BLAS instalada. Não é necessário medir a quantidade efetiva de threads se o backend não a expuser: nesse caso, registre “não observável”, distinguindo-a do limite solicitado.

Uma biblioteca BLAS pode criar várias threads dentro de cada processo. Para começar, **solicite uma thread nativa por processo**, configurando as variáveis **antes de importar NumPy e criar os processos**. Este comando de apoio funciona em shells POSIX:

```bash
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --frozen python caminho/para/seu_benchmark.py
```

As variáveis relevantes dependem do backend instalado; não assuma que todas controlam todas as bibliotecas. Se explorar outras configurações, registre os limites usados e considere a disputa por CPU. Cópias de arrays, serialização, comunicação, acesso à memória e criação de processos podem superar o custo do cálculo, principalmente em lotes pequenos.

**Obter speedup não é obrigatório.** Um resultado mais lento, medido corretamente e explicado com evidências, é válido. Não omita configurações que tiveram desempenho ruim.

## Extensões opcionais

Faça extensões somente se houver tempo depois da entrega mínima. Elas não são requisitos para aprovação:

- Comparar **4 processos** e outros tamanhos de lote, registrando quantas tarefas cada configuração cria.
- Reutilizar um pool e medir a inferência com os processos já prontos. Nesse caso, apresente **também a execução completa**, para explicitar o custo de criar, inicializar e encerrar processos. A diferença entre os cenários não isola cada parcela desse custo.
- Medir memória com `psutil`, incluindo pai e filhos. Ao somar RSS, informe que páginas compartilhadas podem ser contadas mais de uma vez. Registre baseline, pico observado, intervalo e quantidade de amostras; picos curtos podem escapar e alocadores retêm memória entre chamadas. Compare cenários com o mesmo histórico de aquecimento e não interprete o total observado como memória exclusiva do cálculo.
- Avaliar a fusão das duas camadas em `y = Xw + b`. Compare-a separadamente: ela reduz o trabalho e não mede apenas o efeito da paralelização. Informe o custo de preparar `w` e `b` e estime em quantas inferências ele é amortizado. Essa extensão não substitui a comparação principal das duas camadas.

## Entrega

- Função de inferência e script de benchmark, com comandos reproduzíveis e parâmetros explícitos.
- Testes de correção da sua implementação e instruções para executá-los.
- Resultados em CSV ou JSON com tamanho, estratégia, processos, threads solicitadas, lote, número de tarefas, tempos individuais, mediana, dispersão, throughput, speedup e sua base. Inclua memória e threads efetivas apenas quando observadas, deixando claros seus limites.
- Um relatório de até **duas páginas** explicando o ambiente, o protocolo, os resultados, a estratégia recomendada e as limitações. Inclua pelo menos uma tabela comparativa; gráficos são opcionais. Os dados completos das medições ficam no CSV ou JSON, fora desse limite.

Explique, com base nos experimentos, se o paralelismo compensou, quais custos podem explicar o resultado e como reproduzir sua conclusão. Diferencie o que foi medido do que é uma hipótese. Se fizer extensões, discuta o efeito dos lotes e os limites das medições adicionais.

## Critérios de avaliação

Avaliaremos correção numérica e preservação da ordem, clareza da função, uso adequado de vetorização e processos, qualidade das medições, reprodução dos resultados e capacidade de explicar os custos observados. Uma entrega mínima correta e bem explicada é suficiente; complexidade adicional só ajuda quando atende a uma necessidade demonstrada.
