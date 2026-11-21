# Resumo dos resultados

Gerado por `03-hpc/resumo.py`.

## Ambiente

```json
{
  "sistema": "Linux-6.18.44-fc-v553-x86_64-with-glibc2.39",
  "cpu": "Intel(R) Xeon(R) Processor @ 2.30GHz",
  "cpus_logicas": 4,
  "memoria_total_gb": 16.9,
  "python": "3.12.3",
  "numpy": "2.2.6",
  "blas": {
    "name": "scipy-openblas",
    "version": "0.3.29",
    "openblas configuration": "OpenBLAS 0.3.29  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"
  },
  "threads_solicitadas": {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1"
  },
  "threads_do_so_observadas": {
    "pai, limite de 1": 1,
    "pai, limite de 2": 2,
    "pai, limite de 4": 4,
    "filho spawn": 1
  },
  "metodo_de_inicio": "spawn",
  "relogio": "time.perf_counter",
  "aquecimentos": 1,
  "rodadas": 10,
  "dados": "X residente na RAM (np.load sem mmap); leitura do disco fora do intervalo",
  "preparo_fundido_s": 3.720499989867676e-06,
  "rtol": 1e-10,
  "atol": 1e-10
}
```

## Tabela 1. Entrega mínima

| Conjunto (linhas) | Estratégia | Lote (tarefas) | Mediana [mín – máx] ms | Throughput (obs/s) | Speedup vs vetorizada |
|---|---|---|---|---|---|
| pequeno (1.000) | escalar (referência) | 1.000 (1) | 331 [315 – 388] | 3.020 | 1/2.758 |
| pequeno (1.000) | vetorizada, 1 processo | 1.000 (1) | 0,120 [0,110 – 0,189] | 8.315.663 | 1× |
| pequeno (1.000) | paralela completa, 2 processos | 500 (2) | 123 [117 – 158] | 8.162 | 1/1.046 |
| medio (10.000) | escalar (referência) | 10.000 (1) | 3.304 [3.203 – 3.530] | 3.026 | 1/1.948 |
| medio (10.000) | vetorizada, 1 processo | 10.000 (1) | 1,67 [1,30 – 2,13] | 5.978.614 | 1× |
| medio (10.000) | paralela completa, 2 processos | 5.000 (2) | 125 [119 – 139] | 79.864 | 1/73 |
| grande (100.000) | escalar (referência) | 100.000 (1) | 33.799 [33.119 – 34.624] | 2.959 | 1/1.544 |
| grande (100.000) | vetorizada, 1 processo | 100.000 (1) | 21,8 [19,4 – 24,9] | 4.594.657 | 1× |
| grande (100.000) | paralela completa, 2 processos | 50.000 (2) | 172 [163 – 189] | 582.139 | 0,13× |
| enorme (1.000.000) | vetorizada, 1 processo | 1.000.000 (1) | 279 [265 – 314] | 3.580.313 | 1× |
| enorme (1.000.000) | paralela completa, 2 processos | 500.000 (2) | 605 [568 – 682] | 1.654.079 | 0,459× |

## Tabela 2. Quando cada estratégia compensa

| Speedup vs vetorizada (mediana pareada) | pequeno | medio | grande | enorme |
|---|---|---|---|---|
| paralela completa, 2 processos | 1/1.046 | 1/73 | 0,13× | 0,459× |
| paralela completa, 4 processos | 1/1.071 | 1/78 | 0,127× | 0,499× |
| pool pronto, 2 processos | 0,186× | 0,707× | 0,817× | 1,24× |
| pool pronto, 4 processos | 0,136× | 0,785× | 1,12× | 1,57× |
| pool pronto sem cópia, 2 processos | 0,192× | 1,12× | 2,1× | 2,25× |
| pool pronto sem cópia, 4 processos | 0,202× | 1,31× | 3,17× | 4,04× |
| pool pronto por pickle, 2 processos | 1/12 | 0,121× | 0,162× | 0,192× |
| pool pronto por pickle, 4 processos | 1/12 | 0,208× | 0,231× | 0,216× |
| vetorizada, 2 threads do BLAS | 0,988× | 1,29× | 1,39× | 1,36× |
| vetorizada, 4 threads do BLAS | 1,09× | 1,47× | 1,7× | 1,76× |
| vetorizada, lote 4.096 | 0,956× | 1,03× | 0,997× | 1,37× |
| vetorizada, lote 512 | 1,04× | 1,06× | 1,13× | 1,39× |
| fundida y = Xw + b | 5,6× | 2,92× | 2,89× | 2,88× |
| escalar (referência) | 1/2.758 | 1/1.948 | 1/1.544 | — |

## Tabela 3. Para onde vai o tempo da execução paralela

| Conjunto | Processos | Vetorizada | Pool pronto, sem cópia | Pool pronto | Completa | Validação + cópia (pronto − sem cópia) | Criar e encerrar processos + alocar memória compartilhada (completa − pronto) |
|---|---|---|---|---|---|---|---|
| grande | 2 | 21,8 | 9,62 | 25,8 | 172 | 13,9 | 142 |
| grande | 4 | 21,8 | 6,72 | 20,0 | 175 | 13,3 | 155 |
| enorme | 2 | 279 | 123 | 226 | 605 | 104 | 382 |
| enorme | 4 | 279 | 71,7 | 179 | 573 | 106 | 389 |

## Tabela 4. Custo por linha adicional

| Estratégia | Custo por linha adicional (ns), grande → enorme | Teto do speedup (n → ∞) |
|---|---|---|
| vetorizada, 1 processo (base) | 286 | 1× |
| paralela completa, 2 processos | 481 | 0,595× |
| paralela completa, 4 processos | 443 | 0,646× |
| pool pronto, 2 processos | 223 | 1,28× |
| pool pronto, 4 processos | 177 | 1,62× |
| pool pronto sem cópia, 2 processos | 126 | 2,27× |
| pool pronto sem cópia, 4 processos | 72 | 3,97× |
| pool pronto por pickle, 2 processos | 1494 | 0,192× |
| pool pronto por pickle, 4 processos | 1349 | 0,212× |
| vetorizada, 2 threads do BLAS | 210 | 1,36× |
| vetorizada, 4 threads do BLAS | 159 | 1,8× |
| vetorizada, lote 4.096 | 204 | 1,41× |
| vetorizada, lote 512 | 204 | 1,4× |
| fundida y = Xw + b | 101 | 2,85× |

## Tabela 5. Tamanho do lote

| Conjunto | Processos | Lote | Tarefas | Mediana [mín – máx] ms | Speedup vs vetorizada (lote único) |
|---|---|---|---|---|---|
| grande | 2 | 50.000 | 2 | 25,8 [21,1 – 38,2] | 0,817× |
| grande | 2 | 4.096 | 25 | 25,1 [20,3 – 32,2] | 0,86× |
| grande | 2 | 512 | 196 | 26,9 [22,9 – 60,1] | 0,778× |
| grande | 4 | 25.000 | 4 | 20,0 [18,6 – 24,1] | 1,12× |
| grande | 4 | 4.096 | 25 | 21,3 [16,1 – 34,5] | 1,06× |
| grande | 4 | 512 | 196 | 23,0 [21,2 – 28,7] | 0,918× |
| grande | 1 | 4.096 | 25 | 21,0 [18,3 – 27,8] | 0,997× |
| grande | 1 | 512 | 196 | 19,4 [18,1 – 22,2] | 1,13× |
| enorme | 2 | 500.000 | 2 | 226 [196 – 264] | 1,24× |
| enorme | 2 | 4.096 | 245 | 200 [194 – 239] | 1,38× |
| enorme | 2 | 512 | 1954 | 229 [216 – 252] | 1,23× |
| enorme | 4 | 250.000 | 4 | 179 [173 – 200] | 1,57× |
| enorme | 4 | 4.096 | 245 | 171 [157 – 185] | 1,66× |
| enorme | 4 | 512 | 1954 | 198 [174 – 217] | 1,41× |
| enorme | 1 | 4.096 | 245 | 204 [195 – 227] | 1,37× |
| enorme | 1 | 512 | 1954 | 203 [177 – 213] | 1,39× |

## Tabela 6. Amortização do preparo da fusão

| Conjunto | Preparo de w e b (uma vez) | Economia por chamada (ms) | Chamadas para amortizar | Speedup vs vetorizada |
|---|---|---|---|---|
| pequeno | 3,7 µs | 0,0997 | 1 | 5,6× |
| medio | 3,7 µs | 1,08 | 1 | 2,92× |
| grande | 3,7 µs | 14,2 | 1 | 2,89× |
| enorme | 3,7 µs | 183 | 1 | 2,88× |

## Tabela 7. Custo da referência escalar

| Parte | Código | ns | Vezes na linha | % da linha |
|---|---|---|---|---|
| linha inteira do laço interno | `a += float(X[i, j]) * float(m.W1[j, k])` | 151 | 1 | 100% |
| ler um elemento do array e converter | `float(X[i, j])` | 64 | 2 | 85% |
| só a leitura (cria um numpy.float64) | `X[i, j]` | 54 | 2 | 71% |
| buscar o atributo m.W1 do dataclass | `m.W1` | 3 | 1 | 2% |
| multiplicar e acumular floats | `a += b * c` | 10 | 1 | 7% |

Estimativa: 2.080 execuções por observação × 151 ns = 314 µs por observação. Medido no conjunto grande: 338 µs.
