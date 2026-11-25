# Resumo dos resultados

Gerado por `03-hpc/resumo.py`.

## Ambiente

```json
{
  "sistema": "Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.43",
  "cpu": "11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz",
  "cpus_logicas": 8,
  "memoria_total_gb": 16.3,
  "python": "3.12.14",
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
  "preparo_fundido_s": 3.4280019463039935e-06,
  "rtol": 1e-10,
  "atol": 1e-10,
  "modelo": {
    "arquivo": "/home/dell/projects/cloudwalk_challenge_3/03-hpc/modelo/modelo.npz",
    "arquitetura": [
      64,
      32,
      1
    ]
  }
}
```

## Tabela 1. Entrega mínima

| Conjunto (linhas) | Estratégia | Lote (tarefas) | Mediana [mín – máx] ms | Throughput (obs/s) | Speedup vs vetorizada |
|---|---|---|---|---|---|
| pequeno (1.000) | escalar (referência) | 1.000 (1) | 851 [783 – 1.020] | 1.175 | 1/4.778 |
| pequeno (1.000) | vetorizada, 1 processo | 1.000 (1) | 0,176 [0,161 – 0,276] | 5.694.209 | 1× |
| pequeno (1.000) | paralela completa, 2 processos | 500 (2) | 224 [197 – 265] | 4.471 | 1/1.222 |
| medio (10.000) | escalar (referência) | 10.000 (1) | 8.973 [8.019 – 10.325] | 1.114 | 1/3.625 |
| medio (10.000) | vetorizada, 1 processo | 10.000 (1) | 2,59 [1,86 – 8,82] | 3.867.482 | 1× |
| medio (10.000) | paralela completa, 2 processos | 5.000 (2) | 242 [198 – 389] | 41.240 | 1/95 |
| grande (100.000) | escalar (referência) | 100.000 (1) | 77.178 [53.945 – 89.758] | 1.296 | 1/2.919 |
| grande (100.000) | vetorizada, 1 processo | 100.000 (1) | 23,1 [18,4 – 30,4] | 4.335.082 | 1× |
| grande (100.000) | paralela completa, 2 processos | 50.000 (2) | 252 [177 – 353] | 397.093 | 1/11 |
| enorme (1.000.000) | vetorizada, 1 processo | 1.000.000 (1) | 229 [211 – 258] | 4.373.044 | 1× |
| enorme (1.000.000) | paralela completa, 2 processos | 500.000 (2) | 652 [524 – 1.469] | 1.534.859 | 0,357× |

## Tabela 2. Quando cada estratégia compensa

| Speedup vs vetorizada (mediana pareada) | pequeno | medio | grande | enorme |
|---|---|---|---|---|
| paralela completa, 2 processos | 1/1.222 | 1/95 | 1/11 | 0,357× |
| paralela completa, 4 processos | 1/1.467 | 1/118 | 1/13 | 0,335× |
| pool pronto, 2 processos | 0,135× | 0,66× | 1,11× | 1,2× |
| pool pronto, 4 processos | 0,111× | 0,723× | 1,31× | 1,48× |
| pool pronto sem cópia, 2 processos | 0,13× | 1,01× | 1,59× | 1,93× |
| pool pronto sem cópia, 4 processos | 0,138× | 1,04× | 2,41× | 2,75× |
| pool pronto por pickle, 2 processos | 1/16 | 0,147× | 0,174× | 0,155× |
| pool pronto por pickle, 4 processos | 1/13 | 0,161× | 0,193× | 0,166× |
| vetorizada, 2 threads do BLAS | 1,16× | 1,25× | 1,32× | 1,22× |
| vetorizada, 4 threads do BLAS | 0,809× | 1,12× | 1,29× | 1,21× |
| vetorizada, lote 4.096 | 0,907× | 1,18× | 1,14× | 1,22× |
| vetorizada, lote 512 | 0,813× | 1,18× | 1,11× | 1,14× |
| fundida y = Xw + b | 4,76× | 5,13× | 3,29× | 3,24× |
| escalar (referência) | 1/4.778 | 1/3.625 | 1/2.919 | — |

## Tabela 3. Para onde vai o tempo da execução paralela

| Conjunto | Processos | Vetorizada | Pool pronto, sem cópia | Pool pronto | Completa | Validação + cópia (pronto − sem cópia) | Criar e encerrar processos + alocar memória compartilhada (completa − pronto) |
|---|---|---|---|---|---|---|---|
| grande | 2 | 23,1 | 13,3 | 20,6 | 252 | 7,05 | 233 |
| grande | 4 | 23,1 | 10,3 | 17,0 | 297 | 6,97 | 280 |
| enorme | 2 | 229 | 123 | 191 | 652 | 66,2 | 465 |
| enorme | 4 | 229 | 82,9 | 151 | 694 | 65,5 | 537 |

## Tabela 4. Custo por linha adicional

| Estratégia | Custo por linha adicional (ns), grande → enorme | Teto do speedup (n → ∞) |
|---|---|---|
| vetorizada, 1 processo (base) | 228 | 1× |
| paralela completa, 2 processos | 444 | 0,514× |
| paralela completa, 4 processos | 441 | 0,519× |
| pool pronto, 2 processos | 190 | 1,2× |
| pool pronto, 4 processos | 149 | 1,54× |
| pool pronto sem cópia, 2 processos | 121 | 1,88× |
| pool pronto sem cópia, 4 processos | 81 | 2,83× |
| pool pronto por pickle, 2 processos | 1461 | 0,156× |
| pool pronto por pickle, 4 processos | 1428 | 0,16× |
| vetorizada, 2 threads do BLAS | 184 | 1,24× |
| vetorizada, 4 threads do BLAS | 188 | 1,22× |
| vetorizada, lote 4.096 | 185 | 1,23× |
| vetorizada, lote 512 | 194 | 1,18× |
| fundida y = Xw + b | 71 | 3,23× |

## Tabela 5. Tamanho do lote

| Conjunto | Processos | Lote | Tarefas | Mediana [mín – máx] ms | Speedup vs vetorizada (lote único) |
|---|---|---|---|---|---|
| grande | 2 | 50.000 | 2 | 20,6 [17,1 – 29,7] | 1,11× |
| grande | 2 | 4.096 | 25 | 22,6 [17,7 – 29,8] | 1× |
| grande | 2 | 512 | 196 | 31,8 [21,6 – 43,0] | 0,716× |
| grande | 4 | 25.000 | 4 | 17,0 [14,4 – 21,2] | 1,31× |
| grande | 4 | 4.096 | 25 | 18,2 [14,6 – 23,1] | 1,25× |
| grande | 4 | 512 | 196 | 23,8 [20,1 – 38,5] | 0,905× |
| grande | 1 | 4.096 | 25 | 20,4 [16,0 – 24,9] | 1,14× |
| grande | 1 | 512 | 196 | 21,7 [16,9 – 30,2] | 1,11× |
| enorme | 2 | 500.000 | 2 | 191 [175 – 384] | 1,2× |
| enorme | 2 | 4.096 | 245 | 193 [170 – 243] | 1,2× |
| enorme | 2 | 512 | 1954 | 246 [214 – 293] | 0,911× |
| enorme | 4 | 250.000 | 4 | 151 [141 – 172] | 1,48× |
| enorme | 4 | 4.096 | 245 | 145 [131 – 208] | 1,55× |
| enorme | 4 | 512 | 1954 | 361 [239 – 430] | 0,655× |
| enorme | 1 | 4.096 | 245 | 187 [169 – 233] | 1,22× |
| enorme | 1 | 512 | 1954 | 197 [175 – 228] | 1,14× |

## Tabela 6. Amortização do preparo da fusão

| Conjunto | Preparo de w e b (uma vez) | Economia por chamada (ms) | Chamadas para amortizar | Speedup vs vetorizada |
|---|---|---|---|---|
| pequeno | 3,4 µs | 0,134 | 1 | 4,76× |
| medio | 3,4 µs | 1,89 | 1 | 5,13× |
| grande | 3,4 µs | 16,5 | 1 | 3,29× |
| enorme | 3,4 µs | 158 | 1 | 3,24× |

## Tabela 7. Custo da referência escalar

| Parte | Código | ns | Vezes na linha | % da linha |
|---|---|---|---|---|
| linha inteira do laço interno | `a += float(X[i, j]) * float(m.W1[j, k])` | 362 | 1 | 100% |
| ler um elemento do array e converter | `float(X[i, j])` | 152 | 2 | 84% |
| só a leitura (cria um numpy.float64) | `X[i, j]` | 121 | 2 | 67% |
| buscar o atributo m.W1 do dataclass | `m.W1` | 6 | 1 | 2% |
| multiplicar e acumular floats | `a += b * c` | 29 | 1 | 8% |

Estimativa: 2.080 execuções por observação × 362 ns = 752 µs por observação. Medido no conjunto grande: 772 µs.

## Tabela 8. Memória

| Acréscimo de RSS em MB (amostras) | grande | enorme |
|---|---|---|
| vetorizada, 1 processo | 0 (7) | 256 (102) |
| paralela completa, 2 processos | 206 (69) | 1321 (238) |
| paralela completa, 4 processos | 290 (57) | 1339 (191) |
| pool pronto, 2 processos | 0 (6) | 256 (46) |
| pool pronto, 4 processos | 6 (5) | 256 (42) |
| pool pronto por pickle, 2 processos | 50 (29) | 537 (329) |

## Tabela 9. Rede mais larga, 64 → 1024 → 1

| Rede 64 → 1024 → 1: speedup vs vetorizada | largo_pequeno | largo_medio | largo_grande |
|---|---|---|---|
| vetorizada, 1 processo (ms) | 4,11 | 51,9 | 454 |
| paralela completa, 2 processos | 1/69 | 0,167× | 0,751× |
| paralela completa, 4 processos | 1/126 | 1/11 | 0,551× |
| pool pronto, 2 processos | 1,19× | 1,66× | 1,53× |
| pool pronto, 4 processos | 1,27× | 2,28× | 1,91× |
| pool pronto por pickle, 2 processos | 0,944× | 1,32× | 1,21× |
| vetorizada, 4 threads do BLAS | 2,14× | 1,81× | 1,48× |

## Tabela 10. Varredura da camada oculta, 100 mil linhas

| 100 mil linhas; camada oculta → | 8 | 32 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| vetorizada, 1 processo (ms) | 10,6 | 21,8 | 67,7 | 235 | 495 | 978 |
| paralela completa, 2 processos | 1/16 | 0,115× | 0,225× | 0,475× | 0,495× | 0,818× |
| paralela completa, 4 processos | 1/19 | 1/11 | 0,133× | 0,392× | 0,554× | 0,826× |
| pool pronto, 2 processos | 0,859× | 1,26× | 1,54× | 1,56× | 1,34× | 1,42× |
| pool pronto, 4 processos | 1× | 1,35× | 2,15× | 1,9× | 1,87× | 1,76× |
| vetorizada, 2 threads do BLAS | 0,977× | 1,38× | 1,54× | 1,47× | 1,34× | 1,09× |
| vetorizada, 4 threads do BLAS | 1,02× | 1,41× | 1,54× | 1,6× | 1,53× | 1,62× |
