# Simulazioni su GPU — `decumulo_cuda.py`

Accelerazione CUDA delle simulazioni Monte Carlo del progetto **Decumulo**.
Il modulo esegue le simulazioni di `decumulo_corretto.ipynb` sulla GPU con un
approccio **thread-per-simulazione**: ogni thread CUDA esegue *una* simulazione
completa con la **stessa identica logica scalare** del loop Python del notebook
(cicli BTP, vendite condizionate del 60/40 strategico, `break`, FIX bug 1–5).

Non è una riscrittura "vettorizzata": è lo stesso algoritmo, eseguito in parallelo
su migliaia di thread. Per questo i risultati sono fedeli al loop CPU.

**Speedup misurato: ~660–700×** (es. 2000 simulazioni: 41 s su CPU → 0,06 s su GPU).

---

## 1. Requisiti

### Hardware
- Una **GPU NVIDIA** (architettura supportata da Numba/CUDA).
  Testato su **RTX 4060 Ti 16 GB** (Compute Capability 8.9).
- Memoria GPU: il fabbisogno è modesto (gli array dei rendimenti sono
  `~14 × mesi × simulazioni × 4 byte`; per 600 mesi × 10000 sim ≈ 340 MB in float32).

### Software di sistema
- **Driver NVIDIA** recente (testato con 610.43.02). **È l'unico requisito non
  aggirabile**: senza driver non c'è accesso alla GPU. Verifica con `nvidia-smi`.
- **Librerie di compilazione CUDA**: `libNVVM` + `libdevice`, `NVRTC`, `cudart`.
  Numba **non usa `nvcc`** (compila Python → NVVM IR → PTX tramite libNVVM), quindi
  il compilatore da riga di comando *non* è necessario. Queste librerie si possono
  ottenere in uno qualsiasi di questi modi:
  - **CUDA Toolkit** di sistema (è la configurazione testata: `/opt/cuda` 13.3).
    È la via consigliata con CUDA 13.
  - via **pip**, senza toolkit completo — **disponibile solo per CUDA 12.x**
    (per CUDA 13 queste wheel non sono ancora pubblicate su PyPI):
    `pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12`
    (il pacchetto `...-nvcc-...` porta libNVVM/libdevice, non il comando `nvcc`);
  - via **conda**: `conda install -c nvidia cuda-nvrtc cuda-nvcc`.

  Per vedere cosa Numba sta effettivamente usando sul tuo sistema:
  `python -m numba -s` (sezione `__CUDA Information__`).

### Pacchetti Python (environment conda `PColetti`, Python 3.14)
Oltre alle dipendenze già usate dal notebook (`pandas numpy tqdm openpyxl matplotlib ipython`):

```bash
/opt/miniforge3/envs/PColetti/bin/pip install numba numba-cuda
```

Versioni testate: `numba` 0.65.1, `numba-cuda` 0.30.2.

> Nota: il supporto CUDA di Numba è ora nel pacchetto separato **`numba-cuda`**;
> va installato insieme a `numba`.

### Verifica rapida dell'installazione
```bash
/opt/miniforge3/envs/PColetti/bin/python -c "from numba import cuda; print('CUDA OK:', cuda.is_available()); cuda.detect()"
```
Deve stampare `CUDA OK: True` ed elencare la GPU come `[SUPPORTED]`.

---

## 2. Uso dal notebook

L'integrazione è già presente in `decumulo_corretto.ipynb`. Vicino alla cella
delle simulazioni trovi tre celle:

1. **Cella del flag** (subito prima del loop):
   ```python
   USA_CUDA = True     # True -> GPU ; False -> loop Python (riferimento/fallback)
   ```

2. **Loop Python** (cella che inizia con `if not USA_CUDA:`): resta come
   riferimento e fallback. Viene eseguito solo se `USA_CUDA = False`.

3. **Cella GPU** (subito dopo il loop): se `USA_CUDA` è `True`, chiama il kernel
   e popola `risultati` e `sopravvive`:
   ```python
   if USA_CUDA:
       from decumulo_cuda import simula_cuda
       risultati, sopravvive = simula_cuda(
           rendimenti_estratti, prelievi_estratti, nic_grezza_estratta,
           CAPITALE, BUFFER, BUFFER_IDEALE, PREZZO, PREZZO_CARICO0, PREZZO_CARICO1,
           aliquote0, aliquote1, CAPITALE10BTP, CAPITALE5BTP, CAPITALE10BTPacc,
           CAPITALE10accumulo, CAPITALE40, CAPITALE60, dividendo, aliquota_buffer,
           strategie, etf=etf, dtype=np.float32)
   ```

**Come si usa in pratica:** esegui le celle del notebook in ordine fino alla cella
GPU. Dopo, `risultati` e `sopravvive` hanno **lo stesso formato** del loop CPU
(DataFrame con colonne = `strategie`), quindi tutte le celle di analisi/grafici a
valle funzionano senza modifiche.

Per tornare al motore CPU basta impostare `USA_CUDA = False` (`decumulo_cuda.py`
e Numba non sono nemmeno necessari in quel caso).

---

## 3. Precisione: `float32` vs `float64`

| `dtype` | Velocità | Fedeltà al loop CPU |
|--------------|----------|----------------------|
| `np.float32` (default) | massima | err. relativo medio ~1e-5; sopravvivenza uguale ~99,98% |
| `np.float64` | quasi identica a float32 | **bit-identico** alla CPU (`rtol 1e-6`) |

Su questo kernel `float64` costa **quasi quanto** `float32` (il collo di bottiglia
è la latenza della *local memory*, non il throughput in virgola mobile). Quindi:

- usa **`float32`** per le run normali;
- usa **`dtype=np.float64`** se vuoi risultati *identici* alla CPU (es. per
  confronti A/B o validazioni), pagando pochissimo in più.

Le piccole differenze di `float32` derivano da casi-soglia in cui una simulazione
"muore" un mese prima o dopo per arrotondamento: normale e ininfluente in un
Monte Carlo. Gli **input sono identici** alla CPU (il seed resta su CPU nella
cella `indici`), quindi le uniche differenze sono di arrotondamento.

---

## 4. API: `simula_cuda(...)`

```python
risultati, sopravvive = simula_cuda(
    rendimenti_estratti,        # dict: chiavi etf, "60/40", "EUROZONE BOND", "BTP1".."BTP10"
    prelievi_estratti,          # array (mesi, simulazioni)
    nic_grezza_estratta,        # array (mesi, simulazioni) — inflazione grezza
    CAPITALE, BUFFER, BUFFER_IDEALE, PREZZO, PREZZO_CARICO0, PREZZO_CARICO1,  # init (cella 53)
    aliquote0, aliquote1,       # array (9,) — aliquote fiscali per strategia
    CAPITALE10BTP, CAPITALE5BTP, CAPITALE10BTPacc,                # scale BTP iniziali
    CAPITALE10accumulo, CAPITALE40, CAPITALE60,                   # scalari init
    dividendo, aliquota_buffer, # scalari
    strategie,                  # lista nomi strategie (= colonne dei DataFrame)
    etf="MSCI WORLD",           # chiave azionaria in rendimenti_estratti
    dtype=np.float32,           # np.float32 oppure np.float64
    threads_per_block=128,      # dimensione del blocco CUDA (raramente da toccare)
)
```

**Ritorna:** `(risultati, sopravvive)` — due `pandas.DataFrame` con una riga per
simulazione e una colonna per strategia, identici nel formato a quelli del loop CPU.

Il numero di simulazioni è dedotto dalla forma di `prelievi_estratti`
(`mesi, simulazioni`): la versione GPU esegue **sempre tutte** le simulazioni
(il flag `test` del loop CPU non si applica — la GPU è già velocissima).

---

## 5. Uso standalone (senza notebook)

`simula_cuda` è una normale funzione importabile: puoi prepararti gli array di
input come fa il notebook e chiamarla da uno script Python. Vedi la docstring in
testa a `decumulo_cuda.py` per l'esempio completo.

---

## 6. Windows

Sì, funziona anche su Windows: **il codice Python è identico** (`decumulo_cuda.py`
e il notebook non cambiano) e a parità di GPU i **risultati numerici sono identici**
a quelli su Linux. Le differenze sono solo nell'installazione e in un dettaglio del
driver.

- **Wheel disponibili**: esistono per Windows + Python 3.14
  (`numba ...cp314...win_amd64.whl`, `numba_cuda ...cp314...win_amd64.whl`).
  Installazione (con l'environment conda attivo): `pip install numba numba-cuda`.
- **Driver NVIDIA per Windows**: obbligatorio (come su Linux). Verifica con
  `nvidia-smi`.
- **Librerie CUDA su Windows**: usa il **CUDA Toolkit per Windows** (installer
  ufficiale) — è la via consigliata, soprattutto con CUDA 13. L'alternativa "via
  pip" è disponibile solo per **CUDA 12.x** (le wheel `nvidia-*-cu12` esistono anche
  per Windows; per CUDA 13 non ancora).
- **Non serve Visual Studio / MSVC**: Numba compila i kernel via libNVVM, non con
  un compilatore C++.
- **Watchdog TDR (il punto da conoscere su Windows)**: su GeForce in modalità WDDM
  la GPU pilota anche il display e Windows applica un timeout (~2 secondi) ai kernel,
  oltre il quale resetta il driver. **Qui non è un problema**: i nostri kernel durano
  ~0,06 s. Diventerebbe rilevante solo aumentando enormemente mesi/simulazioni per
  singolo lancio (in tal caso si può alzare il limite TDR nel registro di sistema,
  oppure spezzare il lavoro in più lanci).
- **Path dell'environment**: su Windows non si usa `/opt/miniforge3/envs/PColetti/bin/python`;
  attiva l'env (`conda activate PColetti`) e usa `python`, oppure l'eseguibile
  `...\miniforge3\envs\PColetti\python.exe`.
- **Alternativa WSL2**: in WSL2 con CUDA si segue esattamente la procedura Linux di
  questo README.

---

## 7. Note e risoluzione problemi

- **Gli indici delle strategie sono accoppiati all'ordine di `strategie`**
  (esattamente come `aliquote0/aliquote1`, vedi `CLAUDE.md`). Se riordini
  `strategie`, aggiorna le costanti `I_*` in testa a `decumulo_cuda.py`.
- **`cuda.is_available()` è `False`**: controlla driver (`nvidia-smi`), che
  `nvcc` sia nel PATH o che `numba-cuda` trovi il toolkit, e che `numba-cuda`
  sia installato nello stesso environment.
- **`NumbaPerformanceWarning: Grid size ... low occupancy`**: solo un avviso,
  compare quando le simulazioni sono poche; innocuo.
- **Prima esecuzione più lenta**: la prima chiamata compila il kernel (qualche
  secondo). Le chiamate successive con lo stesso `dtype` usano la cache in memoria.
- **Risultati diversi dalla CPU**: atteso con `float32`; passa a
  `dtype=np.float64` per la corrispondenza esatta.
- **Nessuna GPU disponibile**: imposta `USA_CUDA = False` nel notebook per usare
  il loop Python di riferimento.
