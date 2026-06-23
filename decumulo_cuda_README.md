# Simulazioni su GPU — `decumulo_cuda.py`

Accelerazione CUDA delle simulazioni Monte Carlo del progetto **Decumulo**.
Il modulo esegue le simulazioni di `decumulo_corretto.ipynb` sulla GPU con un
approccio **thread-per-simulazione**: ogni thread CUDA esegue *una* simulazione
completa con la **stessa identica logica scalare** del loop Python del notebook
(cicli BTP, vendite condizionate del 60/40 strategico, `break`, FIX bug 1–5).

Non è una riscrittura "vettorizzata": è lo stesso algoritmo, eseguito in parallelo
su migliaia di thread. Per questo i risultati sono fedeli al loop CPU.

**Speedup di riferimento: ~660–700×** (es. 2000 simulazioni: ~41 s su CPU →
~0,06 s su GPU; il guadagno effettivo dipende dalla tua GPU e dal numero di
simulazioni).

> **Nota sui comandi.** In questa guida i comandi usano `python` e `pip`
> "semplici": si intende che tu abbia **prima attivato il tuo environment Python**
> (vedi [§0](#0-il-tuo-environment-python)). Sostituisci nomi di environment,
> percorsi e versioni con quelli del tuo sistema.

---

## 0. Il tuo environment Python

Tutti i comandi vanno eseguiti **dentro l'environment** in cui usi il notebook,
qualunque sia il suo nome o tipo. Alcuni esempi:

- **conda / mamba / miniforge**:
  ```bash
  conda activate IL_TUO_ENV      # es. conda activate decumulo
  ```
- **venv / virtualenv**:
  ```bash
  source /percorso/al/tuo/venv/bin/activate        # Linux/macOS
  .\percorso\al\tuo\venv\Scripts\activate          # Windows (PowerShell)
  ```
- **nessun environment dedicato**: usi direttamente il `python` di sistema (non
  consigliato, ma funziona).

Dopo l'attivazione, `python` e `pip` puntano all'interprete giusto. Se preferisci
non attivare l'environment, puoi sempre richiamare gli eseguibili per percorso
assoluto (es. `/percorso/env/bin/python -m pip ...`), ma negli esempi qui sotto
useremo la forma attivata, più leggibile.

Per sapere quale interprete è attivo: `python -c "import sys; print(sys.executable)"`.

---

## 1. Requisiti

### Hardware
- Una **GPU NVIDIA** con architettura supportata da Numba/CUDA.
  (Sviluppato e testato su una RTX 4060 Ti, Compute Capability 8.9, ma non c'è
  nulla di specifico per quel modello.)
- Memoria GPU: il fabbisogno è modesto — gli array dei rendimenti occupano
  circa `14 × mesi × simulazioni × 4 byte` in float32
  (es. 600 mesi × 10000 simulazioni ≈ 340 MB).

### Software di sistema
- **Driver NVIDIA** recente: **è l'unico requisito non aggirabile**, senza driver
  non c'è accesso alla GPU. Verifica con `nvidia-smi`.
- **Librerie di compilazione CUDA**: `libNVVM` + `libdevice`, `NVRTC`, `cudart`.
  Numba **non usa `nvcc`** (compila Python → NVVM IR → PTX tramite libNVVM), quindi
  il compilatore da riga di comando *non* è necessario. Puoi ottenere queste
  librerie in uno qualsiasi di questi modi:
  - **CUDA Toolkit** installato a livello di sistema (via consigliata con CUDA 13);
  - via **pip**, senza toolkit completo — **disponibile per CUDA 12.x**
    (per CUDA 13 queste wheel non sono ancora pubblicate su PyPI):
    ```bash
    pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12
    ```
    (il pacchetto `...-nvcc-...` porta libNVVM/libdevice, non il comando `nvcc`);
  - via **conda**:
    ```bash
    conda install -c nvidia cuda-nvrtc cuda-nvcc
    ```

  Per vedere cosa Numba sta effettivamente usando sul tuo sistema:
  `python -m numba -s` (sezione `__CUDA Information__`).

### Pacchetti Python
Oltre alle dipendenze già usate dal notebook
(`pandas numpy tqdm openpyxl matplotlib ipython`), servono **`numba`** e
**`numba-cuda`**:

```bash
pip install numba numba-cuda
```

> Nota: il supporto CUDA di Numba vive ora nel pacchetto separato **`numba-cuda`**,
> che va installato **insieme** a `numba`.

Versioni note funzionanti: `numba` 0.65.1 + `numba-cuda` 0.30.2 (Python 3.14). In
generale usa versioni recenti di entrambi, coerenti con la tua versione di Python.

### Verifica rapida dell'installazione
```bash
python -c "from numba import cuda; print('CUDA OK:', cuda.is_available()); cuda.detect()"
```
Deve stampare `CUDA OK: True` ed elencare la tua GPU come `[SUPPORTED]`.

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

Perché l'`import` funzioni, avvia Jupyter dalla cartella del progetto (così
`decumulo_cuda.py` è importabile) e seleziona come **kernel del notebook** lo stesso
environment in cui hai installato `numba`/`numba-cuda`.

**Come si usa in pratica:** esegui le celle del notebook in ordine fino alla cella
GPU. Dopo, `risultati` e `sopravvive` hanno **lo stesso formato** del loop CPU
(DataFrame con colonne = `strategie`), quindi tutte le celle di analisi/grafici a
valle funzionano senza modifiche.

Per tornare al motore CPU basta impostare `USA_CUDA = False` (in quel caso
`decumulo_cuda.py` e Numba non sono nemmeno necessari).

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
    CAPITALE, BUFFER, BUFFER_IDEALE, PREZZO, PREZZO_CARICO0, PREZZO_CARICO1,  # init
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

Funziona anche su Windows: **il codice Python è identico** (`decumulo_cuda.py` e il
notebook non cambiano) e, a parità di GPU, i **risultati numerici sono identici** a
quelli su Linux. Cambiano solo l'installazione e un dettaglio del driver.

- **Pacchetti**: con l'environment attivato, `pip install numba numba-cuda` come su
  Linux. Esistono le wheel per Windows (incluse quelle per Python 3.14).
- **Driver NVIDIA per Windows**: obbligatorio (come su Linux). Verifica con
  `nvidia-smi`.
- **Librerie CUDA su Windows**: usa il **CUDA Toolkit per Windows** (installer
  ufficiale) — via consigliata, soprattutto con CUDA 13. L'alternativa "via pip" è
  disponibile per **CUDA 12.x** (le wheel `nvidia-*-cu12` esistono anche per
  Windows; per CUDA 13 non ancora).
- **Non serve Visual Studio / MSVC**: Numba compila i kernel via libNVVM, non con
  un compilatore C++.
- **Watchdog TDR (il punto da conoscere su Windows)**: su GeForce in modalità WDDM
  la GPU pilota anche il display e Windows applica un timeout (~2 secondi) ai kernel,
  oltre il quale resetta il driver. **Qui non è un problema**: i kernel durano una
  frazione di secondo. Diventerebbe rilevante solo aumentando enormemente
  mesi/simulazioni per singolo lancio (in tal caso si può alzare il limite TDR nel
  registro di sistema, oppure spezzare il lavoro in più lanci).
- **Attivazione environment**: su PowerShell, ad es.
  `conda activate IL_TUO_ENV` (conda) oppure
  `.\percorso\al\venv\Scripts\activate` (venv); poi usa `python`.
- **Alternativa WSL2**: in WSL2 con CUDA si segue esattamente la procedura Linux di
  questo README.

---

## 7. Note e risoluzione problemi

- **Gli indici delle strategie sono accoppiati all'ordine di `strategie`**
  (esattamente come `aliquote0/aliquote1`, vedi `CLAUDE.md`). Se riordini
  `strategie`, aggiorna le costanti `I_*` in testa a `decumulo_cuda.py`.
- **`cuda.is_available()` è `False`**: controlla il driver (`nvidia-smi`), che le
  librerie CUDA siano raggiungibili (vedi `python -m numba -s`) e che `numba-cuda`
  sia installato **nello stesso environment** del kernel Jupyter che stai usando.
- **`ModuleNotFoundError: numba` nel notebook**: il kernel del notebook punta a un
  environment diverso da quello in cui hai installato i pacchetti; cambia kernel o
  installa i pacchetti nell'environment giusto.
- **`NumbaPerformanceWarning: Grid size ... low occupancy`**: solo un avviso,
  compare quando le simulazioni sono poche; innocuo.
- **Prima esecuzione più lenta**: la prima chiamata compila il kernel (qualche
  secondo). Le chiamate successive con lo stesso `dtype` usano la cache in memoria.
- **Risultati diversi dalla CPU**: atteso con `float32`; passa a
  `dtype=np.float64` per la corrispondenza esatta.
- **Nessuna GPU disponibile**: imposta `USA_CUDA = False` nel notebook per usare
  il loop Python di riferimento.
