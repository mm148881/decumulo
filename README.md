# Decumulo — Simulatore Monte Carlo di strategie di prelievo

Simulatore **Monte Carlo** che confronta diverse strategie di **decumulo** di un
patrimonio: si parte da un capitale e si preleva una somma fissa *in valore reale*
(rivalutata con l'inflazione) per molti anni. La domanda a cui risponde è:
**quale strategia di investimento/prelievo fa durare più a lungo il patrimonio e
ne lascia di più alla fine?**

Per ogni strategia vengono simulate migliaia di "vite" possibili (percorsi di
mercato), campionando rendimenti **storici** di azionario, obbligazionario, titoli
di stato e inflazione.

Il progetto è una revisione (con alcuni bug corretti e un'accelerazione su GPU) del
notebook originale di **Paolo Coletti**.

---

## Cosa simula

**9 strategie** di decumulo a prelievo reale costante:

1. **distribuzione** — ETF azionario a distribuzione (dividendi + vendite).
2. **obblig 10a** — scala di BTP a 10 anni (*bond ladder*).
3. **obblig 5a** — scala di BTP a 5 anni.
4. **accumulo** — ETF azionario ad accumulo, si vende solo il necessario.
5. **accumulo + obb 10a** — mix azionario ad accumulo + scala di BTP a 10 anni.
6. **etf obb indic** — ETF obbligazionario indicizzato all'inflazione (sintetico).
7. **buffer 3a** — azionario ad accumulo con cuscinetto di liquidità di 3 anni.
8. **60/40 strategico** — 60% azioni / 40% obbligazioni con prelievo "intelligente"
   (vende azioni se sono salite, obbligazioni se scese).
9. **60/40** — 60/40 ribilanciato semplice, vendita pro-quota.

In tutte le strategie c'è un **buffer** di liquidità da cui escono davvero le spese
mensili, ricostituito vendendo gli asset.

---

## Input e output

**Input**
- Dati storici: azionario (MSCI World Net EUR, scaricato a runtime), obbligazionario
  (iBoxx Eurozone Gov total-return), titoli di stato (BTP 1–10 anni), inflazione
  (indice NIC italiano).
- Parametri (modificabili in cima al notebook): capitale iniziale, prelievo
  iniziale, orizzonte in anni, numero di simulazioni, parametri fiscali.

**Output** — due tabelle `simulazioni × strategie`:
- `risultati` — ricchezza finale (capitale + buffer) a fine orizzonte; `0` se il
  patrimonio si è esaurito prima.
- `sopravvive` — mese in cui il patrimonio si è esaurito; il valore massimo
  (`12 × anni`) indica che è sopravvissuto per tutto l'orizzonte. Diviso per 12 →
  anni di sopravvivenza.

Le **prime 6 simulazioni** sono scenari avversi deterministici ("killer": alta
inflazione, BTP bassi, ecc.); le statistiche "vere" usano le simulazioni successive.

---

## Struttura dei file

| File | Descrizione |
|---|---|
| `decumulo_corretto.ipynb` | Notebook principale (versione corretta). |
| `decumulo_cuda.py` | Accelerazione GPU delle simulazioni (Numba CUDA). |
| `decumulo_cuda_README.md` | Documentazione: requisiti, installazione e uso della parte GPU. |
| `CLAUDE.md` | Note tecniche di progetto (bug corretti, scelte di modellazione, trappole). |
| `*.xlsx` | Dati locali (tassi BTP/eurobond, indice NIC). |

---

## Come si esegue

Requisiti Python:
```bash
pip install pandas numpy tqdm openpyxl matplotlib ipython
```

1. Apri `decumulo_corretto.ipynb` in Jupyter.
2. Assicurati che la cella `CONFIG` abbia `BASE = "./"` (file locali).
3. Esegui tutte le celle (*Restart Kernel and Run All*). L'azionario viene scaricato
   automaticamente; serve quindi una connessione a internet.

Per test rapidi si può abbassare `numero_simulazioni`.

### Accelerazione GPU (opzionale)
Le simulazioni possono girare su GPU NVIDIA (~centinaia di volte più veloci) tramite
`decumulo_cuda.py`: basta impostare `USA_CUDA = True` nel notebook. Requisiti e
installazione sono descritti in **[`decumulo_cuda_README.md`](decumulo_cuda_README.md)**.
Senza GPU, imposta `USA_CUDA = False` per usare il loop Python di riferimento.

---

## Crediti

Basato sul notebook originale di **Paolo Coletti**. Questa versione corregge alcuni
bug del modello (documentati in `CLAUDE.md`) e aggiunge l'esecuzione su GPU.

## Licenza

Distribuito sotto licenza **GNU General Public License v3** (vedi file `LICENSE`).
