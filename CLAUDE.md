# CLAUDE.md — Progetto "Decumulo"

Simulatore Monte Carlo di strategie di **decumulo** di un patrimonio (notebook di P. Coletti, corretto). Confronta 9 strategie di prelievo a valore reale costante su 50 anni, campionando blocchi storici di rendimenti azionari/obbligazionari/inflazione. Codice e commenti sono in **italiano**: mantienili in italiano.

## File
- `decumulo_corretto.ipynb` — notebook principale (già patchato, vedi sotto).
- `tassi_btp_eurobond.xlsx` — fogli: **BTP** (`Date,BTP3,BTP5,BTP10`, yield in decimale), **Eurozone** (col.5 = iBoxx Eurozone Gov **total-return**, base 100 al 31/12/1998; col.8 = iBoxx EZ Gov **inflation-linked** TR, attualmente NON usato), più ESTER/Italy/Amundi/Euribor non usati.
- `prezzi_al_consumo_NIC.xlsx` — indice NIC mensile (inflazione IT), 1996→2025.
- Azionario: scaricato a runtime da `raw.githubusercontent.com/paolocole/...` (MSCI WORLD NET EUR).

## Come si esegue
- Cella `CONFIG`: `BASE = "./"` (file locali) oppure l'URL di Coletti (commentato lì) per Colab.
- Dipendenze: `pandas numpy tqdm openpyxl matplotlib ipython`.
- Parametri chiave in cima: `capitale_iniziale=600000`, `prelievo_iniziale=26000/12`, `anni_simulazione=50`, `numero_simulazioni=10000`. Per test rapidi abbassa `numero_simulazioni` a ~500.
- Le **prime 6 colonne** di `indici` sono sequenze "killer" deterministiche (alta inflazione, BTP bassi, ecc.); le statistiche "reali" usano `[6:]`.
- Output: `risultati` (ricchezza finale per sim×strategia) e `sopravvive` (mese di morte; pieno = sopravvissuto i 50 anni). Una run completa è ~minuti in puro Python.

## Bug già corretti in questo notebook (marcati `# FIX bug N`)
1. **60/40 semplice** ora cresce col rendimento 60/40, non con quello azionario *(impatto grosso: arrivo a 50a 43%→34%)*.
2. `buffer_ideale` rivalutato con inflazione grezza, coerente coi prelievi.
3. Interessi buffer ed ETF inflation-linked usano inflazione grezza (`nic_grezza_estratta`).
4. Sotto-conti delle strategie composite azzerati alla morte (no "resurrezione"; impatto ~nullo ma corretto).
5. Sentinella morte `-1` invece di `0` (0 collideva col mese 0); aggiornate tutte le guardie + init `V40/V60`.

## Scelte di modellazione NON toccate (sono semplificazioni dell'autore, non bug)
- Obbligazioni singole **senza rischio prezzo** (vendute sempre a 100): sottostima il rischio tasso (vedi 2022).
- Nessuna compensazione di minusvalenze (zoccolo fiscale): perdite azzerate con `clip(...,0,None)`.
- Smoothing dell'inflazione/BTP via **EWMA ricorsiva** (`(prec+nuovo)/2`): nel codice diverge da come è descritto nel markdown (media col mese reale successivo). Lasciato com'è.
- Linker modellato come `NIC + EUROZONE/4` (sintetico). Esiste un indice **reale** (Eurozone col.8) se si vuole sostituirlo.

## Convenzioni / trappole
- `strategie` è la lista ordinata; `si[nome]` mappa nome→indice. Non riordinare senza aggiornare `aliquote0/aliquote1`.
- Per A/B puliti: fissare il seed prima di `np.random.randint` (cella `indici = ...`) così le estrazioni sono identiche tra varianti.
- Se su pandas molto recenti `rendimenti` risulta vuoto dopo il `dropna`: il 60/40 va costruito **dopo** il dropna (NaN azionari pre-2000 avvelenano il prodotto cumulato) — già rispettato nel notebook.

## Possibili task futuri
- Sostituire il linker sintetico con l'indice reale (Eurozone col.8).
- Dare un minimo di rischio prezzo ai BTP singoli.
- 60/40 "Lifestrategy" (ribilanciato senza tasse) e strategia multi-ETF stesso indice (idee già nel markdown finale).
