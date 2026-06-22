"""
Simulazione CUDA del decumulo - una simulazione per thread (thread-per-simulation).

Ogni thread GPU esegue UNA simulazione completa (loop sui mesi) con la STESSA logica
scalare del loop Python del notebook `decumulo_corretto.ipynb` (cella delle simulazioni).
Non e' una vettorizzazione "tra simulazioni": e' lo stesso codice scalare, eseguito in
parallelo su migliaia di thread. Questo mantiene la logica leggibile e fedele all'originale
(compresi i cicli sui BTP, le vendite condizionate del 60/40 strategico e i "break").

Uso tipico dal notebook:

    from decumulo_cuda import simula_cuda
    risultati, sopravvive = simula_cuda(
        rendimenti_estratti, prelievi_estratti, nic_grezza_estratta,
        CAPITALE, BUFFER, BUFFER_IDEALE, PREZZO, PREZZO_CARICO0, PREZZO_CARICO1,
        aliquote0, aliquote1, CAPITALE10BTP, CAPITALE5BTP, CAPITALE10BTPacc,
        CAPITALE10accumulo, CAPITALE40, CAPITALE60, dividendo, aliquota_buffer,
        strategie, etf=etf, dtype=np.float32)

Ritorna due DataFrame (risultati, sopravvive) con le stesse colonne `strategie`
del loop CPU.
"""
import numpy as np
import pandas as pd
from numba import cuda, float32, float64

# --- Indici delle strategie ---------------------------------------------------
# ATTENZIONE: questi indici sono accoppiati all'ORDINE della lista `strategie` del
# notebook (e quindi a aliquote0/aliquote1). Se riordini `strategie`, aggiorna anche
# qui (esattamente come avverte CLAUDE.md per aliquote0/aliquote1).
#   ["distribuzione","obblig 10a","obblig 5a","accumulo","accumulo + obb 10a",
#    "etf obb indic","buffer 3a","60/40 strategico","60/40"]
I_DIST    = 0   # distribuzione
I_OB10    = 1   # obblig 10a
I_OB5     = 2   # obblig 5a
I_ACC     = 3   # accumulo
I_ACCOB10 = 4   # accumulo + obb 10a
I_ETFOB   = 5   # etf obb indic
I_BUF3    = 6   # buffer 3a
I_6040S   = 7   # 60/40 strategico
I_6040    = 8   # 60/40
NSTRAT    = 9

_KERNELS = {}   # cache dei kernel compilati, una per dtype


def _get_kernel(flt):
    """Compila (una volta sola, con cache) il kernel per il tipo float scelto."""
    key = "f32" if flt is float32 else "f64"
    if key in _KERNELS:
        return _KERNELS[key]

    @cuda.jit
    def kernel(re_etf, re_6040, re_ezb, nic, re_btp, prelievi,
               CAPITALE, BUFFER, BUFFER_IDEALE, PREZZO, PREZZO_CARICO0, PREZZO_CARICO1,
               aliquote0, aliquote1, CAPITALE10BTP, CAPITALE5BTP, CAPITALE10BTPacc,
               CAPITALE10accumulo, CAPITALE40, CAPITALE60, dividendo, aliquota_buffer,
               n_mesi, out_ris, out_sop):
        s = cuda.grid(1)
        n_sim = out_ris.shape[0]
        if s >= n_sim:
            return

        zero = flt(0.0)
        # --- stato locale del thread (equivalente alle variabili re-inizializzate
        #     ad ogni simulazione nel loop Python) ---
        capitale = cuda.local.array(NSTRAT, flt)
        buffer = cuda.local.array(NSTRAT, flt)
        buffer_ideale = cuda.local.array(NSTRAT, flt)
        prezzo = cuda.local.array(NSTRAT, flt)
        muore = cuda.local.array(NSTRAT, flt)
        differenza_buffer = cuda.local.array(NSTRAT, flt)
        plus0 = cuda.local.array(NSTRAT, flt)
        V = cuda.local.array(NSTRAT, flt)
        for j in range(NSTRAT):
            capitale[j] = CAPITALE[j]
            buffer[j] = BUFFER[j]
            buffer_ideale[j] = BUFFER_IDEALE[j]
            prezzo[j] = PREZZO[j]
            muore[j] = flt(-1.0)

        capitale40 = CAPITALE40
        capitale60 = CAPITALE60
        capitale10accumulo = CAPITALE10accumulo

        capitale10BTP = cuda.local.array(10, flt)
        capitale10BTPacc = cuda.local.array(10, flt)
        rendimenti10BTP = cuda.local.array(10, flt)
        capitale5BTP = cuda.local.array(5, flt)
        rendimenti5BTP = cuda.local.array(5, flt)
        for t in range(10):
            capitale10BTP[t] = CAPITALE10BTP[t]
            capitale10BTPacc[t] = CAPITALE10BTPacc[t]
            rendimenti10BTP[t] = re_btp[t, 0, s]   # cedole fissate al mese 0 (BTP1..BTP10)
        for t in range(5):
            capitale5BTP[t] = CAPITALE5BTP[t]
            rendimenti5BTP[t] = re_btp[t, 0, s]    # BTP1..BTP5

        for m in range(n_mesi):
            # ********* RENDIMENTI ***************
            r1 = flt(1.0) + re_etf[m, s]
            prezzo[I_DIST] *= r1
            prezzo[I_ACC] *= r1
            prezzo[I_BUF3] *= r1
            prezzo[I_ACCOB10] *= r1
            capitale[I_DIST] *= r1
            capitale[I_ACC] *= r1
            capitale[I_BUF3] *= r1
            r6040 = flt(1.0) + re_6040[m, s]   # FIX bug 1: la 60/40 semplice cresce al 60/40
            prezzo[I_6040] *= r6040
            capitale[I_6040] *= r6040
            capitale60 *= r1
            capitale10accumulo *= r1
            # etf obb indicizzati: NIC grezza + 1/4 eurobond (FIX bug 3)
            r1 = flt(1.0) + nic[m, s] + re_ezb[m, s] / flt(4.0)
            prezzo[I_ETFOB] *= r1
            capitale[I_ETFOB] *= r1
            # parte obbligazionaria del 60/40 strategico
            r1 = flt(1.0) + re_ezb[m, s]
            prezzo[I_6040S] *= r1
            capitale40 *= r1

            # rendimento del buffer: media tra BTP1 e inflazione grezza (FIX bug 3), tassata subito
            r = (re_btp[0, m, s] + nic[m, s]) / flt(2.0)
            if r < zero:
                r = zero
            fattore_buffer = flt(1.0) + r * (flt(1.0) - aliquota_buffer)
            for j in range(NSTRAT):
                buffer[j] *= fattore_buffer
            # il buffer ideale si rivaluta con la stessa inflazione grezza dei prelievi (FIX bug 2)
            fattore_ideale = flt(1.0) + nic[m, s]
            for j in range(NSTRAT):
                buffer_ideale[j] *= fattore_ideale

            # ********* CEDOLE E DIVIDENDI ***************
            buffer[I_DIST] += capitale[I_DIST] * dividendo * (flt(1.0) - aliquote0[I_DIST])
            d1 = flt(1.0) - dividendo
            prezzo[I_DIST] *= d1
            capitale[I_DIST] *= d1
            for t in range(10):
                buffer[I_OB10] += capitale10BTP[t] * rendimenti10BTP[t] * (flt(1.0) - aliquote1[I_OB10])
                buffer[I_ACCOB10] += capitale10BTPacc[t] * rendimenti10BTP[t] * (flt(1.0) - aliquote1[I_ACCOB10])
            for t in range(5):
                buffer[I_OB5] += capitale5BTP[t] * rendimenti5BTP[t] * (flt(1.0) - aliquote1[I_OB5])

            # ********* SPENDO E RIPRISTINO IL BUFFER ***************
            prelievo = prelievi[m, s]
            for j in range(NSTRAT):
                buffer[j] -= prelievo

            for j in range(NSTRAT):
                db = buffer_ideale[j] - buffer[j]
                if db < zero:
                    db = zero
                differenza_buffer[j] = db
                # plusvalenza relativa tassabile (riusata per V e per le tasse)
                p = prezzo[j] - PREZZO_CARICO0[j]
                if p < zero:
                    p = zero
                plus0[j] = p / prezzo[j]
                V[j] = db / (flt(1.0) - aliquote0[j] * plus0[j])
            # queste le tratto dopo (vendita dedicata)
            V[I_6040S] = zero
            V[I_OB10] = zero
            V[I_OB5] = zero
            V[I_ACCOB10] = zero
            for j in range(NSTRAT):
                if V[j] > capitale[j]:
                    V[j] = capitale[j]
                buffer[j] += V[j]
                capitale[j] -= V[j]

            # 60/40 strategico: attingo dall'azionario se ha reso, altrimenti dall'obbligazionario
            V40 = zero
            V60 = zero
            if muore[I_6040S] == flt(-1.0):
                V[I_6040S] = zero
                if re_etf[m, s] > zero:   # vendo etf azionari
                    V40 = zero
                    pa = prezzo[I_ACC] - PREZZO_CARICO0[I_6040S]
                    if pa < zero:
                        pa = zero
                    V60 = differenza_buffer[I_6040S] / (flt(1.0) - aliquote0[I_6040S] * pa / prezzo[I_ACC])
                    if V60 > capitale60:
                        resto = V60 - capitale60
                        V40 = capitale40 if resto > capitale40 else resto
                        V60 = capitale60
                        buffer[I_6040S] += V40
                        capitale40 -= V40
                    buffer[I_6040S] += V60
                    capitale60 -= V60
                else:                     # vendo etf bond
                    V60 = zero
                    po = prezzo[I_6040S] - PREZZO_CARICO1[I_6040S]
                    if po < zero:
                        po = zero
                    V40 = differenza_buffer[I_6040S] / (flt(1.0) - aliquote1[I_6040S] * po / prezzo[I_6040S])
                    if V40 > capitale40:
                        resto = V40 - capitale40
                        V60 = capitale60 if resto > capitale60 else resto
                        V40 = capitale40
                        buffer[I_6040S] += V60
                        capitale60 -= V60
                    buffer[I_6040S] += V40
                    capitale40 -= V40
                capitale[I_6040S] = capitale40 + capitale60

            # rollover annuale dei BTP (ultimo mese dell'anno)
            if m % 12 == 11:
                buffer[I_OB10] += capitale10BTP[0]
                capitale[I_OB10] -= capitale10BTP[0]
                buffer[I_ACCOB10] += capitale10BTPacc[0]
                capitale[I_ACCOB10] -= capitale10BTPacc[0]
                for t in range(1, 10):
                    capitale10BTP[t - 1] = capitale10BTP[t]
                    capitale10BTPacc[t - 1] = capitale10BTPacc[t]
                    rendimenti10BTP[t - 1] = rendimenti10BTP[t]
                capitale10BTP[9] = zero
                capitale10BTPacc[9] = zero
                nb10 = re_btp[9, m, s]   # BTP10
                rendimenti10BTP[9] = nb10 if nb10 > zero else zero
                buffer[I_OB5] += capitale5BTP[0]
                capitale[I_OB5] -= capitale5BTP[0]
                for t in range(1, 5):
                    capitale5BTP[t - 1] = capitale5BTP[t]
                    rendimenti5BTP[t - 1] = rendimenti5BTP[t]
                capitale5BTP[4] = zero
                nb5 = re_btp[4, m, s]    # BTP5
                rendimenti5BTP[4] = nb5 if nb5 > zero else zero

            # buffer sotto zero -> vendo l'obbligazione positiva piu' recente (sempre a 100)
            if buffer[I_OB10] < zero:
                for t in range(10):
                    need = -buffer[I_OB10]
                    rec = capitale10BTP[t] if capitale10BTP[t] < need else need
                    buffer[I_OB10] += rec
                    capitale10BTP[t] -= rec
                    capitale[I_OB10] -= rec
                    if buffer[I_OB10] >= zero:
                        break
            if buffer[I_OB5] < zero:
                for t in range(5):
                    need = -buffer[I_OB5]
                    rec = capitale5BTP[t] if capitale5BTP[t] < need else need
                    buffer[I_OB5] += rec
                    capitale5BTP[t] -= rec
                    capitale[I_OB5] -= rec
                    if buffer[I_OB5] >= zero:
                        break

            soglia = buffer_ideale[I_ACCOB10] / flt(10.0)
            if buffer[I_ACCOB10] < soglia:
                for t in range(10):
                    need = soglia - buffer[I_ACCOB10]
                    rec = capitale10BTPacc[t] if capitale10BTPacc[t] < need else need
                    buffer[I_ACCOB10] += rec
                    capitale10BTPacc[t] -= rec
                    capitale[I_ACCOB10] -= rec
                    if buffer[I_ACCOB10] >= soglia:
                        break
            if buffer[I_ACCOB10] < soglia:   # se serve, vendo azionario
                need = soglia - buffer[I_ACCOB10]
                rec = capitale10accumulo if capitale10accumulo < need else need
                capitale10accumulo -= rec
                V[I_ACCOB10] += rec   # cosi' poi ci pago le tasse
                buffer[I_ACCOB10] += rec
                capitale[I_ACCOB10] -= rec

            # ********* TASSE su V, V40 e V60 ***************
            for j in range(NSTRAT):
                buffer[j] -= V[j] * aliquote0[j] * plus0[j]
            pa = prezzo[I_ACC] - PREZZO_CARICO0[I_6040S]
            if pa < zero:
                pa = zero
            buffer[I_6040S] -= V60 * aliquote0[I_6040S] * pa / prezzo[I_ACC]
            po = prezzo[I_6040S] - PREZZO_CARICO1[I_6040S]
            if po < zero:
                po = zero
            buffer[I_6040S] -= V40 * aliquote1[I_6040S] * po / prezzo[I_6040S]

            # ricostruisco il capitale delle strategie composite
            tot10 = zero
            tot10acc = zero
            for t in range(10):
                tot10 += capitale10BTP[t]
                tot10acc += capitale10BTPacc[t]
            tot5 = zero
            for t in range(5):
                tot5 += capitale5BTP[t]
            capitale[I_OB10] = tot10
            capitale[I_OB5] = tot5
            capitale[I_ACCOB10] = capitale10accumulo + tot10acc
            capitale[I_6040S] = capitale40 + capitale60

            # morte: se capitale+buffer <= 0 azzero e segno il mese
            vivo_tot = zero
            for j in range(NSTRAT):
                cb = capitale[j] + buffer[j]
                if cb <= zero:
                    capitale[j] = zero
                    buffer[j] = zero
                    if muore[j] == flt(-1.0):
                        muore[j] = flt(m)
                vivo_tot += capitale[j] + buffer[j]
            # azzero i sotto-conti delle strategie composite appena morte (FIX bug 4)
            if muore[I_6040S] >= zero:
                capitale60 = zero
                capitale40 = zero
            if muore[I_OB10] >= zero:
                for t in range(10):
                    capitale10BTP[t] = zero
            if muore[I_OB5] >= zero:
                for t in range(5):
                    capitale5BTP[t] = zero
            if muore[I_ACCOB10] >= zero:
                for t in range(10):
                    capitale10BTPacc[t] = zero
                capitale10accumulo = zero
            if vivo_tot <= zero:
                break

        for j in range(NSTRAT):
            out_ris[s, j] = capitale[j] + buffer[j]   # arrotondamento fatto sull'host (np.round)
            out_sop[s, j] = flt(n_mesi) if muore[j] == flt(-1.0) else muore[j]

    _KERNELS[key] = kernel
    return kernel


def simula_cuda(rendimenti_estratti, prelievi_estratti, nic_grezza_estratta,
                CAPITALE, BUFFER, BUFFER_IDEALE, PREZZO, PREZZO_CARICO0, PREZZO_CARICO1,
                aliquote0, aliquote1, CAPITALE10BTP, CAPITALE5BTP, CAPITALE10BTPacc,
                CAPITALE10accumulo, CAPITALE40, CAPITALE60, dividendo, aliquota_buffer,
                strategie, etf="MSCI WORLD", dtype=np.float32, threads_per_block=128):
    """Esegue tutte le simulazioni sulla GPU (un thread per simulazione).

    Gli argomenti sono le stesse strutture/costanti gia' presenti nel notebook.
    Ritorna (risultati, sopravvive) come DataFrame con colonne = strategie,
    nello stesso formato del loop CPU.
    """
    flt = float32 if np.dtype(dtype) == np.float32 else float64
    kernel = _get_kernel(flt)

    n_mesi, n_sim = prelievi_estratti.shape

    def dev(a):
        return cuda.to_device(np.ascontiguousarray(a, dtype=dtype))

    re_etf = dev(rendimenti_estratti[etf])
    re_6040 = dev(rendimenti_estratti["60/40"])
    re_ezb = dev(rendimenti_estratti["EUROZONE BOND"])
    nic = dev(nic_grezza_estratta)
    re_btp = dev(np.stack([rendimenti_estratti["BTP" + str(t)] for t in range(1, 11)]))  # (10, mesi, sim)
    prelievi = dev(prelievi_estratti)

    d_CAP = dev(CAPITALE)
    d_BUF = dev(BUFFER)
    d_BUFID = dev(BUFFER_IDEALE)
    d_PRZ = dev(PREZZO)
    d_PC0 = dev(PREZZO_CARICO0)
    d_PC1 = dev(PREZZO_CARICO1)
    d_AL0 = dev(aliquote0)
    d_AL1 = dev(aliquote1)
    d_C10 = dev(CAPITALE10BTP)
    d_C5 = dev(CAPITALE5BTP)
    d_C10acc = dev(CAPITALE10BTPacc)

    out_ris = cuda.device_array((n_sim, NSTRAT), dtype=dtype)
    out_sop = cuda.device_array((n_sim, NSTRAT), dtype=dtype)

    blocks = (n_sim + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](
        re_etf, re_6040, re_ezb, nic, re_btp, prelievi,
        d_CAP, d_BUF, d_BUFID, d_PRZ, d_PC0, d_PC1,
        d_AL0, d_AL1, d_C10, d_C5, d_C10acc,
        dtype(CAPITALE10accumulo), dtype(CAPITALE40), dtype(CAPITALE60),
        dtype(dividendo), dtype(aliquota_buffer),
        n_mesi, out_ris, out_sop)
    cuda.synchronize()

    ris = np.round(out_ris.copy_to_host().astype(np.float64), 0)
    sop = out_sop.copy_to_host().astype(np.float64)
    risultati = pd.DataFrame(ris, columns=strategie)
    sopravvive = pd.DataFrame(sop, columns=strategie)
    return risultati, sopravvive
