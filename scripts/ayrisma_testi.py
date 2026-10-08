"""18 günlük ayrışma testi (ön kayıt: docs/on_kayit.md — yöntem sonuç görülmeden yazıldı).

Susuz bitkiler aynı günlerdeki kontrol bitkilerinden ayrışıyor mu? Eğitim etiketleri yalnızca P1 (14–21.06)
günlerinden: 200/100 mL cihazları = 1, kontrol = 0 (aşırı sulama eğitime girmez). Cihaz-dışarıda-bırak;
dışarıdaki cihazın tüm günleri skorlanır. Bitki başına Δ = skor(P1) − skor(P0, 04–08.06).
Ana istatistik: doz sırası (kontrol 0, 200 1, 100 2) ile Δ arasında Spearman r; p: 90 cihaz atanışı, her birinde
yeniden eğitim. Plasebo: P1 = 07–08.06, P0 = 04–05.06 (tümü tedavi öncesi).

Gömmeler: notebooks/07_colab_18gun.ipynb → data/processed/gomme_<kol>_18gun.npz (Drive'dan indirilir).

Kullanım: python scripts/ayrisma_testi.py [--model ekg rastgele_t1 ... lightgbm]
"""

import argparse
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bitki_ekg.config import PROJECT_ROOT, get_path
from bitki_ekg.preprocessing import domates_pencereler
from bitki_ekg.zaman_testleri import cihaz
from colab_veri_18gun import ILK_GUN, SON_GUN
from doz_etki import CIHAZ_GRUP, DOZ
from stres_dengeli import ekg_sonda, model
from temel_modeller import tsfresh_oznitelikleri

ANA = {"P0": ("2025-06-04", "2025-06-08"), "P1": ("2025-06-14", "2025-06-21")}
PLASEBO = {"P0": ("2025-06-04", "2025-06-05"), "P1": ("2025-06-07", "2025-06-08")}
KOLLAR = ["ekg"] + [f"rastgele_t{t}" for t in range(1, 6)] + ["lightgbm"]


def pencereler() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(PROJECT_ROOT / "data" / "colab" / "domates_18gun_6h_500.npz")
    return z["bitki"].astype(int), z["gun"], z["saat"]


def oznitelik(kol: str, bitki, gun, saat):
    if kol == "lightgbm":
        X, meta = domates_pencereler("6h")
        g = meta.day.astype(str)
        m = ((g >= ILK_GUN) & (g <= SON_GUN)).to_numpy() & (np.isnan(X).mean(axis=1) <= 0.01)
        assert (meta.plant_id.to_numpy()[m] == bitki).all() and (g[m].to_numpy() == gun).all()
        return tsfresh_oznitelikleri("6h", meta)[m], model
    z = np.load(get_path("processed") / f"gomme_{kol}_18gun.npz")
    assert (z["bitki"] == bitki).all() and (z["gun"] == gun).all() and (z["saat"] == saat).all(), "sıra uyuşmuyor"
    return z["E"], ekg_sonda


def donem(gun, aralik):
    return (gun >= aralik[0]) & (gun <= aralik[1])


def skorlar(F, bitki, gun, atama, model_fn, d) -> np.ndarray:
    """Cihaz-dışarıda-bırak skorları (doz cihazları); aşırı sulama: 6 doz cihazıyla eğitilen model."""
    cih = cihaz(bitki)
    grup = np.array([atama[c] for c in cih])
    egitim_donem = donem(gun, d["P1"]) & np.isin(grup, list(DOZ))
    y = (grup != "kontrol").astype(int)
    skor = np.full(len(bitki), np.nan)
    for c in [c for c, g in atama.items() if g in DOZ]:
        tr = egitim_donem & (cih != c)
        skor[cih == c] = model_fn().fit(F[tr], y[tr]).predict_proba(F[cih == c])[:, 1]
    asiri = grup == "asiri"
    skor[asiri] = model_fn().fit(F[egitim_donem], y[egitim_donem]).predict_proba(F[asiri])[:, 1]
    return skor


def delta_r(skor, bitki, gun, atama, d) -> tuple[float, pd.DataFrame]:
    satir = []
    for b in np.unique(bitki):
        m = bitki == b
        satir.append({"bitki": int(b), "grup": atama[int(cihaz(np.array([b]))[0])],
                      "delta": skor[m & donem(gun, d["P1"])].mean() - skor[m & donem(gun, d["P0"])].mean()})
    t = pd.DataFrame(satir)
    k = t[t.grup.isin(DOZ)]
    return spearmanr(k.grup.map(DOZ), k.delta).statistic, t


def test(F, bitki, gun, model_fn, d) -> tuple[dict, pd.DataFrame, np.ndarray]:
    skor = skorlar(F, bitki, gun, CIHAZ_GRUP, model_fn, d)
    r, tablo = delta_r(skor, bitki, gun, CIHAZ_GRUP, d)
    cihazlar = [c for c, g in CIHAZ_GRUP.items() if g in DOZ]
    sifir = []
    for p in sorted(set(permutations(["kontrol", "kontrol", "200", "200", "100", "100"]))):
        atama = {**CIHAZ_GRUP, **dict(zip(cihazlar, p))}
        if atama != CIHAZ_GRUP:
            sifir.append(delta_r(skorlar(F, bitki, gun, atama, model_fn, d), bitki, gun, atama, d)[0])
    sifir = np.array(sifir)
    assert len(sifir) == 89
    ozet = {"r": r, "p": (1 + (sifir >= r).sum()) / (1 + len(sifir)),
            **{f"delta_{g}": tablo[tablo.grup == g].delta.mean() for g in ["kontrol", "200", "100", "asiri"]}}
    return ozet, tablo, skor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=KOLLAR)
    kollar = ap.parse_args().model
    bitki, gun, saat = pencereler()
    ozetler, tablolar, egriler = [], [], []
    for kol in kollar:
        F, model_fn = oznitelik(kol, bitki, gun, saat)
        for ad, d in [("ana", ANA), ("plasebo", PLASEBO)]:
            ozet, tablo, skor = test(F, bitki, gun, model_fn, d)
            ozetler.append({"model": kol, "test": ad, **ozet})
            tablolar.append(tablo.assign(model=kol, test=ad))
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in ozetler[-1].items()}, flush=True)
            if ad == "ana":
                grup = np.array([CIHAZ_GRUP[c] for c in cihaz(bitki)])
                e = pd.DataFrame({"gun": gun, "grup": grup, "skor": skor}).groupby(["gun", "grup"]).skor.mean().unstack()
                egriler.append(e.assign(model=kol))
    ek = "" if kollar == KOLLAR else "_" + "_".join(kollar)
    pd.DataFrame(ozetler).to_csv(get_path("tables") / f"ayrisma_ozet{ek}.csv", index=False)
    pd.concat(tablolar).to_csv(get_path("tables") / f"ayrisma_bitki{ek}.csv", index=False)
    pd.concat(egriler).to_csv(get_path("tables") / f"ayrisma_gunluk{ek}.csv")


if __name__ == "__main__":
    main()
