"""Doz-etki testi: zamana karşı dengelenmiş modelin skor artışı (Δ) sulama dozuyla artıyor mu?

Grup eşlemesi Buss vd.'nin Zenodo 22081982 kaydındaki `All_Plants_smoothed.png` başlıklarından
(docs/veri_notlari.md): aşırı sulama PN2, PN5 · kontrol 400 mL PN8, PN9 · 200 mL PN10, PN11 · 100 mL PN12, PN16.

Model eğitimde dozu görmez (tüm sulama bitkilerinin son günleri aynı "1" etiketi; scripts/stres_dengeli.py).
İstatistik: kontrol (0) < 200 mL (1) < 100 mL (2) sırası ile bitki Δ'sı arasındaki Spearman r.
Şans testi cihaz düzeyinde: 6 cihazın (kontrol, 200, 100) gruplara her atanışı (6!/(2!)^3 = 90) için
kontrol etiketleri değişeceğinden model YENİDEN eğitilir; aşırı sulama cihazları sabit kalır.
En küçük olası p = 1/90 ≈ 0,011.

Kullanım: python scripts/doz_etki.py [--model ekg lightgbm rastgele_t1 ... rastgele_t5] [--cikti ek]
"""

import argparse
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bitki_ekg.config import PROJECT_ROOT, get_path
from bitki_ekg.preprocessing import domates_pencereler
from bitki_ekg.zaman_testleri import cihaz
from stres_dengeli import deltalar, ekg_sonda, model
from temel_modeller import tsfresh_oznitelikleri
from zaman_kontrolu import ILK_GUNLER, SON_GUNLER

# cihaz kodu = bitki // 2 → 0 PN2, 1 PN5, 2 PN8, 3 PN9, 4 PN10, 5 PN11, 6 PN12, 7 PN16
CIHAZ_GRUP = {0: "asiri", 1: "asiri", 2: "kontrol", 3: "kontrol", 4: "200", 5: "200", 6: "100", 7: "100"}
DOZ = {"kontrol": 0, "200": 1, "100": 2}


def veri(ad):
    if ad.startswith(("ekg", "rastgele")):  # donmuş gömmeler: notebooks/05 adım 4 (Drive'dan indirildi)
        g = np.load(get_path("processed") / f"gomme_{ad}_6h_asama1.npz")
        z = np.load(PROJECT_ROOT / "data" / "colab" / "domates_ikili_6h_500.npz")
        k = np.load(PROJECT_ROOT / "data" / "colab" / "domates_kontrol_6h_500.npz")
        return (np.vstack([g["E"], g["Ec"]]), np.concatenate([z["y"], k["y"]]).astype(int),
                np.concatenate([z["bitki"], k["bitki"]]).astype(int), ekg_sonda)
    X_ham, meta = domates_pencereler("6h")
    gun = meta.day.astype(str)
    secim = ((gun.isin(ILK_GUNLER) | gun.isin(SON_GUNLER)).to_numpy() & (np.isnan(X_ham).mean(axis=1) <= 0.01))
    return (tsfresh_oznitelikleri("6h", meta)[secim], gun[secim].isin(SON_GUNLER).to_numpy(int),
            meta.plant_id.to_numpy()[secim], model)


def istatistik(F, son, bitki, model_fn, atama: dict) -> tuple[float, pd.DataFrame]:
    kontrol = tuple(sorted(c for c, g in atama.items() if g == "kontrol"))
    d = deltalar(F, son, bitki, kontrol, model_fn)
    d["grup"] = cihaz(d.bitki.to_numpy()).tolist()
    d["grup"] = d.grup.map(atama)
    k = d[d.grup.isin(DOZ)]
    return spearmanr(k.grup.map(DOZ), k.delta).statistic, d


def calistir(ad):
    F, son, bitki, model_fn = veri(ad)
    r, d = istatistik(F, son, bitki, model_fn, CIHAZ_GRUP)
    cihazlar = [c for c, g in CIHAZ_GRUP.items() if g in DOZ]
    gorulen, sifir = set(), []
    for p in permutations(["kontrol", "kontrol", "200", "200", "100", "100"]):
        if p in gorulen:
            continue
        gorulen.add(p)
        atama = {**CIHAZ_GRUP, **dict(zip(cihazlar, p))}
        if atama == CIHAZ_GRUP:
            continue
        sifir.append(istatistik(F, son, bitki, model_fn, atama)[0])
    sifir = np.array(sifir)
    ozet = {"model": ad, "spearman_r": r, "p_perm_cihaz": (1 + (sifir >= r).sum()) / (1 + len(sifir)),
            "n_perm": len(sifir) + 1, **{f"delta_{g}": d[d.grup == g].delta.mean() for g in ["kontrol", "asiri", "200", "100"]}}
    print(ozet, flush=True)
    return d.assign(model=ad), ozet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["ekg", "lightgbm"])
    ap.add_argument("--cikti", default="", help="çıktı dosya adı eki (ör. _rastgele)")
    args = ap.parse_args()
    sonuc = [calistir(m) for m in args.model]
    pd.concat([s[0] for s in sonuc]).to_csv(get_path("tables") / f"doz_etki{args.cikti}_bitki.csv", index=False)
    pd.DataFrame([s[1] for s in sonuc]).to_csv(get_path("tables") / f"doz_etki{args.cikti}_ozet.csv", index=False)


if __name__ == "__main__":
    main()
