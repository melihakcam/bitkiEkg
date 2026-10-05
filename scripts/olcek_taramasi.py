"""Zaman ölçeği taraması: hangi bitki pencere süresinde EKG ön eğitimi işe yarıyor?

HuBERT-ECG donmuş gömmeleri + lojistik regresyon (LOPO, yazar, rastgele bölme), ön eğitimli ve
rastgele başlatılmış model için. Her pencere süresi modelin "5 saniyesine" sıkıştırılır;
pencere uzadıkça bitkideki yavaş değişimler EKG'nin öğrendiği frekans bandına kayar.

Ölçekler: 5min, 30min, 1h, 6h · eşlemeler: tekrar, parca (bkz. src/bitki_ekg/ekg_girdi.py).
5min'de 20 736 ikili pencere var; süre için bitki ve sınıf başına her 6. pencere alınır
(3 456, 30min ile aynı sayı). Özellik: tüm katmanların zaman ortalamalarının ortalaması.
Gömmeler data/processed altında önbelleklenir; tablo her konfigürasyondan sonra güncellenir.

Kullanım: python scripts/olcek_taramasi.py [--pencere 5min 30min 1h 6h] [--esleme tekrar parca]
Çıktı: results/tables/olcek_taramasi_katman.csv, olcek_taramasi_ozet.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hubert_sonda import gomme_cikar, model_yukle  # noqa: E402

from bitki_ekg.config import get_path, set_seed  # noqa: E402
from bitki_ekg.ekg_girdi import hubert_girdisi  # noqa: E402
from bitki_ekg.preprocessing import domates_pencereler, ikili_gorev, robust_z  # noqa: E402
from bitki_ekg.sonda import sonda_degerlendir  # noqa: E402

set_seed()
SEYRELT = {"5min": 6}


def veri(pencere: str):
    """İkili görev pencereleri + eksiksiz maskesi (gömme önbelleği tüm ikili pencereler için tutulur)."""
    X, meta = domates_pencereler(pencere)
    m = ikili_gorev(meta)
    tam = ikili_gorev(meta, X)[m]
    X, meta = X[m], meta[m].reset_index(drop=True)
    meta["tam"] = tam
    k = SEYRELT.get(pencere, 1)
    if k > 1:  # bitki ve sınıf içinde zaman sırasıyla her k. pencere
        meta = meta.sort_values(["plant_id", "class", "datetime_start"])
        sec = meta.groupby(["plant_id", "class"]).cumcount() % k == 0
        idx = meta.index[sec].to_numpy()
        X, meta = X[idx], meta.loc[idx].reset_index(drop=True)
    return X, meta["class"].to_numpy(int), meta["plant_id"].to_numpy(int), meta["tam"].to_numpy(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", nargs="+", default=["6h", "1h", "30min", "5min"])
    ap.add_argument("--esleme", nargs="+", default=["tekrar", "parca"])
    args = ap.parse_args()

    tablolar = get_path("tables")
    kayit = tablolar / "olcek_taramasi_katman.csv"
    eski = pd.read_csv(kayit) if kayit.exists() else pd.DataFrame()
    bitmis = set(map(tuple, eski[["pencere", "esleme", "model"]].drop_duplicates().to_numpy())) if len(eski) else set()

    for pencere in args.pencere:
        X, y, g, tam = veri(pencere)
        print(f"[{pencere}] {len(y)} pencere ({(~tam).sum()} eksik pencere değerlendirmeden çıkarılır), "
              f"uzunluk {X.shape[1]}, sınıflar {np.bincount(y)}", flush=True)
        for esleme in args.esleme:
            girdi = None
            for ad, onceden in (("onceden_egitilmis", True), ("rastgele_baslatilmis", False)):
                if (pencere, esleme, ad) in bitmis:
                    print(f"  {esleme} {ad}: önceden tamamlanmış, atlandı", flush=True)
                    continue
                dosya = get_path("processed") / f"hubert_gomme_{pencere}_{esleme}_{ad}.npz"
                t0 = time.time()
                if dosya.exists() and len(np.load(dosya)["E"]) == len(y):
                    E = np.load(dosya)["E"]
                else:
                    if girdi is None:
                        girdi = hubert_girdisi(robust_z(X), esleme)
                    E = gomme_cikar(model_yukle(onceden), girdi)
                    np.savez_compressed(dosya, E=E)
                df = sonda_degerlendir(E.mean(axis=1)[tam], y[tam], g[tam]).assign(pencere=pencere, esleme=esleme,
                                                                                    model=ad)
                df.to_csv(kayit, mode="a", header=not kayit.exists(), index=False)
                lopo = df[df.bolme == "lopo"]
                print(f"  {esleme} {ad}: LOPO doğruluk {lopo.dogruluk.mean():.3f}, AUC {lopo.auc.mean():.3f} "
                      f"({time.time() - t0:.0f} s)", flush=True)

    df = pd.read_csv(kayit)
    ozet = (df[df.bolme == "lopo"].groupby(["pencere", "esleme", "model"])[["dogruluk", "auc"]].mean() * 100).round(1)
    ozet = ozet.unstack("model")
    for olcut in ("dogruluk", "auc"):
        ozet[(olcut, "fark")] = ozet[(olcut, "onceden_egitilmis")] - ozet[(olcut, "rastgele_baslatilmis")]
    ozet.to_csv(tablolar / "olcek_taramasi_ozet.csv")
    pd.set_option("display.width", 200)
    print(ozet.sort_index(axis=1).to_string())


if __name__ == "__main__":
    main()
