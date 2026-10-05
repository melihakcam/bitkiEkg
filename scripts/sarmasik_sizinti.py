"""Sarmaşık verisinde değerlendirme yöntemine bağlı şişme (ikinci veri seti, dış ortam).

Görevler (Buss vd., 2025 eşikleri): gündüz/gece, yağmurlu/kuru (çok dengesiz, ~1:10).
Girdi: 1 saatlik, 2 kanallı (CH1, CH2) pencereler, kanal başına robust z-skor, ardından
10 sn'lik ortalamalarla 360 noktaya seyreltme (hesap süresi için; ilgili değişimler dakika–saat
ölçeğinde). Her bölmenin sonucu hemen CSV'ye eklenir; yeniden çalıştırınca biten bölmeler atlanır.
Modeller: MiniRocket (çok değişkenli), kNN (uygun olmayan).
Bölmeler:
  rastgele — tabakalı 5 katlı pencere bölmesi (Buss vd. 2025 ile aynı türden)
  lopo     — bitki-dışarıda-bırak (4 bitki)
  zaman    — takvim haftalarına göre ardışık 5 blok (geleceği tahmin)
Ölçütler: makro F1, dengeli doğruluk, AUC.

Kullanım: python scripts/sarmasik_sizinti.py
"""

import time
import warnings

import numpy as np
import pandas as pd
from aeon.classification.convolution_based import MiniRocketClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from bitki_ekg.config import get_path, set_seed
from bitki_ekg.preprocessing import robust_z, sarmasik_pencereler

warnings.filterwarnings("ignore", category=UserWarning)
SEED = set_seed()
GOREVLER = ("gunduz", "yagmurlu")


def bolmeler(meta: pd.DataFrame, y: np.ndarray):
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, y)):
        yield "rastgele", f"katman {k}", tr, te
    bitki = meta.bitki.to_numpy()
    for b in np.unique(bitki):
        yield "lopo", f"bitki {b}", np.where(bitki != b)[0], np.where(bitki == b)[0]
    hafta = meta.baslangic.dt.to_period("W").astype(str).to_numpy()
    haftalar = np.array(sorted(np.unique(hafta)))
    for k, blok in enumerate(np.array_split(haftalar, 5)):
        te = np.isin(hafta, blok)
        yield "zaman", f"blok {k}", np.where(~te)[0], np.where(te)[0]


def skor(model, X):
    if isinstance(model, MiniRocketClassifier):
        return model.pipeline_.decision_function(X)
    return model.predict_proba(X)[:, 1]


SEYREKLIK = 10  # 10 sn ortalama → 3600 yerine 360 nokta


def main():
    X, meta = sarmasik_pencereler()
    Z = robust_z(X)
    Z = Z.reshape(*Z.shape[:2], -1, SEYREKLIK).mean(axis=-1)  # (n, 2, 360)
    print(f"{len(Z)} pencere, şekil {Z.shape}, bitkiler {meta.bitki.value_counts().to_dict()}", flush=True)
    girdi = {"MiniRocket": Z, "kNN": Z.reshape(len(Z), -1)}
    modeller = {"MiniRocket": lambda: MiniRocketClassifier(random_state=SEED, n_jobs=-1),
                "kNN": lambda: KNeighborsClassifier(n_neighbors=5)}

    tablolar = get_path("tables")
    kayit = tablolar / "sarmasik_sizinti_katman.csv"
    df = pd.read_csv(kayit) if kayit.exists() else pd.DataFrame()
    bitmis = set(map(tuple, df[["gorev", "model", "bolme", "katman"]].to_numpy())) if len(df) else set()
    if bitmis:
        print(f"önceden tamamlanmış {len(bitmis)} bölme atlanacak", flush=True)

    for gorev in GOREVLER:
        y = meta[gorev].to_numpy(int)
        print(f"[{gorev}] sınıflar {np.bincount(y)}", flush=True)
        for ad, kur in modeller.items():
            t0 = time.time()
            for bolme, katman, tr, te in bolmeler(meta, y):
                if (gorev, ad, bolme, katman) in bitmis:
                    continue
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                t1 = time.time()
                m = kur().fit(girdi[ad][tr], y[tr])
                p = m.predict(girdi[ad][te])
                satir = pd.DataFrame([{"gorev": gorev, "model": ad, "bolme": bolme, "katman": katman,
                                       "n_test": len(te), "makro_f1": f1_score(y[te], p, average="macro"),
                                       "dengeli_dogruluk": balanced_accuracy_score(y[te], p),
                                       "auc": roc_auc_score(y[te], skor(m, girdi[ad][te]))}])
                satir.to_csv(kayit, mode="a", header=not kayit.exists(), index=False)
                print(f"    {gorev} {ad} {bolme} {katman}: {time.time() - t1:.0f} s", flush=True)
            print(f"  {ad}: {time.time() - t0:.0f} s", flush=True)

    df = pd.read_csv(kayit)
    ozet = df.groupby(["gorev", "model", "bolme"])[["makro_f1", "dengeli_dogruluk", "auc"]].agg(["mean", "std"])
    (ozet * 100).round(1).to_csv(tablolar / "sarmasik_sizinti_ozet.csv")
    pd.set_option("display.width", 200)
    print((ozet * 100).round(1).to_string())


if __name__ == "__main__":
    main()
