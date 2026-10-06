"""Sarmaşık verisinde değerlendirme yöntemine bağlı şişme (ikinci veri seti, dış ortam).

Görevler (Buss vd., 2025 eşikleri): gündüz/gece, yağmurlu/kuru (çok dengesiz, ~1:10).
Girdi (v2): 1 saatlik, 2 kanallı (CH1, CH2) pencereler; **bitki ve kanal başına** z-skor
(bitkinin tüm kaydının ortalama/std'si, etiket kullanılmaz) → pencerenin seviyesi korunur.
v1'de pencere başına ölçekleme seviye bilgisini siliyordu (gündüz/gece farkı büyük ölçüde
seviyede); v1 sonuçları `sarmasik_sizinti_v1_pencere_z_*.csv`. Ardından 10 sn'lik ortalamalarla
360 noktaya seyreltme. Her bölmenin sonucu hemen CSV'ye eklenir; biten bölmeler atlanır.
Modeller: özet öznitelik (kanal başına ortalama, std, min, maks, eğim, fark std'si, çeyrekler)
+ LightGBM (yazarların öznitelik yaklaşımına yakın, hızlı) ve MiniRocket (çok değişkenli).
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
from lightgbm import LGBMClassifier

from bitki_ekg.config import get_path, set_seed
from bitki_ekg.preprocessing import sarmasik_pencereler

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


def bitki_z(X: np.ndarray, bitki: np.ndarray) -> np.ndarray:
    """Bitki ve kanal başına z-skor (bitkinin tüm penceleri üzerinden; etiketsiz)."""
    Z = np.empty_like(X, dtype=np.float32)
    for b in np.unique(bitki):
        m = bitki == b
        ort = X[m].mean(axis=(0, 2), keepdims=True)
        std = X[m].std(axis=(0, 2), keepdims=True)
        Z[m] = (X[m] - ort) / np.maximum(std, 1e-6)
    return Z


def ozet_oznitelik(Z: np.ndarray) -> np.ndarray:
    """Kanal başına özet öznitelikler: (n, 2, L) → (n, 2 × 9)."""
    t = np.arange(Z.shape[-1]) - Z.shape[-1] / 2
    egim = (Z * t).sum(-1) / (t ** 2).sum()
    q1, q3 = np.percentile(Z, [25, 75], axis=-1)
    oz = [Z.mean(-1), Z.std(-1), Z.min(-1), Z.max(-1), egim, np.diff(Z, axis=-1).std(-1), q1, q3,
          np.abs(np.diff(Z, axis=-1)).mean(-1)]
    return np.concatenate(oz, axis=1)


def main():
    X, meta = sarmasik_pencereler()
    Z = bitki_z(X, meta.bitki.to_numpy())
    Z = Z.reshape(*Z.shape[:2], -1, SEYREKLIK).mean(axis=-1)  # (n, 2, 360)
    print(f"{len(Z)} pencere, şekil {Z.shape}, bitkiler {meta.bitki.value_counts().to_dict()}", flush=True)
    girdi = {"Öznitelik + LightGBM": ozet_oznitelik(Z), "MiniRocket": Z}
    modeller = {"Öznitelik + LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                               random_state=SEED, verbose=-1),
                "MiniRocket": lambda: MiniRocketClassifier(random_state=SEED, n_jobs=-1)}

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
