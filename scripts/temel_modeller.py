"""Aşama 5: domates verisinde temel modeller.

Modeller
  kNN (k=5)            — robust z-skorlu ham pencere (uygun olmayan)
  Naive Bayes          — robust z-skorlu ham pencere (uygun olmayan)
  tsfresh + LightGBM   — yazarların hesapladığı ~780 tsfresh özniteliği (klasik)
  MiniRocket           — robust z-skorlu ham pencere (derin/konvolüsyon tabanlı)

Bölmeler
  lopo     — bitki-dışarıda-bırak, 12 katman (asıl değerlendirme)
  yazar    — Buss vd. (2026) bölmesi: bitki 0 ve 12 test
  rastgele — tabakalı 5 katlı rastgele pencere bölmesi (literatürdeki iyimser yöntem)

Kullanım: python scripts/temel_modeller.py [--pencere 1h 30min]
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from aeon.classification.convolution_based import MiniRocketClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from bitki_ekg.config import get_path, set_seed
from bitki_ekg.data import domates_klasoru
from bitki_ekg.preprocessing import domates_pencereler, ikili_gorev, lopo_bolmeleri, robust_z, yazar_bolmesi

warnings.filterwarnings("ignore", category=UserWarning)
SEED = set_seed()


def tsfresh_oznitelikleri(pencere: str, meta: pd.DataFrame) -> np.ndarray:
    f = domates_klasoru() / "01_features" / "Exp1" / pencere / "features_with_class.csv"
    df = pd.read_csv(f, parse_dates=["datetime_start"])
    sutunlar = [c for c in df.columns if c.startswith("value__")]
    hizali = meta[["plant_id", "datetime_start"]].merge(df[["plant_id", "datetime_start", *sutunlar]],
                                                       on=["plant_id", "datetime_start"], how="left",
                                                       validate="one_to_one")
    F = hizali[sutunlar].to_numpy(np.float64)
    F[~np.isfinite(F)] = np.nan
    return F


def modeller():
    return {
        "kNN": (lambda: KNeighborsClassifier(n_neighbors=5), "ham"),
        "Naive Bayes": (lambda: GaussianNB(), "ham"),
        "tsfresh + LightGBM": (lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED,
                                                      verbose=-1), "oznitelik"),
        "MiniRocket": (lambda: MiniRocketClassifier(random_state=SEED, n_jobs=-1), "seri"),
    }


def skor(model, X):
    # MiniRocketClassifier içindeki ridge sınıflandırıcının predict_proba'sı 0/1 döndürür;
    # AUC için sürekli karar değeri gerekir.
    if isinstance(model, MiniRocketClassifier):
        return model.pipeline_.decision_function(X)  # dönüşüm + ölçekleme + ridge
    return model.predict_proba(X)[:, 1]


def degerlendir(y, tahmin, olasilik):
    return {"dogruluk": accuracy_score(y, tahmin), "f1": f1_score(y, tahmin),
            "auc": roc_auc_score(y, olasilik) if len(np.unique(y)) == 2 else np.nan}


def bolmeler(gruplar, y):
    for g, tr, te in lopo_bolmeleri(gruplar):
        yield "lopo", f"bitki {g}", tr, te
    tr, te = yazar_bolmesi(gruplar)
    yield "yazar", "bitki 0+12", tr, te
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, y)):
        yield "rastgele", f"katman {k}", tr, te


def calistir(pencere: str) -> pd.DataFrame:
    X, meta = domates_pencereler(pencere)
    m = ikili_gorev(meta)
    X, meta = X[m], meta[m].reset_index(drop=True)
    y = meta["class"].to_numpy(int)
    gruplar = meta["plant_id"].to_numpy(int)
    girdi = {"ham": robust_z(X), "oznitelik": tsfresh_oznitelikleri(pencere, meta)}
    girdi["seri"] = girdi["ham"][:, None, :]
    print(f"[{pencere}] {len(y)} pencere, {len(np.unique(gruplar))} bitki, uzunluk {X.shape[1]}")

    sonuc = []
    for ad, (kur, tur) in modeller().items():
        t0 = time.time()
        for bolme, katman, tr, te in bolmeler(gruplar, y):
            model = kur().fit(girdi[tur][tr], y[tr])
            p = model.predict(girdi[tur][te])
            sonuc.append({"pencere": pencere, "model": ad, "bolme": bolme, "katman": katman,
                          **degerlendir(y[te], p, skor(model, girdi[tur][te]))})
        print(f"  {ad}: {time.time() - t0:.0f} s")
    return pd.DataFrame(sonuc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", nargs="+", default=["1h", "30min"])
    args = ap.parse_args()

    katman = pd.concat([calistir(p) for p in args.pencere], ignore_index=True)
    tablolar = get_path("tables")
    katman.to_csv(tablolar / "temel_modeller_katman.csv", index=False)

    ozet = (katman.groupby(["pencere", "model", "bolme"])[["dogruluk", "f1", "auc"]]
            .agg(["mean", "std"]).round(3))
    ozet.to_csv(tablolar / "temel_modeller_ozet.csv")
    pd.set_option("display.width", 200)
    print(ozet.to_string())


if __name__ == "__main__":
    main()
