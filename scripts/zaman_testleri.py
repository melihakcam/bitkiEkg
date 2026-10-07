"""Zaman karıştırıcısı testleri (6 sa) — tsfresh + LightGBM. Ayrıntı: src/bitki_ekg/zaman_testleri.py.

EKG modeli için aynı testler Colab'da (notebooks/06_colab_zaman_testleri.ipynb) donmuş gömmelerle çalışır.

Kullanım: python scripts/zaman_testleri.py
"""

import numpy as np
from lightgbm import LGBMClassifier

from bitki_ekg.config import get_path, set_seed
from bitki_ekg.preprocessing import domates_pencereler
from bitki_ekg.zaman_testleri import tum_testler
from temel_modeller import tsfresh_oznitelikleri
from zaman_kontrolu import ILK_GUNLER, SON_GUNLER

SEED = set_seed()


def main():
    X_ham, meta = domates_pencereler("6h")
    gun = meta.day.astype(str)
    secim = ((gun.isin(ILK_GUNLER) | gun.isin(SON_GUNLER)).to_numpy()
             & (np.isnan(X_ham).mean(axis=1) <= 0.01))
    F = tsfresh_oznitelikleri("6h", meta)[secim]
    son = gun[secim].isin(SON_GUNLER).to_numpy(int)
    bitki = meta.plant_id.to_numpy()[secim]
    print(f"{secim.sum()} pencere, {len(np.unique(bitki))} bitki")

    def model():
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)

    bitki_df, ozet = tum_testler(F, son, bitki, model)
    bitki_df.to_csv(get_path("tables") / "zaman_testleri_lightgbm_bitki.csv", index=False)
    ozet.to_csv(get_path("tables") / "zaman_testleri_lightgbm_ozet.csv", index=False)
    print(bitki_df.round(3).to_string(index=False))
    print(ozet.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
