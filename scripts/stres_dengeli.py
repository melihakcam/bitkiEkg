"""Zamana karşı dengelenmiş eğitim: model stresi mi öğreniyor?

Eğitimde kontrol bitkilerinin SON günleri de "sağlıklı" (0) etiketlenir; tek pozitif sınıf sulama
bitkilerinin son günleridir. Böylece "son gün = stresli" kısayolu cezalandırılır.

Değerlendirme cihaz-dışarıda-bırak (8 cihaz; kontrol bitkileri yalnızca PN8/PN9'da). Her test bitkisi için
  Δ = (son 3 gün ortalama skor) − (ilk 3 gün ortalama skor).
Gerçek stres sinyali varsa sulama bitkilerinde Δ büyük, kontrol bitkilerinde Δ ≈ 0 olmalı.
Şans testi:
  - bitki düzeyinde Mann–Whitney (12 sulama vs 4 kontrol; en küçük p = 1/1820)
  - cihaz düzeyinde kesin permütasyon: hangi 2 cihazın kontrol olduğu 28 biçimde değiştirilir,
    model her seferinde yeniden eğitilir (en küçük p = 1/28 ≈ 0,036)

Kullanım: python scripts/stres_dengeli.py [--pencere 6h 1h] | --ekg
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bitki_ekg.config import PROJECT_ROOT, get_path, set_seed
from bitki_ekg.preprocessing import KONTROL_BITKILERI, domates_pencereler
from bitki_ekg.zaman_testleri import cihaz
from temel_modeller import tsfresh_oznitelikleri
from zaman_kontrolu import ILK_GUNLER, SON_GUNLER

SEED = set_seed()


def model():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                          random_state=SEED, verbose=-1)


def ekg_sonda():
    """Donmuş EKG gömmeleri için doğrusal sonda (notebooks/05 ile aynı, sınıf ağırlıklı)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                                                              random_state=SEED))


def skorlar(F, son, bitki, kontrol_cihazlari, model_fn=model) -> np.ndarray:
    """Cihaz-dışarıda-bırak stres skorları (her pencere, kendi cihazı eğitimde yokken)."""
    cih = cihaz(bitki)
    y = (~np.isin(cih, kontrol_cihazlari) & (son == 1)).astype(int)
    skor = np.empty(len(y))
    for c in np.unique(cih):
        tr, te = cih != c, cih == c
        skor[te] = model_fn().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
    return skor


def deltalar(F, son, bitki, kontrol_cihazlari, model_fn=model) -> pd.DataFrame:
    sulama = ~np.isin(cihaz(bitki), kontrol_cihazlari)
    skor = skorlar(F, son, bitki, kontrol_cihazlari, model_fn)
    satir = []
    for b in np.unique(bitki):
        m = bitki == b
        satir.append({"bitki": int(b), "sulama": bool(sulama[m][0]),
                      "delta": skor[m & (son == 1)].mean() - skor[m & (son == 0)].mean(),
                      "auc_son_ilk": roc_auc_score(son[m], skor[m])})
    return pd.DataFrame(satir)


def calistir(pencere: str) -> tuple[pd.DataFrame, dict]:
    X_ham, meta = domates_pencereler(pencere)
    gun = meta.day.astype(str)
    secim = ((gun.isin(ILK_GUNLER) | gun.isin(SON_GUNLER)).to_numpy()
             & (np.isnan(X_ham).mean(axis=1) <= 0.01))
    F = tsfresh_oznitelikleri(pencere, meta)[secim]
    son = gun[secim].isin(SON_GUNLER).to_numpy(int)
    bitki = meta.plant_id.to_numpy()[secim]
    return degerlendir(F, son, bitki, pencere, model)


def calistir_ekg() -> tuple[pd.DataFrame, dict]:
    """HuBERT-ECG donmuş gömmeleri (6 sa), notebooks/05 adım 4 çıktısı (Drive'dan indirildi)."""
    g = np.load(get_path("processed") / "gomme_ekg_6h_asama1.npz")
    z = np.load(PROJECT_ROOT / "data" / "colab" / "domates_ikili_6h_500.npz")
    k = np.load(PROJECT_ROOT / "data" / "colab" / "domates_kontrol_6h_500.npz")
    F = np.vstack([g["E"], g["Ec"]])
    son = np.concatenate([z["y"], k["y"]]).astype(int)
    bitki = np.concatenate([z["bitki"], k["bitki"]]).astype(int)
    return degerlendir(F, son, bitki, "6h_ekg", ekg_sonda)


def degerlendir(F, son, bitki, pencere, model_fn) -> tuple[pd.DataFrame, dict]:
    secim = np.ones(len(son), dtype=bool)
    gercek = tuple(np.unique(cihaz(np.array(KONTROL_BITKILERI))))

    d = deltalar(F, son, bitki, gercek, model_fn)
    fark = d[d.sulama].delta.mean() - d[~d.sulama].delta.mean()
    p_mw = mannwhitneyu(d[d.sulama].delta, d[~d.sulama].delta, alternative="greater").pvalue

    sifir = []
    for kc in combinations(np.unique(cihaz(bitki)), 2):
        if tuple(sorted(kc)) == gercek:
            continue
        dp = deltalar(F, son, bitki, kc, model_fn)
        print(f"  [{pencere}] permütasyon {len(sifir) + 1}/27", flush=True)
        sifir.append(dp[dp.sulama].delta.mean() - dp[~dp.sulama].delta.mean())
    sifir = np.array(sifir)
    ozet = {"pencere": pencere, "n_pencere": int(secim.sum()),
            "delta_sulama": d[d.sulama].delta.mean(), "delta_kontrol": d[~d.sulama].delta.mean(),
            "fark": fark, "p_mw_bitki": p_mw,
            "p_perm_cihaz": (1 + (sifir >= fark).sum()) / (1 + len(sifir)),
            "sifir_fark_ort": sifir.mean(), "sifir_fark_maks": sifir.max(),
            "auc_son_ilk_sulama": d[d.sulama].auc_son_ilk.mean(),
            "auc_son_ilk_kontrol": d[~d.sulama].auc_son_ilk.mean()}
    print(d.round(3).to_string(index=False), flush=True)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ozet.items()}, flush=True)
    return d.assign(pencere=pencere), ozet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", nargs="+", default=["6h", "1h"])
    ap.add_argument("--ekg", action="store_true", help="yalnızca EKG gömmeleriyle (6 sa)")
    args = ap.parse_args()
    if args.ekg:
        d, o = calistir_ekg()
        d.to_csv(get_path("tables") / "stres_dengeli_ekg_bitki.csv", index=False)
        pd.DataFrame([o]).to_csv(get_path("tables") / "stres_dengeli_ekg_ozet.csv", index=False)
        return
    sonuc = [calistir(p) for p in args.pencere]
    pd.concat([s[0] for s in sonuc]).to_csv(get_path("tables") / "stres_dengeli_bitki.csv", index=False)
    pd.DataFrame([s[1] for s in sonuc]).to_csv(get_path("tables") / "stres_dengeli_ozet.csv", index=False)


if __name__ == "__main__":
    main()
