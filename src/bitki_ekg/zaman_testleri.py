"""Zaman karıştırıcısı testleri: model stresi mi, zamanı/cihazı mı öğreniyor?

Etiket deneyin başı (ilk 3 gün = 0) ve sonuyla (son 3 gün = 1) çakışır; ayrıca 4 kontrol bitkisinin
hepsi iki cihazdadır (PN8, PN9; her cihazda 2 bitki). Testler:

  kontrol   — hiç strese girmeyen kontrol bitkilerinde ilk/son gün ayrımı (LOPO). ~0,5 olmalı.
  aktarim   — sulama bitkilerinde eğitilen stres modeli kontrol bitkilerine uygulanır. ~0,5 olmalı.
  ayni_gun  — SON 3 günde sulama grubu ile kontrol grubu ayrımı: iki sınıf aynı zamandan, zaman
              kullanılamaz. Cihaz-dışarıda-bırak (LODO) ile.
  ilk_gun   — aynı test İLK 3 günde (tedavi başlamadan, herkes 400 mL). ~0,5 olmalı; yüksekse
              ayrım stresten değil bitki/cihaz farkından gelir (plasebo testi).
  fark      — her pencereden o bitkinin ilk 3 gün ortalaması çıkarılır, sonra ayni_gun testi.
              Bitki/cihaz sabit farkları ve tüm bitkilerde ortak zaman kayması düşer.

Şans testi:
  kontrol/aktarim — bitki başına Mann–Whitney p (pencereler bağımsız sayıldığı için iyimser) ve
                    bitki düzeyinde işaret testi (temkinli: k/n bitkide AUC > 0,5).
  ayni_gun/ilk_gun/fark — cihaz düzeyinde kesin permütasyon testi: 8 cihazdan hangi 2'sinin "kontrol"
                    olduğu tüm C(8,2) = 28 biçimde değiştirilir. En küçük olası p = 1/28 ≈ 0,036.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

KONTROL = (4, 5, 6, 7)


def cihaz(bitki: np.ndarray) -> np.ndarray:
    """Her cihazda ardışık iki bitki: 0–1 PN2, 2–3 PN5, 4–5 PN8, 6–7 PN9, ... (veri_notlari.md)."""
    return np.asarray(bitki) // 2


def _mw_p(y, s) -> float:
    return float(mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").pvalue)


def _bitki_basina(y, s, gruplar, test_ad) -> list[dict]:
    satir = []
    for b in np.unique(gruplar):
        m = gruplar == b
        satir.append({"test": test_ad, "bitki": int(b), "n": int(m.sum()), "auc": roc_auc_score(y[m], s[m]),
                      "dogruluk": float(((s[m] >= 0.5) == y[m]).mean()), "p_mw": _mw_p(y[m], s[m])})
    return satir


def test_kontrol(F, son, bitki, model_fn) -> list[dict]:
    k = np.isin(bitki, KONTROL)
    X, y, g = F[k], son[k], bitki[k]
    s = np.empty(len(y))
    for b in np.unique(g):
        tr, te = g != b, g == b
        s[te] = model_fn().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return _bitki_basina(y, s, g, "kontrol")


def test_aktarim(F, son, bitki, model_fn) -> list[dict]:
    k = np.isin(bitki, KONTROL)
    s = model_fn().fit(F[~k], son[~k]).predict_proba(F[k])[:, 1]
    return _bitki_basina(son[k], s, bitki[k], "aktarim")


def _lodo_auc(X, y, cih, model_fn) -> tuple[float, float]:
    s = np.empty(len(y))
    for c in np.unique(cih):
        tr, te = cih != c, cih == c
        s[te] = model_fn().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, s), balanced_accuracy_score(y, s >= 0.5)


def test_grup(F, secim, bitki, model_fn, test_ad) -> dict:
    """Seçilen pencerelerde sulama (1) / kontrol (0) grubu ayrımı, LODO + cihaz düzeyinde permütasyon."""
    X, b = F[secim], bitki[secim]
    cih = cihaz(b)
    y = (~np.isin(b, KONTROL)).astype(int)
    auc, bacc = _lodo_auc(X, y, cih, model_fn)
    cihazlar = np.unique(cih)
    gercek = tuple(sorted(np.unique(cihaz(np.array(KONTROL)))))
    sifir = []
    for kontrol_cih in combinations(cihazlar, 2):
        if tuple(sorted(kontrol_cih)) == gercek:
            continue
        yp = (~np.isin(cih, kontrol_cih)).astype(int)
        sifir.append(_lodo_auc(X, yp, cih, model_fn)[0])
    sifir = np.array(sifir)
    return {"test": test_ad, "n": int(len(y)), "auc": auc, "dengeli_dogruluk": bacc,
            "p_perm": float((1 + (sifir >= auc).sum()) / (1 + len(sifir))),
            "sifir_auc_ort": float(sifir.mean()), "sifir_auc_maks": float(sifir.max())}


def ilk_gun_farki(F, son, bitki) -> np.ndarray:
    """Her pencereden kendi bitkisinin ilk 3 gün ortalamasını çıkarır."""
    D = F.copy()
    for b in np.unique(bitki):
        m = bitki == b
        D[m] = F[m] - np.nanmean(F[m & (son == 0)], axis=0)
    return D


def tum_testler(F, son, bitki, model_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F: 16 bitkinin ilk/son 3 gün pencereleri; son: 0 = ilk 3 gün, 1 = son 3 gün."""
    son, bitki = np.asarray(son).astype(int), np.asarray(bitki).astype(int)
    bitki_df = pd.DataFrame(test_kontrol(F, son, bitki, model_fn) + test_aktarim(F, son, bitki, model_fn))
    grup = [test_grup(F, son == 1, bitki, model_fn, "ayni_gun"),
            test_grup(F, son == 0, bitki, model_fn, "ilk_gun"),
            test_grup(ilk_gun_farki(F, son, bitki), son == 1, bitki, model_fn, "fark")]
    ozet = []
    for t, g in bitki_df.groupby("test"):
        ozet.append({"test": t, "n": int(g.n.sum()), "auc": g.auc.mean(), "dogruluk": g.dogruluk.mean(),
                     "bitki_auc_gt_05": f"{(g.auc > 0.5).sum()}/{len(g)}",
                     "p_isaret": binomtest(int((g.auc > 0.5).sum()), len(g), 0.5, alternative="greater").pvalue,
                     "p_mw_en_buyuk": g.p_mw.max()})
    return bitki_df, pd.concat([pd.DataFrame(ozet), pd.DataFrame(grup)], ignore_index=True)
