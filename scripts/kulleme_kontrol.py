"""Domates külleme verisi (Matić vd. 2025, Mendeley 10.17632/yr8zhsc6mh.1): bitki/kanal karıştırıcısı testi.

Soru: Sağlıklı ve hastalıklı (inokule) bitkiler, hastalığın henüz etki gösteremeyeceği ilk gün de
ayırt edilebiliyor mu? Ayrılabiliyorsa fark hastalıktan değil, bitkinin/elektrot kanalının kendisinden gelir.

Her deney (A, B, C) için her bitkinin (kablo = 2 hat) günlük ortalama potansiyeli hesaplanır;
sağlıklı–hastalıklı farkı gün gün raporlanır, bitki-dışarıda-bırak sınıflandırma (LOPO) ilk gün ile
son günlerde karşılaştırılır.

Kullanım: python scripts/kulleme_kontrol.py
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bitki_ekg.config import get_path

KLASOR = get_path("raw") / "domates_kulleme" / "DATASET_ElectricalSignaling-in-tomato-for-detect-P" / "ExcelProcessedData"
SAYFALAR = {"A": ("MisureElettriche_Prove(ABC)_STAT_MalBianco.xlsx", "Prova(A) Data"),
            "B": ("MisureElettriche_Prove(ABC)_STAT_MalBianco.xlsx", "Prova(B) Data"),
            "C": ("MisureElettriche_Prove(C)_MalBianco.xlsx", "Prova(C) Data")}


def oku(deney: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Bitki hatlarının meta verisi, zaman (gün) ve potansiyel matrisi (hat × zaman)."""
    dosya, sayfa = SAYFALAR[deney]
    d = pd.read_excel(KLASOR / dosya, sheet_name=sayfa, header=None)
    t = pd.to_numeric(d.iloc[0, 5:], errors="coerce").to_numpy(float)
    if np.nanmax(t) > 100:  # saniye cinsinden yazılmışsa
        t = t / 86400
    meta = d.iloc[2:, :5].copy()
    meta.columns = ["saglik", "su", "kontrol", "kablo", "hat"]
    meta["kablo"] = meta["kablo"].ffill().infer_objects()
    meta = meta[pd.to_numeric(meta.saglik, errors="coerce").notna()]  # bitkisiz kontrol ve ışık/sıcaklık satırlarını at
    X = d.loc[meta.index, d.columns[5:]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    meta = meta.assign(saglik=meta.saglik.astype(int), su=meta.su.astype(int), kablo=meta.kablo.astype(int))
    return meta.reset_index(drop=True), t, X


def bitki_gunluk(meta, t, X) -> pd.DataFrame:
    """Bitki (kablo) başına her gün için ortalama potansiyel (mV) ve oynaklık."""
    satir = []
    for g in range(int(np.nanmax(t)) + 1):
        m = (t >= g) & (t < g + 1)
        if m.sum() < 50:
            continue
        for k, mk in meta.groupby("kablo"):
            x = X[mk.index][:, m] * 1000
            satir.append({"gun": g, "kablo": k, "saglik": mk.saglik.iloc[0], "su": mk.su.iloc[0],
                          "ort": np.nanmean(x), "std": np.nanmean(np.nanstd(x, axis=1)),
                          "egim": np.nanmean(np.polyfit(np.arange(m.sum()), np.nan_to_num(x.T), 1)[0])})
    return pd.DataFrame(satir)


def lopo_auc(gun_df: pd.DataFrame) -> float:
    """Bitki-dışarıda-bırak: o günün öznitelikleriyle sağlıklı/hastalıklı ayrımı (AUC)."""
    gun_df = gun_df.dropna(subset=["ort", "std", "egim"])
    if gun_df.saglik.value_counts().min() < 2 or gun_df.saglik.nunique() < 2:
        return np.nan
    F = gun_df[["ort", "std", "egim", "su"]].to_numpy()
    y = 1 - gun_df.saglik.to_numpy()  # 1 = hastalıklı
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        p[i] = LogisticRegression(C=1.0).fit((F[tr] - mu) / sd, y[tr]).predict_proba(((F[i] - mu) / sd)[None])[0, 1]
    return roc_auc_score(y, p)


def main():
    tum = []
    for deney in SAYFALAR:
        meta, t, X = oku(deney)
        g = bitki_gunluk(meta, t, X).assign(deney=deney)
        tum.append(g)
        print(f"\n=== Deney {deney}: {meta.kablo.nunique()} bitki "
              f"(sağlıklı {meta[meta.saglik == 1].kablo.nunique()}, hastalıklı {meta[meta.saglik == 0].kablo.nunique()}), "
              f"{np.nanmax(t):.1f} gün")
        for gun, gd in g.groupby("gun"):
            s, h = gd[gd.saglik == 1].ort, gd[gd.saglik == 0].ort
            p = mannwhitneyu(s, h).pvalue if len(s) > 1 and len(h) > 1 else np.nan
            print(f"  gün {gun:2d}: sağlıklı {s.mean():8.1f} mV · hastalıklı {h.mean():8.1f} mV · p={p:.3f} · LOPO AUC={lopo_auc(gd):.2f}")
    sonuc = pd.concat(tum)
    sonuc.to_csv(get_path("tables") / "kulleme_bitki_gunluk.csv", index=False)


if __name__ == "__main__":
    main()
