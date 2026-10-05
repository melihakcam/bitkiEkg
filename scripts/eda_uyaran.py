"""Uyaran veri seti (Buss vd., 2023) için keşifsel grafikler (results/figures/uyaran_*.png).

Pencere formatı (yazarların `getTrainTestData/ElectropotentialTestTrain.py` kodundan):
1. sütun etiket, ardından 512 örnek = uyaran öncesi 256 + uyaran sonrası 256 örnek.
Etiketler: 0 rüzgâr, 1 ısı, 2 uyaran yok, 3 mavi ışık, 4 kırmızı ışık.

Kullanım: python scripts/eda_uyaran.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bitki_ekg.config import get_path

KLASOR = get_path("raw") / "uyaran_siniflandirma" / "SupplementaryCode" / "SupplementaryCode" / "datasets"
SINIFLAR = {0: "rüzgâr", 1: "ısı", 2: "uyaran yok", 3: "mavi ışık", 4: "kırmızı ışık"}
RENK = {0: "#1565c0", 1: "#c62828", 2: "#616161", 3: "#3949ab", 4: "#d84315"}
SEKIL = get_path("figures")


def oku(ad: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(KLASOR / ad, delimiter="\t")
    return d[:, 0].astype(int), d[:, 1:]


def main():
    y_tr, X_tr = oku("train_ANN.tsv")
    y_te, X_te = oku("test_ANN.tsv")
    print(f"eğitim {X_tr.shape}, test {X_te.shape}, eksik değer: {np.isnan(X_tr).sum() + np.isnan(X_te).sum()}")

    # 1) Sınıf dağılımı
    say = pd.DataFrame({"eğitim": pd.Series(y_tr).value_counts(), "test": pd.Series(y_te).value_counts()}).sort_index()
    say.index = [SINIFLAR[i] for i in say.index]
    ax = say.plot.bar(figsize=(7, 3.8), color=["#455a64", "#90a4ae"], rot=0)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    ax.set_ylabel("pencere sayısı")
    ax.set_title("Uyaran veri seti: sınıf dağılımı (yazarların %70/%30 bölmesi)")
    plt.tight_layout()
    plt.savefig(SEKIL / "uyaran_sinif_dagilimi.png", dpi=120)
    plt.close()

    # 2) Sınıf ortalama tepkisi: uyaran öncesi son değere göre fark (ham sayım)
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    yari = X.shape[1] // 2
    fark = X - X[:, [yari - 1]]
    t = np.arange(X.shape[1]) - yari
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s, ad in SINIFLAR.items():
        m = np.median(fark[y == s], axis=0)
        q1, q3 = np.percentile(fark[y == s], [25, 75], axis=0)
        ax.plot(t, m, color=RENK[s], label=f"{ad} (n={np.sum(y == s)})")
        ax.fill_between(t, q1, q3, color=RENK[s], alpha=0.12)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.text(2, ax.get_ylim()[1] * 0.9, "uyaran başlangıcı", fontsize=8)
    ax.set_xlabel("uyaran başlangıcına göre örnek (≈1,7 s/örnek)")
    ax.set_ylabel("uyaran öncesine göre fark [ham sayım]")
    ax.set_title("Sınıf başına ortanca tepki (gölge: çeyrekler arası aralık)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(SEKIL / "uyaran_sinif_tepkisi.png", dpi=120)
    plt.close()

    # 3) Her sınıftan örnek pencereler
    fig, axs = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
    rng = np.random.default_rng(42)
    for ax, (s, ad) in zip(axs, SINIFLAR.items()):
        for i in rng.choice(np.where(y == s)[0], 5, replace=False):
            ax.plot(t, fark[i], color=RENK[s], lw=0.7, alpha=0.8)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(ad, fontsize=10)
    axs[0].set_ylabel("fark [ham sayım]")
    fig.suptitle("Her sınıftan rastgele 5 pencere")
    plt.tight_layout()
    plt.savefig(SEKIL / "uyaran_ornek_pencereler.png", dpi=120)
    plt.close()

    say.to_csv(get_path("tables") / "uyaran_sinif_dagilimi.csv")
    print(say.to_string())


if __name__ == "__main__":
    main()
