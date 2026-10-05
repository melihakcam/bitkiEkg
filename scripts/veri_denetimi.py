"""Domates verisi kalite denetimi (1 sa pencereler, 16 bitki × 18 gün, 1 Hz, mV).

Pencere (bitki × saat) başına:
  eksik_orani   : NaN örnek oranı (kısa/eksik dosya)
  duz_orani     : ardışık örnek farkının |Δ| < 1e-6 mV olduğu oran (donmuş/düz sinyal)
  sicrama       : |Δ| > 10 × (bitkinin medyan mutlak farkı) olan örnek sayısı
  std_mV, aralik_mV
Ayrıca: dosya sürekliliği (bitki başına 432 saat, eksik saat), 6 sa pencerelerdeki eksiklerin
yeri ve bitki bazında özet. Uyarı eşikleri raporda.

Kullanım: python scripts/veri_denetimi.py
Çıktı: results/tables/veri_denetimi_{pencere,bitki}.csv, results/figures/veri_denetimi.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bitki_ekg.config import get_path
from bitki_ekg.preprocessing import KONTROL_BITKILERI, domates_pencereler


def main():
    X, meta = domates_pencereler("1h")
    fark = np.diff(X, axis=1)
    pencere = meta[["plant_id", "node", "kanal", "datetime_start", "day", "class"]].copy()
    pencere["eksik_orani"] = np.isnan(X).mean(axis=1)
    pencere["duz_orani"] = (np.abs(fark) < 1e-6).mean(axis=1)
    pencere["std_mV"] = np.nanstd(X, axis=1)
    pencere["aralik_mV"] = np.nanmax(X, axis=1) - np.nanmin(X, axis=1)
    mad = {p: np.nanmedian(np.abs(fark[(meta.plant_id == p).to_numpy()])) for p in meta.plant_id.unique()}
    esik = pencere.plant_id.map(lambda p: 10 * max(mad[p], 1e-3)).to_numpy()
    pencere["sicrama"] = (np.abs(fark) > esik[:, None]).sum(axis=1)

    # Süreklilik: her bitki için beklenen saat ızgarası
    bekl = pd.date_range("2025-06-04 00:00", "2025-06-21 23:00", freq="1h")
    sureklilik = {p: len(set(bekl) - set(g.datetime_start)) for p, g in pencere.groupby("plant_id")}

    bitki = pencere.groupby("plant_id").agg(
        node=("node", "first"), kanal=("kanal", "first"), saat=("datetime_start", "size"),
        eksik_ort=("eksik_orani", "mean"), duz_ort=("duz_orani", "mean"),
        duz_yuksek_saat=("duz_orani", lambda s: int((s > 0.5).sum())),
        sicrama_toplam=("sicrama", "sum"), std_medyan=("std_mV", "median"), std_maks=("std_mV", "max"),
        aralik_maks=("aralik_mV", "max"))
    bitki["eksik_saat"] = pd.Series(sureklilik)
    bitki["grup"] = np.where(bitki.index.isin(KONTROL_BITKILERI), "kontrol", "sulama")

    # 6 sa: eksik örneklerin yeri
    X6, m6 = domates_pencereler("6h")
    m6 = m6.assign(eksik_orani=np.isnan(X6).mean(axis=1))
    eksik6 = m6[m6.eksik_orani > 0][["plant_id", "datetime_start", "class", "eksik_orani"]]

    tablolar = get_path("tables")
    pencere.to_csv(tablolar / "veri_denetimi_pencere.csv", index=False)
    bitki.round(4).to_csv(tablolar / "veri_denetimi_bitki.csv")
    eksik6.to_csv(tablolar / "veri_denetimi_6h_eksik.csv", index=False)

    # Şekil: bitki × gün ısı haritaları (düz oranı, sıçrama, std)
    gun = pencere.assign(gun=pencere.datetime_start.dt.strftime("%m-%d"))
    fig, axs = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, (s, ad, cmap) in zip(axs, [("duz_orani", "düz sinyal oranı", "Reds"),
                                        ("sicrama", "sıçrama sayısı (saatlik ort.)", "Oranges"),
                                        ("std_mV", "gün içi std [mV] (log)", "viridis")]):
        t = gun.pivot_table(index="plant_id", columns="gun", values=s, aggfunc="mean")
        v = np.log10(t + 1e-3) if s == "std_mV" else t
        im = ax.imshow(v, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(t.index)), [f"{p} {bitki.node[p]}-{bitki.kanal[p]}" for p in t.index], fontsize=7)
        ax.set_title(ad, loc="left", fontsize=10)
        plt.colorbar(im, ax=ax)
    axs[-1].set_xticks(range(len(t.columns)), t.columns, rotation=90, fontsize=7)
    fig.suptitle("Domates veri kalite denetimi (bitki × gün)")
    plt.tight_layout()
    plt.savefig(get_path("figures") / "veri_denetimi.png", dpi=110)

    pd.set_option("display.width", 220)
    print(bitki.round(3).to_string())
    print("\n6 sa eksik pencereler:", len(eksik6), "| sınıflar:", eksik6["class"].value_counts().to_dict())
    print(eksik6.sort_values("eksik_orani", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
