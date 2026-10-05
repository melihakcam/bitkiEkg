"""Domates verisi için ilk keşifsel grafikler (results/figures/domates_*.png).

Kullanım: python scripts/eda_domates.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bitki_ekg.config import get_path
from bitki_ekg.data import DOMATES_CIHAZLAR, DOMATES_KONTROL, domates_cihaz, domates_etiketler

SEKIL = get_path("figures")
SAGLIKLI = ("2025-06-04", "2025-06-07")  # ilk 3 gün (sınıf 0)
STRESLI = ("2025-06-19", "2025-06-22")   # son 3 gün (sınıf 1)
SULAMA_DEGISTI = "2025-06-08"            # gruplara ayrılma
YESIL, KIRMIZI, GRI = "#2e7d32", "#c62828", "#757575"


def donemleri_isaretle(ax):
    ax.axvspan(*SAGLIKLI, color=YESIL, alpha=0.12, label="sağlıklı (ilk 3 gün)")
    ax.axvspan(*STRESLI, color=KIRMIZI, alpha=0.12, label="stresli (son 3 gün)")
    ax.axvline(pd.Timestamp(SULAMA_DEGISTI), color="k", ls="--", lw=1, label="sulama değişti")


def main():
    print("Cihazlar okunuyor (1 dk ortalama)...")
    veri = {c: domates_cihaz(c, yeniden_ornekle="1min") for c in DOMATES_CIHAZLAR}

    # 1) Bir sulama grubu cihazı ve bir kontrol cihazı: 18 günlük sinyal
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for ax, cihaz, baslik in [(axs[0], "PN10", "Sulaması değiştirilen bitkiler (PN10)"),
                              (axs[1], "PN8", "Kontrol bitkileri, her gün 400 mL (PN8)")]:
        d = veri[cihaz].resample("10min").mean()
        ax.plot(d.index, d.CH1, lw=0.6, label="bitki A (CH1)")
        ax.plot(d.index, d.CH2, lw=0.6, label="bitki B (CH2)")
        donemleri_isaretle(ax)
        ax.set_ylabel("potansiyel [mV]")
        ax.set_title(baslik, loc="left", fontsize=11)
    axs[0].legend(loc="upper left", fontsize=8, ncol=5)
    fig.suptitle("Domates: 18 günlük elektrik sinyali (10 dk ortalama)")
    fig.tight_layout()
    fig.savefig(SEKIL / "domates_18_gun.png", dpi=120)

    # 2) Tüm bitkiler: günlük dalgalanma (gün içi standart sapma)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for cihaz, d in veri.items():
        kontrol = cihaz in DOMATES_KONTROL
        for ch in ("CH1", "CH2"):
            gunluk = d[ch].resample("1D").std()
            ax.plot(gunluk.index, gunluk, marker="o", ms=3, lw=1.2 if kontrol else 0.8,
                    color=YESIL if kontrol else KIRMIZI, alpha=0.9 if kontrol else 0.45)
    donemleri_isaretle(ax)
    ax.plot([], [], color=KIRMIZI, label="sulaması değiştirilen 12 bitki")
    ax.plot([], [], color=YESIL, label="kontrol 4 bitki (400 mL)")
    ax.set_ylabel("gün içi std [mV]")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_title("Bitki başına günlük sinyal dalgalanması (yüksek = gün içinde daha çok değişim)")
    fig.tight_layout()
    fig.savefig(SEKIL / "domates_gunluk_dalgalanma.png", dpi=120)

    # 3) Aynı bitkinin sağlıklı ve stresli bir günü (24 saat)
    d = veri["PN10"].CH1
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, gun, renk, ad in [(axs[0], "2025-06-05", YESIL, "sağlıklı"), (axs[1], "2025-06-20", KIRMIZI, "stresli")]:
        g = d.loc[gun]
        ax.plot(g.index.hour + g.index.minute / 60, g, color=renk, lw=0.8)
        ax.set_title(f"{gun} — {ad}", fontsize=11)
        ax.set_xlabel("saat")
        ax.set_xticks(range(0, 25, 4))
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("potansiyel [mV]")
    fig.suptitle("Aynı bitki (PN10, CH1): sağlıklı ve stresli bir gün")
    fig.tight_layout()
    fig.savefig(SEKIL / "domates_saglikli_vs_stresli.png", dpi=120)

    # 4) Sınıf dağılımı (1 saatlik pencere)
    e = domates_etiketler("1h")
    adlar = {0: "sağlıklı (0)", 1: "stresli (1)", 3: "kullanılmayan (3)"}
    say = e["class"].map(adlar).value_counts().reindex(adlar.values())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cubuk = ax.bar(say.index, say.values, color=[YESIL, KIRMIZI, GRI])
    ax.bar_label(cubuk)
    ax.set_ylabel("1 saatlik pencere sayısı")
    ax.set_title("Sınıf dağılımı (16 bitki × 18 gün × 24 saat)")
    fig.tight_layout()
    fig.savefig(SEKIL / "domates_sinif_dagilimi.png", dpi=120)

    # Özet tablo
    ozet = pd.DataFrame({
        f"{c}_{ch}": {"ortalama_mV": d[ch].mean(), "std_mV": d[ch].std(), "eksik_%": d[ch].isna().mean() * 100,
                      "grup": "kontrol" if c in DOMATES_KONTROL else "sulama değişti"}
        for c, d in veri.items() for ch in ("CH1", "CH2")}).T
    ozet.to_csv(get_path("tables") / "domates_bitki_ozet.csv")
    print(ozet.round(2).to_string())
    print("Şekiller:", SEKIL)


if __name__ == "__main__":
    main()
