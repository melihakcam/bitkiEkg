"""Modelleri görülmemiş bitkiler üzerinde eşleştirilmiş olarak karşılaştırır.

Her model için LOPO'daki 12 bitkinin doğruluk/AUC değerleri toplanır; bir referansa göre
bitki bazında farklar alınır ve
  - Wilcoxon işaretli sıralar testi (iki yönlü, eşleştirilmiş),
  - farkın ortalaması için 10 000 tekrarlı bootstrap %95 güven aralığı
raporlanır. 12 bitkiyle istatistiksel güç sınırlıdır; p değerleri bu bağlamda yorumlanmalıdır.

Kaynaklar:
  results/tables/temel_modeller_katman.csv            (kNN, NB, LightGBM, MiniRocket)
  results/tables/hubert_sonda_katman.csv              (donmuş HuBERT, "ortalama" özellik)
  results/colab/<pencere>/<deney>/bitki_XX.json       (HuBERT ince ayar, Colab çıktısı)

Kullanım: python scripts/karsilastir.py [--pencere 1h] [--referans "tsfresh + LightGBM"]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bitki_ekg.config import PROJECT_ROOT, get_path

SEED = 42


def bitki_no(katman: str) -> int:
    return int(str(katman).split()[-1])


def tablo_olustur(pencere: str) -> pd.DataFrame:
    """Satır: bitki, sütun: (model, ölçüt)."""
    parcalar = []
    t = pd.read_csv(get_path("tables") / "temel_modeller_katman.csv")
    t = t[(t.pencere == pencere) & (t.bolme == "lopo")].assign(bitki=lambda d: d.katman.map(bitki_no))
    parcalar.append(t[["model", "bitki", "dogruluk", "auc"]])

    s = get_path("tables") / "hubert_sonda_katman.csv"
    if s.exists():
        s = pd.read_csv(s)
        s = s[(s.pencere == pencere) & (s.bolme == "lopo") & (s.ozellik == "ortalama")]
        s = s.assign(bitki=s.katman.map(bitki_no),
                     model=s.model.map({"onceden_egitilmis": "HuBERT donmuş (önceden eğitilmiş)",
                                        "rastgele_baslatilmis": "HuBERT donmuş (rastgele)"}))
        parcalar.append(s[["model", "bitki", "dogruluk", "auc"]])

    colab = PROJECT_ROOT / "results" / "colab" / pencere
    adlar = {"onceden_egitilmis": "HuBERT ince ayar (önceden eğitilmiş)",
             "rastgele_baslatilmis": "HuBERT ince ayar (rastgele)"}
    ozet = PROJECT_ROOT / "results" / "colab" / f"ozet_{pencere}.csv"  # Colab not defterinin özet çıktısı
    for deney, ad in adlar.items():
        satir = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((colab / deney).glob("bitki_*.json"))]
        if satir:
            parcalar.append(pd.DataFrame([{"model": ad, "bitki": r["test_bitkisi"], "dogruluk": r["dogruluk"],
                                           "auc": r["auc"]} for r in satir]))
        elif ozet.exists():
            o = pd.read_csv(ozet)
            o = o[o.deney == deney]
            parcalar.append(pd.DataFrame({"model": ad, "bitki": o.bitki, "dogruluk": o.dogruluk, "auc": o.auc}))

    return pd.concat(parcalar).pivot_table(index="bitki", columns="model", values=["dogruluk", "auc"])


def bootstrap_ga(fark: np.ndarray, n: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    ort = rng.choice(fark, (n, len(fark)), replace=True).mean(axis=1)
    return float(np.percentile(ort, 2.5)), float(np.percentile(ort, 97.5))


def karsilastir(tablo: pd.DataFrame, referans: str, ciftler: list[tuple[str, str]]) -> pd.DataFrame:
    satirlar = []
    modeller = tablo["dogruluk"].columns
    hepsi = [(m, referans) for m in modeller if m != referans] + ciftler
    for a, b in hepsi:
        if a not in modeller or b not in modeller:
            continue
        for olcut in ("dogruluk", "auc"):
            x = tablo[olcut][[a, b]].dropna()
            if len(x) < 5:
                continue
            fark = (x[a] - x[b]).to_numpy() * 100
            alt, ust = bootstrap_ga(fark)
            p = wilcoxon(fark).pvalue if np.any(fark != 0) else 1.0
            satirlar.append({"model": a, "karsi": b, "olcut": olcut, "n_bitki": len(x),
                             "model_ort": x[a].mean() * 100, "karsi_ort": x[b].mean() * 100,
                             "fark_ort": fark.mean(), "ga_alt": alt, "ga_ust": ust,
                             "daha_iyi_bitki": int((fark > 0).sum()), "wilcoxon_p": p})
    return pd.DataFrame(satirlar)


def ciz(tablo: pd.DataFrame, pencere: str) -> None:
    d = tablo["dogruluk"] * 100
    d = d[d.mean().sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(d.columns) + 1.5))
    for i, m in enumerate(d.columns):
        ax.scatter(d[m], np.full(len(d), i), color="#90a4ae", s=18, zorder=2)
        ax.errorbar(d[m].mean(), i, xerr=d[m].std(), fmt="o", color="#c62828", capsize=3, zorder=3)
    ax.axvline(50, color="k", ls=":", lw=1)
    ax.set_yticks(range(len(d.columns)), d.columns)
    ax.set_xlabel("görülmemiş bitkide doğruluk [%] (gri: bitki, kırmızı: ortalama ± std)")
    ax.set_title(f"Bitki-dışarıda-bırak karşılaştırması ({pencere})")
    plt.tight_layout()
    plt.savefig(get_path("figures") / f"karsilastirma_{pencere}.png", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", default="1h")
    ap.add_argument("--referans", default="tsfresh + LightGBM")
    args = ap.parse_args()

    tablo = tablo_olustur(args.pencere)
    ciftler = [("HuBERT ince ayar (önceden eğitilmiş)", "HuBERT ince ayar (rastgele)"),
               ("HuBERT donmuş (önceden eğitilmiş)", "HuBERT donmuş (rastgele)")]
    sonuc = karsilastir(tablo, args.referans, ciftler)
    sonuc.round(4).to_csv(get_path("tables") / f"karsilastirma_{args.pencere}.csv", index=False)
    (tablo * 100).round(1).to_csv(get_path("tables") / f"bitki_bazinda_{args.pencere}.csv")
    ciz(tablo, args.pencere)

    pd.set_option("display.width", 220)
    print((tablo["dogruluk"] * 100).round(0).to_string())
    print()
    print(sonuc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
