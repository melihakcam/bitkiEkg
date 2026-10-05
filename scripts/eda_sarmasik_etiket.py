"""Sarmaşık verisi: 1 saatlik pencerelerin makaledeki eşiklerle etiket dağılımı.

Etiket kuralları (Buss vd., 2025):
  gündüz/gece    : saatlik ortalama ışınım > 50 W/m² → gündüz
  yağmurlu/kuru  : saatteki toplam yağış > 0 mm → yağmurlu
  soğuk/sıcak    : saatlik ortalama sıcaklık > 25 °C → sıcak (yalnızca 08:00–20:00)
  rüzgârlı/sakin : saatlik ortalama rüzgâr > 1,25 m/s → rüzgârlı (yalnızca 08:00–20:00)
Pencere, bitki verisinin en az %80'i doluysa kullanılır (makaledeki gün kuralının saatlik karşılığı).

Kullanım: python scripts/eda_sarmasik_etiket.py
"""

import pandas as pd

from bitki_ekg.config import get_path
from bitki_ekg.data import SARMASIK_BITKILER, sarmasik_bitki, sarmasik_hava


def main():
    hava = sarmasik_hava()
    saatlik = pd.DataFrame({
        "isinim": hava.isinim_wm2.resample("1h").mean(),
        "yagis": hava.yagis_mm.resample("1h").sum(),
        "sicaklik": hava.sicaklik_c.resample("1h").mean(),
        "ruzgar": hava.ruzgar_hizi_ms.resample("1h").mean(),
    })
    gunduz_saati = (saatlik.index.hour >= 8) & (saatlik.index.hour < 20)

    satirlar = []
    for b in SARMASIK_BITKILER:
        d = sarmasik_bitki(b, yeniden_ornekle="1min")
        dolu = d.CH1.notna().resample("1h").mean()
        gecerli = dolu[dolu >= 0.8].index.intersection(saatlik.index)
        s = saatlik.loc[gecerli]
        g = gunduz_saati[saatlik.index.get_indexer(gecerli)]
        satirlar.append({
            "bitki": b, "gecerli_saat": len(gecerli),
            "gunduz": int((s.isinim > 50).sum()), "gece": int((s.isinim <= 50).sum()),
            "yagmurlu": int((s.yagis > 0).sum()), "kuru": int((s.yagis == 0).sum()),
            "sicak_08_20": int((s.sicaklik[g] > 25).sum()), "soguk_08_20": int((s.sicaklik[g] <= 25).sum()),
            "ruzgarli_08_20": int((s.ruzgar[g] > 1.25).sum()), "sakin_08_20": int((s.ruzgar[g] <= 1.25).sum()),
        })
    tablo = pd.DataFrame(satirlar).set_index("bitki")
    tablo.loc["toplam"] = tablo.sum()
    tablo.to_csv(get_path("tables") / "sarmasik_sinif_dagilimi.csv")
    print(tablo.to_string())


if __name__ == "__main__":
    main()
