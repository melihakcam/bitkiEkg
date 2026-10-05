"""DAPT deneyi için Colab veri dosyaları (6 saatlik ölçek, HuBERT derivasyon uzunluğu 500).

  data/colab/domates_ikili_6h_500.npz — değerlendirme: 12 sulama bitkisi, eksiksiz 6 sa pencereler
      (L: robust z → 500 örnek, y, bitki). HuBERT girdisi: np.tile(L, (1, 12)).
  data/colab/dapt_6h_500.npz — DAPT (etiketsiz, değerlendirmede HİÇ test edilmeyen kaynaklar):
      * kontrol domates bitkileri 4–7: ardışık 6 saatlik dilimler (1 sa pencerelerden, 3 sa adım)
      * sarmaşık P1, P2, P3, P5: ardışık 6 saatlik dilimler, her kanal ayrı örnek (3 sa adım)
      (D: robust z → 500 örnek, kaynak etiketi)

Kullanım: python scripts/colab_veri_dapt.py
"""

import numpy as np
import pandas as pd

from bitki_ekg.config import PROJECT_ROOT
from bitki_ekg.ekg_girdi import HUBERT_DERIVASYON_UZUNLUK, yeniden_ornekle
from bitki_ekg.preprocessing import (KONTROL_BITKILERI, domates_pencereler, ikili_gorev, robust_z,
                                     sarmasik_pencereler)

ADIM_SAAT = 3


def ardisik_6sa(seriler: np.ndarray, zamanlar: pd.Series) -> list[np.ndarray]:
    """Saatlik pencerelerden (n, L) ardışık 6'lı birleşimler (3 sa adımla)."""
    sira = np.argsort(zamanlar.to_numpy())
    S, t = seriler[sira], zamanlar.to_numpy()[sira]
    cikti = []
    for i in range(0, len(t) - 5, ADIM_SAAT):
        if (t[i + 5] - t[i]) == np.timedelta64(5, "h"):  # 6 ardışık saat, boşluk yok
            cikti.append(np.concatenate(S[i:i + 6]))
    return cikti


def kucult(P: np.ndarray) -> np.ndarray:
    return yeniden_ornekle(robust_z(P), HUBERT_DERIVASYON_UZUNLUK).astype(np.float16)


def main():
    hedef = PROJECT_ROOT / "data" / "colab"

    # Değerlendirme seti (yazarların 6 sa pencereleri, eksik son pencereler hariç)
    X6, m6 = domates_pencereler("6h")
    k = ikili_gorev(m6, X6)
    np.savez_compressed(hedef / "domates_ikili_6h_500.npz", L=kucult(X6[k]),
                        y=m6.loc[k, "class"].to_numpy(np.int8), bitki=m6.loc[k, "plant_id"].to_numpy(np.int8))
    print(f"değerlendirme: {k.sum()} pencere, bitkiler {sorted(m6.loc[k, 'plant_id'].unique())}")

    # DAPT: kontrol bitkileri
    parcalar, kaynak = [], []
    X1, m1 = domates_pencereler("1h")
    for p in KONTROL_BITKILERI:
        sec = (m1.plant_id == p).to_numpy()
        for s in ardisik_6sa(X1[sec], m1.loc[sec, "datetime_start"].dt.floor("h")):
            if np.isnan(s).mean() <= 0.01:
                parcalar.append(s)
                kaynak.append(f"domates_kontrol_{p}")

    # DAPT: sarmaşık (her kanal ayrı)
    XS, mS = sarmasik_pencereler()
    for b in mS.bitki.unique():
        sec = (mS.bitki == b).to_numpy()
        for kanal in (0, 1):
            for s in ardisik_6sa(XS[sec, kanal], mS.loc[sec, "baslangic"]):
                parcalar.append(s)
                kaynak.append(f"sarmasik_{b}_CH{kanal + 1}")

    D = kucult(np.stack(parcalar).astype(np.float32))
    kaynak = np.array(kaynak)
    np.savez_compressed(hedef / "dapt_6h_500.npz", D=D, kaynak=kaynak)
    print(f"DAPT: {len(D)} pencere | " + str(pd.Series([k.rsplit('_', 1)[0] if k.startswith('domates') else
                                                        k.split('_')[0] for k in kaynak]).value_counts().to_dict()))
    for f in ("domates_ikili_6h_500.npz", "dapt_6h_500.npz"):
        print(f, f"{(hedef / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
