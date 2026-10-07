"""Aşama 1 Colab verisi: 6 saatlik pencerede kontrol bitkisi testi.

  data/colab/domates_kontrol_6h_500.npz — hiç strese girmeyen 4 kontrol bitkisinin (4–7) ilk 3 günü
      (y = 0) ve son 3 günü (y = 1), yazarların 6 sa pencereleri; %1'den fazla eksik pencereler hariç
      (ikili görevle aynı kural). L: robust z → 500 örnek; HuBERT girdisi: np.tile(L, (1, 12)).

Sulama bitkilerinde aynı günler sağlıklı/stresli etiketine denk gelir. Kontrol bitkilerinde
model ilk/son günleri ayırabiliyorsa, öğrenilen şey stres değil zamandır.

Kullanım: python scripts/colab_veri_kontrol.py
"""

import numpy as np

from bitki_ekg.config import PROJECT_ROOT
from bitki_ekg.ekg_girdi import HUBERT_DERIVASYON_UZUNLUK, yeniden_ornekle
from bitki_ekg.preprocessing import KONTROL_BITKILERI, domates_pencereler, robust_z

ILK_GUNLER = ("2025-06-04", "2025-06-05", "2025-06-06")  # scripts/zaman_kontrolu.py ile aynı
SON_GUNLER = ("2025-06-19", "2025-06-20", "2025-06-21")


def main():
    X, meta = domates_pencereler("6h")
    gun = meta.day.astype(str)
    maske = (meta.plant_id.isin(KONTROL_BITKILERI) & (gun.isin(ILK_GUNLER) | gun.isin(SON_GUNLER))).to_numpy()
    maske &= np.isnan(X).mean(axis=1) <= 0.01
    L = yeniden_ornekle(robust_z(X[maske]), HUBERT_DERIVASYON_UZUNLUK).astype(np.float16)
    y = gun[maske].isin(SON_GUNLER).to_numpy(np.int8)
    bitki = meta.plant_id.to_numpy(np.int8)[maske]

    hedef = PROJECT_ROOT / "data" / "colab" / "domates_kontrol_6h_500.npz"
    np.savez_compressed(hedef, L=L, y=y, bitki=bitki)
    for b in np.unique(bitki):
        print(f"bitki {b}: ilk {(y[bitki == b] == 0).sum()}, son {(y[bitki == b] == 1).sum()} pencere")
    print(hedef.name, L.shape, f"{hedef.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
