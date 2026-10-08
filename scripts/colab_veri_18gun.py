"""18 gün Colab verisi: tüm bitkilerin tüm 6 sa pencereleri (ayrışma testi için).

  data/colab/domates_18gun_6h_500.npz — 16 bitki × 04–21.06 günleri, yazarların 6 sa pencereleri;
      %1'den fazla eksik pencereler hariç (önceki kurallarla aynı). L: robust z → 500 örnek;
      HuBERT girdisi: np.tile(L, (1, 12)). Etiket yok: gün ve bitki kimliği saklanır,
      etiketler yerelde (scripts/ayrisma_testi.py) önceden belirlenen plana göre verilir.

Kullanım: python scripts/colab_veri_18gun.py
"""

import numpy as np

from bitki_ekg.config import PROJECT_ROOT
from bitki_ekg.ekg_girdi import HUBERT_DERIVASYON_UZUNLUK, yeniden_ornekle
from bitki_ekg.preprocessing import domates_pencereler, robust_z

ILK_GUN, SON_GUN = "2025-06-04", "2025-06-21"


def main():
    X, meta = domates_pencereler("6h")
    gun = meta.day.astype(str)
    maske = ((gun >= ILK_GUN) & (gun <= SON_GUN)).to_numpy() & (np.isnan(X).mean(axis=1) <= 0.01)
    L = yeniden_ornekle(robust_z(X[maske]), HUBERT_DERIVASYON_UZUNLUK).astype(np.float16)
    bitki = meta.plant_id.to_numpy(np.int8)[maske]
    gunler = gun[maske].to_numpy().astype("U10")
    saat = meta.loc[maske, "datetime_start"].astype(str).to_numpy().astype("U19")

    hedef = PROJECT_ROOT / "data" / "colab" / "domates_18gun_6h_500.npz"
    np.savez_compressed(hedef, L=L, bitki=bitki, gun=gunler, saat=saat)
    print(hedef.name, L.shape, f"{hedef.stat().st_size / 1e6:.2f} MB")
    print("bitki başına pencere:", dict(zip(*np.unique(bitki, return_counts=True))))
    print("gün başına pencere:", dict(zip(*np.unique(gunler, return_counts=True))))


if __name__ == "__main__":
    main()
