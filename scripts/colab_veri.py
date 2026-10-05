"""Colab için küçük veri dosyası üretir: data/colab/domates_ikili_<pencere>.npz

İçerik: yalnızca ikili görevdeki pencereler (12 sulama bitkisi, ilk/son 3 gün),
robust z-skorlu (float16), etiket ve bitki kimliği. Kaynak veri: Buss vd. (2026),
Zenodo 10.5281/zenodo.18876513, CC-BY 4.0.

Kullanım: python scripts/colab_veri.py
"""

import numpy as np

from bitki_ekg.config import PROJECT_ROOT
from bitki_ekg.preprocessing import domates_pencereler, ikili_gorev, robust_z


def main():
    hedef = PROJECT_ROOT / "data" / "colab"
    hedef.mkdir(parents=True, exist_ok=True)
    for pencere in ("1h", "30min"):
        X, meta = domates_pencereler(pencere)
        m = ikili_gorev(meta)
        Z = robust_z(X[m]).astype(np.float16)
        y = meta.loc[m, "class"].to_numpy(np.int8)
        g = meta.loc[m, "plant_id"].to_numpy(np.int8)
        dosya = hedef / f"domates_ikili_{pencere}.npz"
        np.savez_compressed(dosya, Z=Z, y=y, bitki=g)
        print(f"{dosya.name}: {Z.shape}, {dosya.stat().st_size / 1e6:.1f} MB, sınıflar {np.bincount(y)}")


if __name__ == "__main__":
    main()
