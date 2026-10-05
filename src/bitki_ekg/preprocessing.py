"""Pencere veri kümelerini oluşturma, normalizasyon ve bitki bazlı bölmeler."""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from bitki_ekg.config import get_path
from bitki_ekg.data import domates_etiketler, domates_klasoru

# Yazarların (Buss vd., 2026) test bitkileri: PN2-CH1 (aşırı sulanmış ya da 100 mL grubundan biri)
# ve PN12-CH1. Kalan 10 sulama bitkisi eğitim/doğrulama; 4 kontrol bitkisi (4–7) dışarıda.
YAZAR_TEST_BITKILERI = (0, 12)
KONTROL_BITKILERI = (4, 5, 6, 7)


def domates_pencereler(pencere: str = "1h", onbellek: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Domates pencerelerini (n, L) dizisi ve meta tablosu olarak döndürür.

    Meta sütunları: plant_id, node, kanal, day, datetime_start, class.
    plant_id eşlemesi doğrulandı: her cihazda küçük numara CH1, büyük numara CH2
    (`value__mean` özniteliği ile pencere ortalaması birebir aynı).
    Sonuç `data/processed/domates_<pencere>.npz` dosyasına önbelleklenir.
    """
    hedef = get_path("processed") / f"domates_{pencere}.npz"
    if onbellek and hedef.exists():
        z = np.load(hedef, allow_pickle=False)
        meta = pd.read_csv(hedef.with_suffix(".csv"), parse_dates=["datetime_start"])
        return z["X"], meta

    etiket = domates_etiketler(pencere)
    ilk_id = etiket.groupby("node").plant_id.min()  # CH1 = küçük numara
    klasor = domates_klasoru() / "00_time_windows" / "Exp1" / pencere

    X, satirlar = [], []
    for node, grup in etiket.groupby("node"):
        for bas in sorted(grup.datetime_start.unique()):
            ts = pd.Timestamp(bas)
            f = klasor / f"{node}_{ts:%Y-%m-%d_%H-%M}.csv"
            w = pd.read_csv(f, usecols=["CH1", "CH2"])
            for kanal in ("CH1", "CH2"):
                pid = ilk_id[node] + (kanal == "CH2")
                X.append(w[kanal].to_numpy(np.float32))
                satirlar.append((pid, node, kanal, ts))
    L = int(np.median([len(x) for x in X]))
    X = np.stack([np.pad(x[:L], (0, max(0, L - len(x))), constant_values=np.nan) for x in X])

    meta = pd.DataFrame(satirlar, columns=["plant_id", "node", "kanal", "datetime_start"])
    meta = meta.merge(etiket[["plant_id", "datetime_start", "day", "class"]], on=["plant_id", "datetime_start"],
                      how="left", validate="one_to_one")
    if meta["class"].isna().any():
        raise ValueError("Etiketi bulunamayan pencere var")

    np.savez_compressed(hedef, X=X)
    meta.to_csv(hedef.with_suffix(".csv"), index=False)
    return X, meta


def robust_z(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pencere başına robust z-skor: (x − medyan) / IQR (Buss vd., 2026)."""
    med = np.nanmedian(X, axis=1, keepdims=True)
    q1, q3 = np.nanpercentile(X, [25, 75], axis=1, keepdims=True)
    Z = (X - med) / np.maximum(q3 - q1, eps)
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def ikili_gorev(meta: pd.DataFrame) -> np.ndarray:
    """İkili görevde kullanılan pencerelerin maskesi: sınıf 0 (sağlıklı) ve 1 (stresli)."""
    return meta["class"].isin([0, 1]).to_numpy()


def lopo_bolmeleri(gruplar: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Bitki-dışarıda-bırak: her katmanda bir bitki test, kalanlar eğitim."""
    for g in np.unique(gruplar):
        yield int(g), np.where(gruplar != g)[0], np.where(gruplar == g)[0]


def yazar_bolmesi(gruplar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yazarların bölmesi: bitki 0 ve 12 test, diğerleri eğitim."""
    test = np.isin(gruplar, YAZAR_TEST_BITKILERI)
    return np.where(~test)[0], np.where(test)[0]
