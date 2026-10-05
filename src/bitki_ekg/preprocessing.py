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

    X, satirlar, eksik = [], [], 0
    for node, grup in etiket.groupby("node"):
        for bas in sorted(grup.datetime_start.unique()):
            ts = pd.Timestamp(bas)
            f = klasor / f"{node}_{ts:%Y-%m-%d_%H-%M}.csv"
            if not f.exists():
                # 6h'de deneyin ilk 2 saati (00:00–01:59, sınıf 3) ayrı dosya değil; önceki günün
                # 20:00 dosyasının içinde. Böyle kısa kenar pencereleri atlanır.
                eksik += 1
                continue
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
    if eksik:
        kalan = etiket.merge(meta[["plant_id", "datetime_start"]], how="left", indicator=True)
        atlanan = kalan.loc[kalan["_merge"] == "left_only", "class"].value_counts().to_dict()
        print(f"[{pencere}] dosyası olmayan {eksik} kenar penceresi atlandı (sınıflar: {atlanan})")

    np.savez_compressed(hedef, X=X)
    meta.to_csv(hedef.with_suffix(".csv"), index=False)
    return X, meta


def sarmasik_pencereler(onbellek: bool = True, min_doluluk: float = 0.8) -> tuple[np.ndarray, pd.DataFrame]:
    """Sarmaşık verisinden 1 saatlik, 2 kanallı (CH1, CH2) pencereler ve etiketler.

    Dönüş: X (n, 2, 3600) ham değer (boşluklar doğrusal interpolasyonla doldurulmuş) ve meta
    (bitki, baslangic, doluluk, gunduz, yagmurlu, sicak_08_20, ruzgarli_08_20). Etiket eşikleri
    Buss vd. (2025): ışınım > 50 W/m², yağış > 0 mm, sıcaklık > 25 °C, rüzgâr > 1,25 m/s;
    sıcaklık ve rüzgâr etiketleri yalnızca 08:00–20:00 için tanımlıdır (dışında NaN).
    Pencere, verinin en az %80'i doluysa kullanılır.
    """
    from bitki_ekg.data import SARMASIK_BITKILER, sarmasik_bitki, sarmasik_hava

    hedef = get_path("processed") / "sarmasik_1h.npz"
    if onbellek and hedef.exists():
        return np.load(hedef)["X"], pd.read_csv(hedef.with_suffix(".csv"), parse_dates=["baslangic"])

    hava = sarmasik_hava()
    saatlik = pd.DataFrame({
        "isinim": hava.isinim_wm2.resample("1h").mean(),
        "yagis": hava.yagis_mm.resample("1h").sum(),
        "sicaklik": hava.sicaklik_c.resample("1h").mean(),
        "ruzgar": hava.ruzgar_hizi_ms.resample("1h").mean(),
    })

    X, satirlar = [], []
    for b in SARMASIK_BITKILER:
        d = sarmasik_bitki(b).resample("1s").mean()  # 1 Hz ızgara; eksik saniyeler NaN
        for bas, parca in d.groupby(d.index.floor("1h")):
            if len(parca) != 3600 or bas not in saatlik.index:
                continue
            doluluk = parca.CH1.notna().mean()
            if doluluk < min_doluluk:
                continue
            w = parca[["CH1", "CH2"]].interpolate(limit_direction="both").to_numpy(np.float32).T
            if not np.isfinite(w).all():
                continue
            X.append(w)
            satirlar.append((b, bas, doluluk))

    meta = pd.DataFrame(satirlar, columns=["bitki", "baslangic", "doluluk"])
    s = saatlik.loc[meta.baslangic].reset_index(drop=True)
    gunduz_saati = meta.baslangic.dt.hour.between(8, 19)
    meta["gunduz"] = (s.isinim > 50).astype(int)
    meta["yagmurlu"] = (s.yagis > 0).astype(int)
    meta["sicak_08_20"] = np.where(gunduz_saati, (s.sicaklik > 25).astype(float), np.nan)
    meta["ruzgarli_08_20"] = np.where(gunduz_saati, (s.ruzgar > 1.25).astype(float), np.nan)

    X = np.stack(X)
    np.savez_compressed(hedef, X=X)
    meta.to_csv(hedef.with_suffix(".csv"), index=False)
    return X, meta


def robust_z(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pencere (ve kanal) başına robust z-skor: (x − medyan) / IQR (Buss vd., 2026); son eksende."""
    med = np.nanmedian(X, axis=-1, keepdims=True)
    q1, q3 = np.nanpercentile(X, [25, 75], axis=-1, keepdims=True)
    Z = (X - med) / np.maximum(q3 - q1, eps)
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def ikili_gorev(meta: pd.DataFrame, X: np.ndarray | None = None, en_fazla_eksik: float = 0.01) -> np.ndarray:
    """İkili görevde kullanılan pencerelerin maskesi: sınıf 0 (sağlıklı) ve 1 (stresli).

    X verilirse örneklerinin %1'inden fazlası eksik olan pencereler de çıkarılır. Neden: 6 sa'te
    her bitkinin son penceresi (21.06 20:00, kayıt 00:00'da bitiyor) ~1/3 boş ve hepsi "stresli";
    boşluk sıfırla dolunca model "sıfır = stresli" kısayolunu öğrenebilir (veri_denetimi.py).
    """
    maske = meta["class"].isin([0, 1]).to_numpy()
    if X is not None:
        maske &= np.isnan(X).mean(axis=1) <= en_fazla_eksik
    return maske


def lopo_bolmeleri(gruplar: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Bitki-dışarıda-bırak: her katmanda bir bitki test, kalanlar eğitim."""
    for g in np.unique(gruplar):
        yield int(g), np.where(gruplar != g)[0], np.where(gruplar == g)[0]


def yazar_bolmesi(gruplar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yazarların bölmesi: bitki 0 ve 12 test, diğerleri eğitim."""
    test = np.isin(gruplar, YAZAR_TEST_BITKILERI)
    return np.where(~test)[0], np.where(test)[0]
