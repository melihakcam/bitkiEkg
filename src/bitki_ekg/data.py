"""Ham veri setlerini okuyan yardımcı fonksiyonlar."""

from pathlib import Path

import pandas as pd

from bitki_ekg.config import get_path

# --- Domates sulama stresi (Buss vd., 2026) ---------------------------------

# Her PhytoNode cihazı iki bitkiyi ölçer: CH1 ve CH2 ayrı bitkilerdir.
DOMATES_CIHAZLAR = ("PN2", "PN5", "PN8", "PN9", "PN10", "PN11", "PN12", "PN16")
DOMATES_KONTROL = ("PN8", "PN9")  # 400 mL/gün kontrol grubu (bitki 4–7), hep sınıf 3


def domates_klasoru() -> Path:
    return get_path("raw") / "domates_su_stresi" / "AdditionalMaterial" / "Tomato_Zenodo"


def domates_cihaz(cihaz: str, yeniden_ornekle: str | None = None) -> pd.DataFrame:
    """Bir cihazın 18 günlük kaydını 1 saatlik pencere dosyalarından birleştirir.

    Sütunlar: CH1, CH2 (mV, 1 Hz) — iki ayrı bitki. `yeniden_ornekle` (ör. "1min")
    verilirse ortalama alınarak seyreltilir.
    """
    dosyalar = sorted((domates_klasoru() / "00_time_windows" / "Exp1" / "1h").glob(f"{cihaz}_*.csv"))
    parcalar = []
    for f in dosyalar:
        df = pd.read_csv(f, parse_dates=["datetime"], index_col="datetime")
        if yeniden_ornekle:
            df = df.resample(yeniden_ornekle).mean()
        parcalar.append(df)
    df = pd.concat(parcalar).sort_index()
    return df[~df.index.duplicated(keep="first")]


def domates_etiketler(pencere: str = "1h") -> pd.DataFrame:
    """Yazarların pencere etiketleri: plant_id, node, day, datetime_start/end, class.

    class: 0 = sağlıklı (ilk 3 gün), 1 = stresli (son 3 gün), 3 = kullanılmayan
    (ara günler ve kontrol bitkileri).
    """
    f = domates_klasoru() / "01_features" / "Exp1" / pencere / "features_with_class.csv"
    return pd.read_csv(f, usecols=["plant_id", "node", "day", "datetime_start", "datetime_end", "class"],
                       parse_dates=["datetime_start", "datetime_end"])


# --- Sarmaşık dış ortam (Buss vd., 2025) ------------------------------------

SARMASIK_BITKILER = ("P1", "P2", "P3", "P5")

# Hava durumu dosyasındaki kanallar (Almanca adı -> kısa ad); her kanal 8 sütun kaplar,
# ilk sütun ölçülen değerdir.
HAVA_KANALLARI = {
    "Windgeschwindigkeit": "ruzgar_hizi_ms",
    "Windrichtung": "ruzgar_yonu_derece",
    "Lufttemperatur": "sicaklik_c",
    "rel. Feuchte": "bagil_nem_yuzde",
    "Globalstrahlung": "isinim_wm2",
    "Niederschlag": "yagis_mm",
    "Taupunkttemperatur": "ciy_noktasi_c",
    "Verdunstung Haude": "buharlasma_mm",
}


def sarmasik_klasoru() -> Path:
    return get_path("raw") / "sarmasik_dis_ortam"


def sarmasik_dosyalari(bitki: str) -> list[Path]:
    """Bir bitkinin 12 saatlik CSV parçalarını zaman sırasıyla döndürür."""
    return sorted((sarmasik_klasoru() / "Plant_data" / bitki).glob("*.csv"))


def sarmasik_bitki(bitki: str, yeniden_ornekle: str | None = None) -> pd.DataFrame:
    """Bir sarmaşık bitkisinin tüm kaydını okur.

    Sütunlar: CH1, CH2 (ham değer, ~1 Hz). İndeks: zaman damgası.
    `yeniden_ornekle` verilirse (ör. "1min", "10min") ortalama alınarak seyreltilir;
    tüm kayıt 1 Hz'de birkaç milyon satır olduğundan görselleştirme için önerilir.
    """
    parcalar = []
    for f in sarmasik_dosyalari(bitki):
        df = pd.read_csv(f, parse_dates=["datetime"], index_col="datetime")
        # P1/P3_2024-08-10_12-00-00.csv, 2024-01-01'den başlayan ~19 milyon boş satır içeriyor
        df = df.dropna(how="all")
        if yeniden_ornekle:
            df = df.resample(yeniden_ornekle).mean()
        parcalar.append(df)
    df = pd.concat(parcalar).sort_index()
    df = df[~df.index.duplicated(keep="first")]  # parça sınırlarındaki çakışan satırlar
    if yeniden_ornekle:
        # Dosyalar arası boşluklar NaN olsun; yoksa grafikler boşlukları düz çizgiyle birleştirir
        df = df.asfreq(yeniden_ornekle)
    df.attrs["bitki"] = bitki
    return df


def sarmasik_hava() -> pd.DataFrame:
    """Aylık hava durumu dosyalarını tek tabloda birleştirir (10 dk aralık)."""
    parcalar = []
    for f in sorted((sarmasik_klasoru() / "weather_data" / "04_weather").glob("*.csv")):
        ham = pd.read_csv(f, header=None, skiprows=4, dtype=str, encoding="latin-1")
        kanallar = pd.read_csv(f, header=None, nrows=2, dtype=str, encoding="latin-1").iloc[1]
        # Gece yarısı "24:00:00" olarak yazılmış: ertesi günün 00:00'ı
        gece = ham[1].str.strip() == "24:00:00"
        zaman = pd.to_datetime(ham[0] + " " + ham[1].where(~gece, "00:00:00"), format="%d.%m.%Y %H:%M:%S")
        zaman = zaman + pd.to_timedelta(gece.astype(int), unit="D")
        df = pd.DataFrame(index=pd.DatetimeIndex(zaman))
        for sutun, ad in kanallar.items():
            if isinstance(ad, str) and ad.strip() in HAVA_KANALLARI:
                deger = ham[sutun].str.replace(",", ".", regex=False).str.strip()
                df[HAVA_KANALLARI[ad.strip()]] = pd.to_numeric(deger, errors="coerce").to_numpy()
        parcalar.append(df)
    df = pd.concat(parcalar).sort_index()
    df.index.name = "datetime"
    return df
