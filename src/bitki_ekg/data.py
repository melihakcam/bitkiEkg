"""Ham veri setlerini okuyan yardımcı fonksiyonlar."""

from pathlib import Path

import pandas as pd

from bitki_ekg.config import get_path

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
