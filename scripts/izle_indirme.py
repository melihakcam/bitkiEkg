"""Süren bir indirmenin anlık ilerlemesini gösterir.

Kullanım:
    python scripts/izle_indirme.py                      # domates indirmesi
    python scripts/izle_indirme.py sarmasik_dis_ortam   # başka bir veri seti

Durdurmak için Ctrl+C (yalnızca izlemeyi kapatır, indirme sürer).
"""

import os
import sys
import time
from pathlib import Path

KOK = Path(__file__).resolve().parents[1] / "data" / "raw"
TOPLAM = {"domates_su_stresi": 10_217_462_756}  # bayt, Zenodo kaydından


def main() -> None:
    veri_seti = sys.argv[1] if len(sys.argv) > 1 else "domates_su_stresi"
    klasor = KOK / veri_seti
    parcalar = list(klasor.glob("*.part"))
    if not parcalar:
        bitmis = list(klasor.glob("*.zip"))
        print("Süren indirme yok." + (f" İnmiş dosya: {bitmis[0].name}" if bitmis else ""))
        return
    part = parcalar[0]
    toplam = TOPLAM.get(veri_seti)
    print(f"İzleniyor: {part}\n(Ctrl+C ile çık, indirme sürer)\n")

    onceki, t0 = os.stat(part).st_size, time.time()
    hizlar = []
    while True:
        time.sleep(1)
        try:
            boyut = os.stat(part).st_size
        except FileNotFoundError:
            print("\n\nİndirme bitti (dosya .zip olarak kaydedildi).")
            return
        t = time.time()
        hizlar = (hizlar + [(boyut - onceki) / (t - t0)])[-5:]  # son 5 sn ortalaması
        onceki, t0 = boyut, t
        hiz = sum(hizlar) / len(hizlar)
        satir = f"\r{boyut / 1e6:,.0f} MB"
        if toplam:
            kalan = (toplam - boyut) / hiz if hiz > 0 else 0
            satir += f" / {toplam / 1e6:,.0f} MB  (%{boyut / toplam * 100:.1f})"
            satir += f"   kalan: {time.strftime('%H:%M:%S', time.gmtime(kalan)) if hiz > 0 else '--:--:--'}"
        satir += f"   hız: {hiz / 1e6:.2f} MB/s     "
        print(satir, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nİzleme kapatıldı (indirme arka planda sürüyor).")
