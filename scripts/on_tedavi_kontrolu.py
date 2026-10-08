"""Tedavi öncesi (plasebo) kontrolü: doz-etki bulgusu cihaz/donanım kaymasından mı geliyor?

04–06.06'da (ilk 3 gün) tüm bitkiler aynı suyu (400 mL) alıyordu. Doz-etki testindeki modelin aynısıyla
(zamana karşı dengelenmiş, cihaz-dışarıda-bırak, EKG donmuş gömmeleri) her bitki için
  Δ_plasebo = (06.06 ortalama skor) − (04.06 ortalama skor)
hesaplanır ve doz sırasıyla (kontrol < 200 mL < 100 mL) ilişkisine bakılır.
  - Gerçek doz-etki (son günler) var, plasebo yok → sinyal tedaviden sonra ortaya çıkıyor (stres).
  - Plasebo da güçlü → 100 mL cihazları zaten farklı kayıyor (donanım/bitki farkı).
Ayrıca ham sinyalde cihaz başına tedavi öncesi kayma (06.06 ile 04.06 ortalama potansiyel farkı) raporlanır.

Kullanım: python scripts/on_tedavi_kontrolu.py
"""

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bitki_ekg.config import get_path
from bitki_ekg.preprocessing import domates_pencereler, ikili_gorev
from doz_etki import CIHAZ_GRUP, DOZ, veri
from stres_dengeli import skorlar
from bitki_ekg.zaman_testleri import cihaz
from colab_veri_kontrol import ILK_GUNLER, SON_GUNLER


def gunler() -> np.ndarray:
    """notebooks/05 gömme satırlarının günleri: önce 12 sulama bitkisi (ikili görev), sonra 4 kontrol bitkisi."""
    X6, m6 = domates_pencereler("6h")
    k = ikili_gorev(m6, X6)
    gun = m6.day.astype(str)
    kk = (m6.plant_id.isin([4, 5, 6, 7]) & (gun.isin(ILK_GUNLER) | gun.isin(SON_GUNLER))).to_numpy()
    kk &= np.isnan(X6).mean(axis=1) <= 0.01
    return np.concatenate([gun[k].to_numpy(), gun[kk].to_numpy()]), \
        np.concatenate([m6.plant_id[k].to_numpy(), m6.plant_id[kk].to_numpy()])


def r_doz(skor, gun, bitki, atama, bas, son_gun) -> tuple[float, pd.DataFrame]:
    satir = []
    for b in np.unique(bitki):
        m = bitki == b
        satir.append({"bitki": int(b), "grup": atama[int(cihaz(np.array([b]))[0])],
                      "delta": skor[m & (gun == son_gun)].mean() - skor[m & (gun == bas)].mean()})
    d = pd.DataFrame(satir)
    k = d[d.grup.isin(DOZ)]
    return spearmanr(k.grup.map(DOZ), k.delta).statistic, d


def main():
    F, son, bitki, model_fn = veri("ekg")
    gun, bitki_kontrol = gunler()
    assert (bitki_kontrol == bitki).all(), "gömme satırları ile gün sırası uyuşmuyor"

    sonuc, kayit = [], []
    cihazlar = [c for c, g in CIHAZ_GRUP.items() if g in DOZ]
    gorulen = set()
    for p in [None] + list(permutations(["kontrol", "kontrol", "200", "200", "100", "100"])):
        if p is not None and (p in gorulen):
            continue
        atama = CIHAZ_GRUP if p is None else {**CIHAZ_GRUP, **dict(zip(cihazlar, p))}
        if p is not None:
            gorulen.add(p)
            if atama == CIHAZ_GRUP:
                continue
        kontrol = tuple(sorted(c for c, g in atama.items() if g == "kontrol"))
        skor = skorlar(F, son, bitki, kontrol, model_fn)
        r_son, d_son = r_doz(skor, gun, bitki, atama, "2025-06-04", "2025-06-21")  # bilgi: uç günler
        r_pla, d_pla = r_doz(skor, gun, bitki, atama, "2025-06-04", "2025-06-06")
        kayit.append((r_son, r_pla))
        if p is None:
            sonuc = d_pla.rename(columns={"delta": "delta_plasebo"}).assign(delta_son=d_son.delta)
    gercek, sifir = kayit[0], np.array(kayit[1:])
    ozet = {"r_plasebo": gercek[1], "p_plasebo": (1 + (sifir[:, 1] >= gercek[1]).sum()) / len(kayit),
            "r_son_gun": gercek[0], "p_son_gun": (1 + (sifir[:, 0] >= gercek[0]).sum()) / len(kayit),
            "n_atama": len(kayit)}
    print(sonuc.groupby("grup")[["delta_plasebo", "delta_son"]].mean().round(3).to_string())
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in ozet.items()})

    # Ham sinyal: cihaz başına tedavi öncesi kayma (mV)
    X, meta = domates_pencereler("6h")
    g = meta.day.astype(str)
    ham = []
    for b in range(16):
        m = (meta.plant_id == b).to_numpy()
        ham.append({"bitki": b, "grup": CIHAZ_GRUP[b // 2],
                    "kayma_mV": np.nanmean(X[m & (g == "2025-06-06").to_numpy()]) - np.nanmean(X[m & (g == "2025-06-04").to_numpy()]),
                    "oynaklik_mV": np.nanmean(np.nanstd(X[m & g.isin(ILK_GUNLER).to_numpy()], axis=1))})
    ham = pd.DataFrame(ham)
    print(ham.groupby("grup")[["kayma_mV", "oynaklik_mV"]].agg(lambda s: s.abs().mean() if s.name == "kayma_mV" else s.mean()).round(2).to_string())

    sonuc.to_csv(get_path("tables") / "on_tedavi_bitki.csv", index=False)
    pd.DataFrame([ozet]).to_csv(get_path("tables") / "on_tedavi_ozet.csv", index=False)
    ham.to_csv(get_path("tables") / "on_tedavi_ham.csv", index=False)


if __name__ == "__main__":
    main()
