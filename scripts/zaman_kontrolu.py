"""Zaman karıştırıcısı testi.

Soru: Model "sağlıklı (ilk 3 gün) / stresli (son 3 gün)" ayrımını sulama stresinden mi,
yoksa yalnızca deneyin başı/sonu farkından (elektrot oturması, bitki büyümesi, sera koşulları)
mı öğreniyor?

Yöntem: Hiç strese girmeyen 4 kontrol bitkisinin (400 mL/gün) ilk 3 günü "0", son 3 günü "1"
olarak etiketlenir ve aynı model (tsfresh + LightGBM) bitki-dışarıda-bırak ile değerlendirilir.
  - Kontrolde doğruluk ≈ %50 → ayrım sulamadan kaynaklanıyor.
  - Kontrolde doğruluk yüksek → zaman etkisi var; sulama bitkilerindeki sonucun bir kısmı
    zamandan kaynaklanıyor olabilir.
Karşılaştırma için aynı model sulama bitkilerinde (12 bitki) de çalıştırılır.

Kullanım: python scripts/zaman_kontrolu.py [--pencere 1h 30min]
"""

import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from bitki_ekg.config import get_path, set_seed
from bitki_ekg.preprocessing import KONTROL_BITKILERI, domates_pencereler, lopo_bolmeleri

SEED = set_seed()
ILK_GUNLER = ("2025-06-04", "2025-06-05", "2025-06-06")
SON_GUNLER = ("2025-06-19", "2025-06-20", "2025-06-21")


def etiketle(meta: pd.DataFrame, grup: str) -> tuple[np.ndarray, np.ndarray]:
    """Grubun ilk/son 3 gün pencereleri için maske ve etiket (0 = ilk, 1 = son)."""
    kontrol = meta.plant_id.isin(KONTROL_BITKILERI)
    bitki = kontrol if grup == "kontrol" else ~kontrol
    gun = meta.day.astype(str)
    maske = (bitki & (gun.isin(ILK_GUNLER) | gun.isin(SON_GUNLER))).to_numpy()
    y = gun[maske].isin(SON_GUNLER).astype(int).to_numpy()
    return maske, y


def calistir(pencere: str) -> list[dict]:
    from temel_modeller import tsfresh_oznitelikleri  # aynı öznitelik hizalaması

    _, meta = domates_pencereler(pencere)
    F = tsfresh_oznitelikleri(pencere, meta)
    satirlar = []
    for grup in ("kontrol", "sulama"):
        maske, y = etiketle(meta, grup)
        X, gruplar = F[maske], meta.plant_id.to_numpy()[maske]
        for g, tr, te in lopo_bolmeleri(gruplar):
            model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
            model.fit(X[tr], y[tr])
            p, s = model.predict(X[te]), model.predict_proba(X[te])[:, 1]
            satirlar.append({"pencere": pencere, "grup": grup, "bitki": g, "n": len(te),
                             "dogruluk": accuracy_score(y[te], p), "f1": f1_score(y[te], p),
                             "auc": roc_auc_score(y[te], s)})
        print(f"[{pencere}] {grup}: {len(np.unique(gruplar))} bitki, {maske.sum()} pencere")
    return satirlar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", nargs="+", default=["1h", "30min"])
    args = ap.parse_args()

    df = pd.DataFrame([s for p in args.pencere for s in calistir(p)])
    df.to_csv(get_path("tables") / "zaman_kontrolu_katman.csv", index=False)
    ozet = df.groupby(["pencere", "grup"])[["dogruluk", "f1", "auc"]].agg(["mean", "std"]).round(3)
    ozet.to_csv(get_path("tables") / "zaman_kontrolu_ozet.csv")
    print(ozet.to_string())


if __name__ == "__main__":
    main()
