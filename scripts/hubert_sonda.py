"""Ön test: HuBERT-ECG doğrusal sondası (GPU gerektirmez).

Model eğitilmez; yalnızca bitki pencerelerinden gömme (embedding) çıkarılır ve üstüne
lojistik regresyon konur. Aynı işlem **rastgele başlatılmış** aynı mimari için de yapılır:
önceden eğitilmiş model belirgin biçimde daha iyiyse EKG ön eğitimi bitki sinyalinde işe
yarayan bir temsil sağlıyor demektir.

Önceden belirlenen ölçütler (test bitkisine bakılarak seçim yapılmaz):
  - "son"     : son transformer katmanının zaman ortalaması
  - "ortalama": tüm katmanların (CNN çıkışı + 8 transformer) zaman ortalamalarının ortalaması
Katman katman sonuçlar yalnızca keşif amaçlı kaydedilir.

Kullanım: python scripts/hubert_sonda.py [--pencere 1h 30min] [--esleme tekrar]
Çıktılar: data/processed/hubert_gomme_*.npz, results/tables/hubert_sonda_*.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel

from bitki_ekg.config import get_path, set_seed
from bitki_ekg.ekg_girdi import hubert_girdisi
from bitki_ekg.preprocessing import domates_pencereler, ikili_gorev, lopo_bolmeleri, robust_z, yazar_bolmesi

MODEL_ADI = "Edoardo-Coppola/hubert-ecg-small"
MODEL_SURUM = "eca1c5af82da493a86a2bf0a89733234c9c3617d"  # 2026-06-04; tekrarlanabilirlik için sabit
SEED = set_seed()


def model_yukle(onceden_egitilmis: bool) -> torch.nn.Module:
    if onceden_egitilmis:
        model = AutoModel.from_pretrained(MODEL_ADI, revision=MODEL_SURUM, trust_remote_code=True)
    else:
        torch.manual_seed(SEED)
        config = AutoConfig.from_pretrained(MODEL_ADI, revision=MODEL_SURUM, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
    return model.eval()


@torch.no_grad()
def gomme_cikar(model: torch.nn.Module, X: np.ndarray, toplu: int = 32) -> np.ndarray:
    """(n, 6000) → (n, katman, gizli_boyut): her katmanın zaman ortalaması."""
    parcalar = []
    for i in range(0, len(X), toplu):
        x = torch.from_numpy(X[i:i + toplu])
        cikis = model(x, output_hidden_states=True)
        parcalar.append(torch.stack([h.mean(dim=1) for h in cikis.hidden_states], dim=1).numpy())
    return np.concatenate(parcalar)


def sonda() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))


def degerlendir(E: np.ndarray, y: np.ndarray, gruplar: np.ndarray) -> list[dict]:
    bolmeler = [("lopo", f"bitki {g}", tr, te) for g, tr, te in lopo_bolmeleri(gruplar)]
    tr, te = yazar_bolmesi(gruplar)
    bolmeler.append(("yazar", "bitki 0+12", tr, te))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    bolmeler += [("rastgele", f"katman {k}", tr, te) for k, (tr, te) in enumerate(skf.split(y, y))]

    sonuc = []
    for bolme, katman, tr, te in bolmeler:
        m = sonda().fit(E[tr], y[tr])
        p, s = m.predict(E[te]), m.predict_proba(E[te])[:, 1]
        sonuc.append({"bolme": bolme, "katman": katman, "dogruluk": accuracy_score(y[te], p),
                      "f1": f1_score(y[te], p), "auc": roc_auc_score(y[te], s)})
    return sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", nargs="+", default=["1h", "30min"])
    ap.add_argument("--esleme", default="tekrar")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    satirlar = []
    for pencere in args.pencere:
        X, meta = domates_pencereler(pencere)
        m = ikili_gorev(meta)
        X, meta = X[m], meta[m].reset_index(drop=True)
        y, gruplar = meta["class"].to_numpy(int), meta["plant_id"].to_numpy(int)
        girdi = hubert_girdisi(robust_z(X), args.esleme)

        for onceden in (True, False):
            ad = "onceden_egitilmis" if onceden else "rastgele_baslatilmis"
            dosya = get_path("processed") / f"hubert_gomme_{pencere}_{args.esleme}_{ad}.npz"
            if dosya.exists():
                E = np.load(dosya)["E"]
            else:
                t0 = time.time()
                E = gomme_cikar(model_yukle(onceden), girdi)
                np.savez_compressed(dosya, E=E)
                print(f"[{pencere}] {ad}: gömme {E.shape}, {time.time() - t0:.0f} s", flush=True)

            ozellikler = {"son": E[:, -1], "ortalama": E.mean(axis=1)}
            ozellikler.update({f"katman_{k}": E[:, k] for k in range(E.shape[1])})
            for oz, F in ozellikler.items():
                for s in degerlendir(F, y, gruplar):
                    satirlar.append({"pencere": pencere, "esleme": args.esleme, "model": ad, "ozellik": oz, **s})

    df = pd.DataFrame(satirlar)
    tablolar = get_path("tables")
    df.to_csv(tablolar / "hubert_sonda_katman.csv", index=False)
    ana = df[df.ozellik.isin(["son", "ortalama"])]
    ozet = ana.groupby(["pencere", "model", "ozellik", "bolme"])[["dogruluk", "f1", "auc"]].agg(["mean", "std"])
    ozet.round(3).to_csv(tablolar / "hubert_sonda_ozet.csv")
    pd.set_option("display.width", 200)
    print((ozet * 100).round(1).to_string())


if __name__ == "__main__":
    main()
