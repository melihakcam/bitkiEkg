"""Donmuş temel model gömmeleri + lojistik regresyon sondası (LOPO, yazar, rastgele bölme).

`scripts/hubert_sonda.py` ile aynı protokol; ECG-FM için de kullanılır.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bitki_ekg.preprocessing import lopo_bolmeleri, yazar_bolmesi

SEED = 42


@torch.no_grad()
def ecgfm_gomme(govde: torch.nn.Module, X: np.ndarray, cihaz: str, toplu: int = 32) -> np.ndarray:
    """(n, 12, 2500) → (n, 768): son katman çıktısının zaman ortalaması."""
    govde.eval().to(cihaz)
    parcalar = []
    for i in range(0, len(X), toplu):
        x = torch.from_numpy(X[i:i + toplu]).to(cihaz)
        h = govde(source=x, mask=False, features_only=True)["x"]
        parcalar.append(h.mean(dim=1).float().cpu().numpy())
    return np.concatenate(parcalar)


@torch.no_grad()
def hubert_gomme(govde: torch.nn.Module, X: np.ndarray, cihaz: str, toplu: int = 32) -> np.ndarray:
    """(n, 6000) → (n, 512): tüm katmanların zaman ortalamalarının ortalaması (scripts/olcek_taramasi.py ile aynı)."""
    govde.eval().to(cihaz)
    parcalar = []
    for i in range(0, len(X), toplu):
        cikis = govde(torch.from_numpy(X[i:i + toplu]).to(cihaz), output_hidden_states=True)
        h = torch.stack([k.mean(dim=1) for k in cikis.hidden_states], dim=1).mean(dim=1)
        parcalar.append(h.float().cpu().numpy())
    return np.concatenate(parcalar)


def sonda_degerlendir(E: np.ndarray, y: np.ndarray, gruplar: np.ndarray) -> pd.DataFrame:
    bolmeler = [("lopo", f"bitki {g}", tr, te) for g, tr, te in lopo_bolmeleri(gruplar)]
    tr, te = yazar_bolmesi(gruplar)
    bolmeler.append(("yazar", "bitki 0+12", tr, te))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    bolmeler += [("rastgele", f"katman {k}", tr, te) for k, (tr, te) in enumerate(skf.split(y, y))]

    satir = []
    for bolme, katman, tr, te in bolmeler:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))
        m.fit(E[tr], y[tr])
        p, s = m.predict(E[te]), m.predict_proba(E[te])[:, 1]
        satir.append({"bolme": bolme, "katman": katman, "dogruluk": accuracy_score(y[te], p),
                      "f1": f1_score(y[te], p), "auc": roc_auc_score(y[te], s)})
    return pd.DataFrame(satir)
