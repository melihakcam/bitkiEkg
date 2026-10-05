"""Alana uyarlamalı ek ön eğitim (DAPT): HuBERT-ECG gövdesini etiketsiz bitki verisiyle uyarlama.

Amaç: EKG ağırlıklarından (ya da kontrol için rastgele ağırlıklardan) başlayıp modeli etiket
kullanmadan bitki sinyaline alıştırmak; ardından aynı LOPO ince ayar (`ince_ayar.py`).

Hedef (maskeli yeniden yapılandırma): zaman adımlarının ~%30'u `mask_time_indices` ile
maskelenir; model maskeli adımlarda **donmuş CNN öznitelik çıkarıcısının** çıktısını (katman
normalize) doğrusal bir başla tahmin eder, kayıp MSE. CNN ince ayarda da donmuş olduğundan
hedef iki aşamada da aynı uzaydadır.

Sızıntı yok: DAPT verisi yalnızca değerlendirmede test edilmeyen kaynaklardan gelir
(kontrol domates bitkileri 4–7 + sarmaşık). Ayarlar deney öncesi sabittir (DAPT_AYARLARI).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from bitki_ekg.ince_ayar import MODEL_ADI, MODEL_SURUM, tohumla


@dataclass(frozen=True)
class DaptAyarlari:
    epoch: int = 10
    toplu: int = 32
    lr: float = 5e-5
    agirlik_curumesi: float = 0.01
    isinma_orani: float = 0.1
    maske_orani: float = 0.3   # maskelenen zaman adımı oranı
    maske_uzunluk: int = 5     # ardışık maskeli adım (span) uzunluğu
    tohum: int = 42


DAPT_AYARLARI = DaptAyarlari()


def hubert_govde(onceden_egitilmis: bool, tohum: int) -> nn.Module:
    config = AutoConfig.from_pretrained(MODEL_ADI, revision=MODEL_SURUM, trust_remote_code=True)
    if onceden_egitilmis:
        return AutoModel.from_pretrained(MODEL_ADI, revision=MODEL_SURUM, config=config, trust_remote_code=True)
    torch.manual_seed(tohum)
    return AutoModel.from_config(config, trust_remote_code=True)


def span_maske(B: int, T: int, oran: float, uzunluk: int, rng: np.random.Generator) -> np.ndarray:
    """(B, T) bool maske: her örnekte ~oran × T adım, `uzunluk` uzunluğunda parçalar halinde."""
    m = np.zeros((B, T), dtype=bool)
    n_span = max(1, int(round(oran * T / uzunluk)))
    for b in range(B):
        for s in rng.integers(0, max(1, T - uzunluk), n_span):
            m[b, s:s + uzunluk] = True
    return m


class DaptModeli(nn.Module):
    def __init__(self, govde: nn.Module):
        super().__init__()
        self.govde = govde
        self.govde.feature_extractor._freeze_parameters()
        cnn_boyut = govde.config.conv_dim[-1]
        self.bas = nn.Linear(govde.config.hidden_size, cnn_boyut)

    def forward(self, x: torch.Tensor, maske: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hedef = self.govde.feature_extractor(x).transpose(1, 2)  # (B, T, C)
            hedef = nn.functional.layer_norm(hedef.float(), hedef.shape[-1:])
        h = self.govde(x, mask_time_indices=maske).last_hidden_state  # (B, T, D)
        tahmin = self.bas(h).float()
        return nn.functional.mse_loss(tahmin[maske], hedef[maske])


def dapt_egit(X: np.ndarray, onceden_egitilmis: bool, cikti: Path, cihaz: str,
              ayar: DaptAyarlari = DAPT_AYARLARI, sinir: int | None = None) -> Path:
    """DAPT çalıştırır; gövde ağırlıklarını `cikti/govde.pt`, geçmişi `cikti/gecmis.json`'a yazar.

    Dosya zaten varsa atlanır (oturum koparsa yeniden hesaplanmaz).
    """
    cikti.mkdir(parents=True, exist_ok=True)
    agirlik = cikti / "govde.pt"
    if agirlik.exists():
        print(f"  DAPT önceden tamamlanmış, atlandı: {agirlik}", flush=True)
        return agirlik

    tohumla(ayar.tohum)
    rng = np.random.default_rng(ayar.tohum)
    if sinir:
        X = X[rng.permutation(len(X))[:sinir]]
    model = DaptModeli(hubert_govde(onceden_egitilmis, ayar.tohum)).to(cihaz)
    with torch.no_grad():
        T = model.govde.feature_extractor(torch.from_numpy(X[:1]).to(cihaz)).shape[-1]

    parametreler = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parametreler, lr=ayar.lr, weight_decay=ayar.agirlik_curumesi)
    adim = ayar.epoch * int(np.ceil(len(X) / ayar.toplu))
    zam = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ayar.lr, total_steps=adim,
                                              pct_start=ayar.isinma_orani, anneal_strategy="linear")
    olcek = torch.amp.GradScaler(enabled=cihaz.startswith("cuda"))
    gecmis, t0 = [], time.time()
    for ep in range(ayar.epoch):
        model.train()
        toplam, n = 0.0, 0
        sira = rng.permutation(len(X))
        for i in range(0, len(sira), ayar.toplu):
            idx = sira[i:i + ayar.toplu]
            x = torch.from_numpy(X[idx]).to(cihaz)
            m = torch.from_numpy(span_maske(len(idx), T, ayar.maske_orani, ayar.maske_uzunluk, rng)).to(cihaz)
            with torch.autocast(device_type=cihaz.split(":")[0], enabled=cihaz.startswith("cuda")):
                kayip = model(x, m)
            opt.zero_grad(set_to_none=True)
            olcek.scale(kayip).backward()
            olcek.unscale_(opt)
            nn.utils.clip_grad_norm_(parametreler, 1.0)
            olcek.step(opt)
            olcek.update()
            zam.step()
            toplam += kayip.item() * len(idx)
            n += len(idx)
        gecmis.append({"epoch": ep, "kayip": toplam / n})
        print(f"  DAPT epoch {ep}: kayıp {toplam / n:.4f} ({time.time() - t0:.0f} s)", flush=True)

    torch.save(model.govde.state_dict(), agirlik)
    (cikti / "gecmis.json").write_text(json.dumps({"onceden_egitilmis": onceden_egitilmis, "n": len(X),
                                                   "gecmis": gecmis, "ayarlar": asdict(ayar)},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return agirlik
