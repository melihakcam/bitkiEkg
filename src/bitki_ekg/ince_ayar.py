"""HuBERT-ECG ince ayarı: bitki-dışarıda-bırak (LOPO) değerlendirme.

Tüm ayarlar deney başlamadan sabitlenmiştir (AYARLAR); test bitkisine bakılarak hiçbir seçim
yapılmaz. Her LOPO katmanında eğitim bitkilerinden 2'si iç doğrulama için ayrılır ve en iyi
epoch iç doğrulama AUC'sine göre seçilir. Her katmanın sonucu ayrı bir JSON dosyasına yazılır;
oturum koparsa tamamlanan katmanlar atlanarak kaldığı yerden devam edilir.

Deneyler:
  onceden_egitilmis   — HuBERT-ECG small ağırlıkları (EKG ön eğitimi)
  rastgele_baslatilmis — aynı mimari, rastgele ağırlıklar (kontrol)
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from transformers import AutoConfig, AutoModel

MODEL_ADI = "Edoardo-Coppola/hubert-ecg-small"
MODEL_SURUM = "eca1c5af82da493a86a2bf0a89733234c9c3617d"


@dataclass(frozen=True)
class Ayarlar:
    epoch: int = 15
    toplu: int = 32
    lr_govde: float = 3e-5      # önceden eğitilmiş gövde
    lr_bas: float = 1e-3        # sınıflandırma başı
    agirlik_curumesi: float = 0.01
    isinma_orani: float = 0.1
    mask_time_prob: float = 0.05  # ön eğitimdeki 0,33 yerine ince ayar için tipik değer
    ic_dogrulama_bitki: int = 2
    tohum: int = 42


AYARLAR = Ayarlar()


class Siniflandirici(nn.Module):
    def __init__(self, onceden_egitilmis: bool, ayar: Ayarlar = AYARLAR):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_ADI, revision=MODEL_SURUM, trust_remote_code=True)
        config.mask_time_prob = ayar.mask_time_prob
        if onceden_egitilmis:
            self.govde = AutoModel.from_pretrained(MODEL_ADI, revision=MODEL_SURUM, config=config,
                                                   trust_remote_code=True)
        else:
            self.govde = AutoModel.from_config(config, trust_remote_code=True)
        self.govde.feature_extractor._freeze_parameters()  # CNN öznitelik çıkarıcı dondurulur
        self.bas = nn.Sequential(nn.Dropout(0.1), nn.Linear(config.hidden_size, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.govde(x).last_hidden_state  # (B, T, D)
        return self.bas(h.mean(dim=1))


def tohumla(tohum: int) -> None:
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    torch.cuda.manual_seed_all(tohum)


@torch.no_grad()
def olasilik(model: nn.Module, X: np.ndarray, cihaz: str, toplu: int = 64) -> np.ndarray:
    model.eval()
    p = []
    for i in range(0, len(X), toplu):
        x = torch.from_numpy(X[i:i + toplu]).to(cihaz)
        with torch.autocast(device_type=cihaz.split(":")[0], enabled=cihaz.startswith("cuda")):
            p.append(torch.softmax(model(x).float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(p)


def olc(y: np.ndarray, p: np.ndarray) -> dict:
    t = (p >= 0.5).astype(int)
    return {"dogruluk": float(accuracy_score(y, t)), "f1": float(f1_score(y, t, zero_division=0)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")}


def katman_egit(X, y, gruplar, test_bitkisi: int, onceden_egitilmis: bool, cihaz: str,
                ayar: Ayarlar = AYARLAR, sinir: int | None = None) -> dict:
    """Tek bir LOPO katmanı: eğit, iç doğrulamada en iyi epoch'u seç, test bitkisinde ölç.

    `sinir` yalnızca hızlı yerel test içindir (her kümeden ilk n pencere).
    """
    tohumla(ayar.tohum + test_bitkisi)
    egitim_bitkileri = sorted(set(gruplar) - {test_bitkisi})
    rng = np.random.default_rng(ayar.tohum + test_bitkisi)
    ic_dog = set(rng.choice(egitim_bitkileri, ayar.ic_dogrulama_bitki, replace=False).tolist())
    tr = np.where(~np.isin(gruplar, [test_bitkisi, *ic_dog]))[0]
    va = np.where(np.isin(gruplar, list(ic_dog)))[0]
    te = np.where(gruplar == test_bitkisi)[0]
    if sinir:
        tr, va, te = (rng.permutation(i)[:sinir] for i in (tr, va, te))

    model = Siniflandirici(onceden_egitilmis, ayar).to(cihaz)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.govde.parameters() if p.requires_grad], "lr": ayar.lr_govde},
        {"params": model.bas.parameters(), "lr": ayar.lr_bas},
    ], weight_decay=ayar.agirlik_curumesi)
    adim = ayar.epoch * int(np.ceil(len(tr) / ayar.toplu))
    zamanlayici = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[ayar.lr_govde, ayar.lr_bas], total_steps=adim,
                                                      pct_start=ayar.isinma_orani, anneal_strategy="linear")
    olcekleyici = torch.amp.GradScaler(enabled=cihaz.startswith("cuda"))
    kayip_fn = nn.CrossEntropyLoss()

    en_iyi = {"auc": -1.0, "epoch": -1, "durum": None}
    gecmis = []
    for ep in range(ayar.epoch):
        model.train()
        sira = rng.permutation(tr)
        toplam = 0.0
        for i in range(0, len(sira), ayar.toplu):
            j = sira[i:i + ayar.toplu]
            x = torch.from_numpy(X[j]).to(cihaz)
            hedef = torch.from_numpy(y[j]).long().to(cihaz)
            with torch.autocast(device_type=cihaz.split(":")[0], enabled=cihaz.startswith("cuda")):
                kayip = kayip_fn(model(x).float(), hedef)
            opt.zero_grad(set_to_none=True)
            olcekleyici.scale(kayip).backward()
            olcekleyici.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            olcekleyici.step(opt)
            olcekleyici.update()
            zamanlayici.step()
            toplam += kayip.item() * len(j)
        dog = olc(y[va], olasilik(model, X[va], cihaz))
        gecmis.append({"epoch": ep, "kayip": toplam / len(tr), **{f"ic_{k}": v for k, v in dog.items()}})
        if dog["auc"] > en_iyi["auc"]:
            en_iyi = {"auc": dog["auc"], "epoch": ep,
                      "durum": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(en_iyi["durum"])
    test = olc(y[te], olasilik(model, X[te], cihaz))
    return {"test_bitkisi": int(test_bitkisi), "onceden_egitilmis": onceden_egitilmis,
            "ic_dogrulama_bitkileri": sorted(int(b) for b in ic_dog), "en_iyi_epoch": en_iyi["epoch"],
            "n_egitim": int(len(tr)), "n_test": int(len(te)), **test, "gecmis": gecmis, "ayarlar": asdict(ayar)}


def lopo_calistir(X, y, gruplar, onceden_egitilmis: bool, cikti: Path, cihaz: str,
                  ayar: Ayarlar = AYARLAR, sinir: int | None = None, bitkiler=None) -> list[dict]:
    """Tüm LOPO katmanlarını çalıştırır; tamamlanmış katmanları (JSON varsa) atlar."""
    cikti.mkdir(parents=True, exist_ok=True)
    sonuclar = []
    for g in (bitkiler if bitkiler is not None else sorted(np.unique(gruplar))):
        dosya = cikti / f"bitki_{int(g):02d}.json"
        if dosya.exists():
            sonuclar.append(json.loads(dosya.read_text(encoding="utf-8")))
            print(f"  bitki {g}: önceden tamamlanmış, atlandı", flush=True)
            continue
        t0 = time.time()
        s = katman_egit(X, y, gruplar, int(g), onceden_egitilmis, cihaz, ayar, sinir)
        s["sure_sn"] = round(time.time() - t0, 1)
        dosya.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        sonuclar.append(s)
        print(f"  bitki {g}: doğruluk {s['dogruluk']:.3f}, AUC {s['auc']:.3f}, "
              f"en iyi epoch {s['en_iyi_epoch']}, {s['sure_sn']:.0f} s", flush=True)
    return sonuclar
