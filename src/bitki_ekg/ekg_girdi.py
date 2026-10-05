"""Bitki penceresini EKG temel modellerinin girdi biçimine dönüştürme.

HuBERT-ECG girdisi (yazarların `hubert_ecg/dataset.py` kodundan doğrulandı):
12 derivasyon × 2 500 örnek (5 s, 500 Hz) → derivasyonlar uç uca eklenir (12 × 2 500)
→ 5 kat seyreltilir → **6 000 örnek = 12 derivasyon × 500 örnek (100 Hz)**.

Bitki penceresi (ör. 1 sa = 3 600 örnek, 1 Hz) önce derivasyon uzunluğuna (500) kenar
yumuşatmalı yeniden örneklenir; böylece pencere süresi modelin "5 saniyesine" sıkıştırılır
(gerekçe: docs/girdi_uyarlama_plani.md §2).
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

HUBERT_DERIVASYON = 12
HUBERT_DERIVASYON_UZUNLUK = 500  # 5 s × 100 Hz
HUBERT_GIRDI = HUBERT_DERIVASYON * HUBERT_DERIVASYON_UZUNLUK  # 6 000

KANAL_ESLEME = ("tekrar", "tek", "parca")

# ECG-FM girdisi (bowang-lab/ECG-FM infer_quickstart.ipynb'den doğrulandı):
# (B, 12, 2 500) = 12 derivasyon × 5 s × 500 Hz, derivasyon başına standartlaştırma.
ECGFM_DERIVASYON = 12
ECGFM_UZUNLUK = 2500


def yeniden_ornekle(X: np.ndarray, hedef: int) -> np.ndarray:
    """(n, L) → (n, hedef); kenar yumuşatmalı polifaz yeniden örnekleme."""
    L = X.shape[1]
    if L == hedef:
        return X.astype(np.float32)
    g = gcd(hedef, L)
    return resample_poly(X, hedef // g, L // g, axis=1).astype(np.float32)


def hubert_girdisi(Z: np.ndarray, esleme: str = "tekrar") -> np.ndarray:
    """Normalize edilmiş bitki pencerelerini (n, L) HuBERT-ECG girdisine (n, 6000) çevirir.

    esleme:
      tekrar — pencere 500 örneğe indirilir ve 12 derivasyona kopyalanır (varsayılan)
      tek    — pencere yalnızca derivasyon II'ye (2. sıra) konur, diğerleri sıfır
      parca  — pencere 6 000 örneğe yeniden örneklenir; ardışık 12 parça 12 derivasyon olur
    """
    if esleme not in KANAL_ESLEME:
        raise ValueError(f"esleme {KANAL_ESLEME} içinden olmalı: {esleme}")
    if esleme == "parca":
        return yeniden_ornekle(Z, HUBERT_GIRDI)
    d = yeniden_ornekle(Z, HUBERT_DERIVASYON_UZUNLUK)
    if esleme == "tekrar":
        return np.tile(d, (1, HUBERT_DERIVASYON))
    cikti = np.zeros((len(d), HUBERT_GIRDI), dtype=np.float32)
    cikti[:, HUBERT_DERIVASYON_UZUNLUK:2 * HUBERT_DERIVASYON_UZUNLUK] = d
    return cikti


def ecgfm_girdisi(Z: np.ndarray) -> np.ndarray:
    """Bitki pencerelerini (n, L) ECG-FM girdisine (n, 12, 2500) çevirir.

    Pencere 2 500 örneğe yeniden örneklenir, z-skorla standartlaştırılır (ECG-FM'in
    `Standardize` dönüşümüyle aynı) ve 12 derivasyona kopyalanır.
    """
    d = yeniden_ornekle(Z, ECGFM_UZUNLUK)
    d = (d - d.mean(axis=1, keepdims=True)) / np.maximum(d.std(axis=1, keepdims=True), 1e-6)
    return np.repeat(d[:, None, :], ECGFM_DERIVASYON, axis=1).astype(np.float32)
