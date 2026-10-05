# Sonuçlar

Tüm sayılar `results/tables/` altındaki CSV dosyalarından alınmıştır; üretmek için gereken
komutlar her bölümde verilmiştir. Rastgelelik tohumu 42.

## 1. Temel modeller — domates, ikili görev (sağlıklı / sulama stresi)

`python scripts/temel_modeller.py` → `temel_modeller_katman.csv`, `temel_modeller_ozet.csv`
(2026-10-05, Intel i5-13420H, GPU yok)

### Kurulum
| Model | Girdi | Ayarlar | Eğitim süresi (18 bölme, 1 sa / 30 dk) |
|---|---|---|---|
| kNN | robust z-skorlu ham pencere | k = 5 | 4 s / 4 s |
| Naive Bayes | robust z-skorlu ham pencere | Gauss | 2 s / 2 s |
| tsfresh + LightGBM | yazarların ~780 tsfresh özniteliği | 300 ağaç, öğrenme oranı 0,05 | 91 s / 149 s |
| MiniRocket | robust z-skorlu ham pencere | aeon `MiniRocketClassifier` varsayılan | 386 s / 462 s |

- Veri: 12 sulama bitkisi; 1 sa → 1 728 pencere (bitki başına 144), 30 dk → 3 456 pencere
  (bitki başına 288); sınıflar dengeli.
- Bölmeler: **LOPO** (bitki-dışarıda-bırak, 12 katman — asıl değerlendirme), **yazar**
  (Buss vd. 2026: bitki 0 ve 12 test), **rastgele** (tabakalı 5 katlı pencere bölmesi).

### Doğruluk (%) — ortalama ± std
| Model | LOPO 1 sa | LOPO 30 dk | Yazar 1 sa | Yazar 30 dk | Rastgele 1 sa | Rastgele 30 dk |
|---|---|---|---|---|---|---|
| kNN | 50,5 ± 4,3 | 52,5 ± 2,9 | 45,5 | 51,6 | 52,1 ± 3,5 | 52,0 ± 1,1 |
| Naive Bayes | 50,6 ± 2,7 | 50,7 ± 3,3 | 51,0 | 51,2 | 50,4 ± 2,1 | 50,3 ± 0,6 |
| tsfresh + LightGBM | **72,9 ± 11,3** | 70,5 ± 11,4 | **87,2** | **83,9** | **90,9 ± 0,9** | **92,2 ± 1,2** |
| MiniRocket | 72,5 ± 13,2 | **73,1 ± 12,4** | 81,9 | 83,3 | 79,7 ± 1,5 | 78,9 ± 1,4 |

### ROC-AUC (%) — LightGBM
LOPO 80,7 ± 15,4 (1 sa), 79,3 ± 15,2 (30 dk) · yazar 94,6 / 91,9 · rastgele 96,6 / 97,6.

### Bulgular
1. **Boru hattı doğrulandı:** yazarların bölmesinde LightGBM 1 sa %87,2 ve 30 dk %83,9;
   Buss vd. (2026) HGB test doğruluğu 1 sa %84,0 ve 30 dk %83,2 → aynı düzeyde.
2. **Rastgele bölme sonuçları şişiriyor:** LightGBM rastgele bölmede %91–92, görülmemiş bitkide
   (LOPO) %71–73 → **~20 puan fark**. Yazarların 2 bitkilik testi de (%84–87) LOPO
   ortalamasının üzerinde; 2 bitki şanslı bir seçim olabilir.
3. **Bitkiden bitkiye büyük fark:** LOPO'da bitki başına doğruluk %51 ile %94 arasında
   (std ±11–13). Bitki 1 ve 13 neredeyse tahmin edilemiyor (%51–60).
4. **Uygun olmayan modeller beklendiği gibi şans düzeyinde** (~%50): kNN ve Naive Bayes ham
   pencerede bitkiler arası genlik/faz farklarını aşamıyor; NB çoğunlukla tek sınıf tahmin
   ediyor (F1 %12–18).
5. LightGBM ve MiniRocket LOPO'da eşdeğer (~%72–73); rastgele bölmede LightGBM çok daha iyi
   görünüyor → öznitelik tabanlı model pencereler arası benzerliği (sızıntı) daha çok kullanıyor.

### LOPO bitki başına doğruluk (%), 1 sa
| Bitki | 0 | 1 | 2 | 3 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | 94 | 60 | 72 | 81 | 74 | 71 | 74 | 69 | 79 | 57 | 58 | 85 |
| MiniRocket | 85 | 53 | 83 | 93 | 81 | 76 | 61 | 63 | 79 | 51 | 67 | 78 |

### Bilinen eksikler (sonraki çalıştırmada düzeltilecek)
- MiniRocket'in `predict_proba` çıktısı 0/1 olduğu için AUC'si doğrulukla aynı çıkıyor → ridge
  karar fonksiyonundan AUC hesaplanmalı.
- MiniRocket dönüşümü her bölmede yeniden hesaplanıyor ve çekirdeklerin hepsini kullanmıyor
  → `n_jobs=-1` ve dönüşüm önbelleği ile hızlandırılabilir.

## 2. Zaman karıştırıcısı testi

`python scripts/zaman_kontrolu.py` → `zaman_kontrolu_katman.csv`, `zaman_kontrolu_ozet.csv`

Aynı model (tsfresh + LightGBM, LOPO) ile "ilk 3 gün / son 3 gün" ayrımı:

| Grup | 1 sa doğruluk | 1 sa AUC | 30 dk doğruluk | 30 dk AUC |
|---|---|---|---|---|
| Kontrol (4 bitki, hiç stres yok) | 63,7 ± 20,6 | 65,0 ± 27,2 | 53,3 ± 15,9 | 49,3 ± 25,7 |
| Sulama (12 bitki) | 72,9 ± 11,3 | 80,7 ± 15,4 | 70,5 ± 11,4 | 79,3 ± 15,2 |

Kontrol bitkisi başına (1 sa / 30 dk): bitki 4 %78 / %47 · bitki 5 %50 / %50 ·
bitki 6 %84 / %76 · bitki 7 %42 / %40.

**Yorum:**
- 30 dk pencerede kontrol bitkileri **şans düzeyinde** (%53, AUC 0,49) → sulama bitkilerindeki
  %70,5'lik ayrım zamandan değil sulama stresinden kaynaklanıyor.
- 1 sa pencerede kontrolde de kısmi ayrım var (%64), ama 4 bitkiden yalnızca 2'sinde (4 ve 6);
  sulama bitkileriyle fark ~9 puan (AUC'de ~16 puan).
- Sonuç: **sinyalin ana kaynağı sulama stresi, ancak zaman etkisi sıfır değil.** Yalnızca 4
  kontrol bitkisi olduğundan istatistiksel güç düşük; bu bir kısıt olarak raporlanmalı.
- Bu test Buss vd. (2026)'da yapılmamış → makaleye özgün katkı.
