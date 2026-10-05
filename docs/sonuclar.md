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

## 2. HuBERT-ECG doğrusal sonda (ön test, GPU'suz)

`python scripts/hubert_sonda.py` → `hubert_sonda_katman.csv`, `hubert_sonda_ozet.csv`

- Model: `Edoardo-Coppola/hubert-ecg-small` (30,5 M parametre), sürüm `eca1c5a…` sabit.
- Girdi: robust z-skorlu pencere → 500 örneğe yeniden örnekleme → 12 derivasyona kopyalama
  (6 000 örnek). Model **eğitilmez**; her katmanın zaman ortalaması gömme olarak alınır,
  üstüne lojistik regresyon (StandardScaler + C=1).
- Kontrol: aynı mimari, rastgele ağırlıklar. Gömme çıkarma CPU'da 87–333 s.

### LOPO (12 bitki), ortalama ± std (%)
| Pencere | Model | Özellik | Doğruluk | AUC |
|---|---|---|---|---|
| 1 sa | Önceden eğitilmiş | tüm katman ort. | 70,1 ± 10,0 | 78,9 ± 11,2 |
| 1 sa | Önceden eğitilmiş | son katman | 69,4 ± 10,6 | 78,2 ± 11,4 |
| 1 sa | Rastgele başlatılmış | tüm katman ort. | 68,9 ± 11,5 | 74,4 ± 13,5 |
| 1 sa | Rastgele başlatılmış | son katman | 70,1 ± 10,9 | 75,3 ± 13,3 |
| 30 dk | Önceden eğitilmiş | tüm katman ort. | 69,2 ± 10,8 | 78,1 ± 12,4 |
| 30 dk | Rastgele başlatılmış | tüm katman ort. | 68,3 ± 10,1 | 74,2 ± 12,0 |

Karşılaştırma (LOPO): LightGBM %72,9 / AUC %80,7 · MiniRocket %72,5.

### Yorum
- Donmuş EKG gömmeleri görülmemiş bitkide **%69–70** doğruluk veriyor; temel modellerin
  (%72–73) biraz altında.
- Önceden eğitilmiş ve rastgele başlatılmış gömmeler arasında doğrulukta fark **yok denecek kadar
  az** (+1,0–1,2 puan; bitkilerin yalnızca 6/12'sinde önde). **AUC'de ~4 puanlık tutarlı üstünlük**
  var (78,9'a karşı 74,4) → ön eğitim sıralama bilgisini biraz iyileştiriyor ama karar sınırını
  belirgin biçimde değiştirmiyor.
- Rastgele ağırlıklı bir ağın bile %69 vermesi, ROCKET'in mantığıyla uyumlu: rastgele
  konvolüsyonlar bu sinyalde zaten işe yarar özellik çıkarıyor.
- Katman taraması (keşif): önceden eğitilmiş modelde erken katmanlar (0–3) biraz daha iyi
  (%71); fark küçük.
- **Sonuç:** Donmuş haliyle EKG ön eğitimi belirgin bir kazanç sağlamıyor. Asıl hipotez testi
  ince ayardır (Colab, `notebooks/02_colab_hubert_ince_ayar.ipynb`); beklenti ılımlı tutulmalı.

## 3. Zaman karıştırıcısı testi

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
