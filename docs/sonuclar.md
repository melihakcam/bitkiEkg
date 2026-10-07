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

### Düzeltme (2026-10-05)
MiniRocket AUC'si ilk çalıştırmada 0/1 olasılıklardan hesaplanıyordu (doğrulukla aynı çıkıyordu);
ridge karar fonksiyonuyla düzeltildi ve `n_jobs=-1` eklendi. Diğer modellerin sonuçları aynı kaldı.
MiniRocket düzeltilmiş AUC: LOPO 79,5 (1 sa ve 30 dk) · yazar 89,5 / 89,8 · rastgele 88,0 / 86,8.

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

## 3. HuBERT-ECG ince ayarı (Colab, T4 GPU)

`notebooks/02_colab_hubert_ince_ayar.ipynb` → `results/colab/ozet_1h.csv`
(bitki başına JSON ve epoch geçmişi Drive'da `bitkiEkg_sonuclar/1h/`)

- Ayarlar (deney öncesi sabit): 15 epoch, toplu 32, AdamW (gövde 3e-5, baş 1e-3), ağırlık
  çürümesi 0,01, %10 ısınma + doğrusal azalma, CNN öznitelik çıkarıcı dondurulmuş,
  mask_time_prob 0,05; her katmanda 2 eğitim bitkisi iç doğrulama, en iyi epoch iç doğrulama
  AUC'siyle seçildi. Bitki başına ~50–70 s (T4). Yalnızca 1 sa pencere (30 dk kısmı GPU kotası
  için durduruldu).

| Deney | Doğruluk | F1 | AUC | En iyi epoch (ortanca) |
|---|---|---|---|---|
| Önceden eğitilmiş | 71,1 ± 8,9 | 69,6 ± 13,6 | 79,4 ± 11,5 | 4 (aralık 0–12) |
| Rastgele başlatılmış | 69,2 ± 11,6 | 65,3 ± 18,6 | 77,0 ± 16,2 | 12 (aralık 0–14) |

## 4. İstatistiksel karşılaştırma (1 sa, LOPO, 12 bitki)

`python scripts/karsilastir.py --pencere 1h` → `karsilastirma_1h.csv`, `bitki_bazinda_1h.csv`,
`results/figures/karsilastirma_1h.png`. Referans: tsfresh + LightGBM. Fark = model − referans
(puan), %95 bootstrap GA, eşleştirilmiş Wilcoxon.

| Model | Doğruluk | Fark [%95 GA] | p | AUC |
|---|---|---|---|---|
| tsfresh + LightGBM | 72,9 | — | — | 80,7 |
| MiniRocket | 72,5 | −0,3 [−5,0; 4,6] | 0,99 | 79,5 |
| HuBERT ince ayar (önceden eğitilmiş) | 71,1 | −1,8 [−6,6; 2,8] | 0,44 | 79,4 |
| HuBERT donmuş (önceden eğitilmiş) | 70,1 | −2,7 [−6,5; 1,0] | 0,12 | 78,9 |
| HuBERT ince ayar (rastgele) | 69,2 | −3,7 [−11,5; 3,9] | 0,41 | 77,0 |
| HuBERT donmuş (rastgele) | 68,9 | −3,9 [−9,8; 1,2] | 0,34 | 74,4 |
| Naive Bayes | 50,6 | −22,2 [−28,4; −16,4] | <0,001 | 52,6 |
| kNN | 50,5 | −22,4 [−29,6; −15,6] | <0,001 | 50,7 |

Ön eğitimin etkisi (önceden eğitilmiş − rastgele, aynı mimari):
- İnce ayar: doğruluk +1,9 [−2,0; 6,1], p = 0,61 · AUC +2,4 [−2,3; 8,0], p = 0,79
- Donmuş: doğruluk +1,2 [−2,1; 4,6], p = 0,66 · **AUC +4,5 [1,6; 7,7], p = 0,016** (9/12 bitki)

### Yorum
- LightGBM, MiniRocket ve HuBERT-ECG (ince ayar) arasında **istatistiksel olarak anlamlı fark
  yok**; üçü de görülmemiş bitkide ~%71–73. Yalnızca kNN ve NB anlamlı biçimde kötü.
- **EKG ön eğitimi doğruluğu anlamlı artırmıyor.** Tek anlamlı etki donmuş gömmelerde AUC
  (+4,5, p = 0,016; düzeltilmemiş p, 18 karşılaştırma içinde — Bonferroni sonrası anlamlı değil).
- Ön eğitim **daha hızlı yakınsama** sağlıyor (en iyi epoch ortancası 4'e karşı 12) ve
  bitkiler arası değişkenliği azaltıyor (std 8,9'a karşı 11,6).
- 12 bitkiyle güç düşük: ±5 puanlık farklar ayırt edilemiyor. Bu bir kısıt olarak raporlanmalı.
- Çıkarım: bu veri ölçeğinde EKG temel modeli bitki sinyaline **aktarılabiliyor ama üstünlük
  sağlamıyor** (Wahid vd. 2026'nın hipotezine kısmi, olumsuz yönde ilk deneysel kanıt).

## 5. ECG-FM (Colab, `notebooks/03_colab_kalan_deneyler.ipynb`, 1 sa, LOPO)

Sonuçlar: `results/colab/ecgfm/sonda_1h.csv` (donmuş), `results/colab/ecgfm/ozet_1h.csv`
(ince ayar; Drive JSON'larından düz metin okunup `dogruluk × 144` tam sayı kontrolüyle doğrulandı).
ECG-FM önceden eğitilmiş (90,9 M), girdi 12 × 2 500, ince ayar ayarları HuBERT ile aynı;
bitki başına ~3,6 dk (T4).

| Deney | Doğruluk | AUC | En iyi epoch (ortanca) |
|---|---|---|---|
| ECG-FM ince ayar, önceden eğitilmiş | 70,8 ± 12,5 | 78,9 ± 13,6 | 7,5 |
| ECG-FM ince ayar, rastgele | 70,4 ± 11,4 | 77,2 ± 13,6 | 9,5 |
| ECG-FM donmuş, önceden eğitilmiş | 64,8 ± 11,9 | 72,8 ± 13,9 | — |
| ECG-FM donmuş, rastgele | 70,0 ± 8,7 | 76,7 ± 11,6 | — |

Ön eğitimin etkisi (önceden eğitilmiş − rastgele, eşleştirilmiş, 12 bitki):
- İnce ayar: doğruluk +0,5 [−2,5; 3,5], p = 0,90 · AUC +1,7 [−1,3; 5,0], p = 0,20
- Donmuş: doğruluk **−5,2 [−9,1; −1,6]**, p = 0,058 (9/12 bitkide kötü) · AUC −3,8, p = 0,11

LightGBM'e göre ECG-FM ince ayar: doğruluk −2,0 [−5,6; 1,9], p = 0,44 (anlamlı fark yok).

**Yorum:** İkinci ve daha büyük EKG temel modeli de aynı tabloyu veriyor: ince ayarla görülmemiş
bitkide ~%71, temel modellerle istatistiksel olarak eşdeğer, EKG ön eğitiminin anlamlı katkısı
yok. Donmuş temsillerde EKG ön eğitimi rastgele özelliklerden kötü → EKG'ye özgü özellikler
bitki sinyaline uymuyor. Bulgu tek bir modele özgü değil.

## 6. Zaman ölçeği taraması (HuBERT donmuş, LOPO)

`python scripts/olcek_taramasi.py` → `olcek_taramasi_{katman,ozet}.csv`. Eksik son pencereler
(%1'den fazla boş) çıkarıldı (`scripts/veri_denetimi.py`; 6 sa'te her bitkinin son penceresi
%34 boş ve "stresli" idi — filtre öncesi 6 sa sonucu %80,6 idi, filtreyle %77,9).

| Pencere | Önceden eğitilmiş | Rastgele | Fark (doğruluk) | Fark (AUC) |
|---|---|---|---|---|
| 5 dk (seyreltilmiş) | 67,1 | — | — | — |
| 30 dk | 69,2 | 68,3 | +0,9 | +3,9 |
| 1 sa | 70,1 | 68,9 | +1,2 | +4,5 |
| **6 sa** | **77,9** | **72,8** | **+5,1** | **+8,2** |

6 sa temel modeller (LOPO): LightGBM %81,2 (AUC 86,7) · MiniRocket %76,1 (87,1) ·
NB %62,7 · kNN %54,3. Rastgele bölmede LightGBM %85,9, yazar bölmesinde %95,7.

## 7. DAPT 2×2 deneyi (6 sa, `notebooks/04_colab_dapt.ipynb`)

DAPT: etiketsiz 4 400 pencere (kontrol domates 572 + sarmaşık 3 828; test edilen 12 bitki yok),
maskeli yeniden yapılandırma, 10 epoch; kayıp 0,71→0,13 (EKG başlangıç), 0,73→0,20 (rastgele).
Sonuçlar `results/colab/dapt/` (Drive'dan düz metin, `doğruluk×23` tam sayı kontrolü).

| Kol (LOPO, 12 bitki) | Doğruluk | AUC |
|---|---|---|
| tsfresh + LightGBM | **81,2 ± 17,5** | 86,7 |
| Donmuş, EKG + DAPT | 79,7 ± 10,7 | 87,8 |
| Donmuş, EKG | 77,9 ± 13,0 | **88,4** |
| İnce ayar, EKG + DAPT | 76,1 ± 15,9 | 84,3 |
| MiniRocket | 76,1 ± 16,3 | 87,1 |
| İnce ayar, EKG | 74,6 ± 13,3 | 84,2 |
| Donmuş, rastgele + DAPT | 73,2 ± 12,4 | 79,2 |
| Donmuş, rastgele | 72,8 ± 15,2 | 80,2 |
| İnce ayar, rastgele + DAPT | 66,3 ± 12,6 | 73,5 |
| İnce ayar, rastgele | 62,0 ± 7,7 | 72,5 |

Eşleştirilmiş karşılaştırmalar (fark, %95 bootstrap GA, Wilcoxon):
- **EKG ön eğitiminin etkisi (6 sa):** ince ayar +12,7 [+5,1; +19,9] p = 0,010 (10/12 bitki);
  ince ayar + DAPT AUC +10,8 [+3,3; +17,9] p = 0,027; donmuş + DAPT +6,5 p = 0,049;
  donmuş AUC +8,1 p = 0,012 → **EKG ön eğitimi 6 sa ölçekte anlamlı ve tutarlı katkı sağlıyor.**
- **DAPT'ın etkisi:** EKG + DAPT − EKG: ince ayar +1,4 (p = 0,65), donmuş +1,8 (p = 0,52)
  → **anlamlı ek katkı yok.**
- **LightGBM'e göre:** en iyi EKG kolu (donmuş + DAPT) −1,4 [−11,2; +9,4] p = 0,66 → istatistiksel
  olarak eşdeğer; LightGBM geçilemiyor.

**Sonuç:** İnsan EKG ön eğitimi bitki sinyaline **doğru zaman ölçeğinde (6 sa → 5 s) aktarılıyor**
ve aynı mimarinin rastgele başlatılmışına göre 5–13 puan kazandırıyor; 30 dk–1 sa ölçeğinde
bu etki yok. Etiketsiz bitki verisiyle ek uyarlama (DAPT) katkı sağlamıyor. EKG tabanlı modeller
güçlü öznitelik tabanlı temel modelle eşdeğer, onu geçmiyor (12 bitkiyle ±10 puanlık GA).

## 8. Sarmaşık: değerlendirme yöntemine göre şişme (ikinci tür)

`python scripts/sarmasik_sizinti.py` → `sarmasik_sizinti_{katman,ozet}.csv` (v2: bitki ve kanal
başına z-skor, 10 sn seyreltme). v1 (pencere başına z-skor, seviye bilgisini sildiği için
geçersiz): `sarmasik_sizinti_v1_pencere_z_*.csv`. 6 089 saatlik pencere, 4 bitki, 2 kanal.

| Görev | Model | Rastgele | LOPO | Zaman bloğu | AUC (rastgele / LOPO) |
|---|---|---|---|---|---|
| Gündüz/gece | Öznitelik + LightGBM | 76,7 | 67,4 | 68,4 | 85,5 / 74,7 |
| Gündüz/gece | MiniRocket | 76,1 | 70,6 | 71,2 | 84,8 / 77,9 |
| Yağmurlu/kuru | Öznitelik + LightGBM | 72,6 | 65,4 | 66,2 | 89,8 / 84,3 |
| Yağmurlu/kuru | MiniRocket | 70,4 | 67,0 | 64,5 | 87,7 / 83,7 |

(makro F1, %). Rastgele bölme LOPO'ya göre 3–9 puan, zaman bloğuna göre 4–8 puan iyimser → şişme
bulgusu ikinci türde ve dış ortamda da geçerli.

## 9. Zaman karıştırıcısı testi

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

## 10. Aşama 1: tekrar ve zaman karıştırıcısı testleri (6 sa, 2026-10-07)

Not defterleri: `notebooks/05_colab_asama1.ipynb`, `notebooks/06_colab_zaman_testleri.ipynb`;
yerel: `scripts/zaman_kontrolu.py`, `scripts/zaman_testleri.py` (`src/bitki_ekg/zaman_testleri.py`).
Sonuçlar: `results/colab/asama1/`, `results/tables/zaman_*`.

### 10.1 Tekrar (5 farklı başlangıç; ince ayar, LOPO, 12 bitki)
| Kol | Doğruluk | AUC | Tek sınıfa çöken bitki (60 katman) |
|---|---|---|---|
| EKG ağırlıkları | 77,0 (74,3–79,7) | 85,7 | 1 |
| Rastgele ağırlıklar | 63,9 (60,1–67,0) | 69,5 | 1 |

EKG − rastgele (bitki başına 5 tekrar ortalaması): doğruluk **+13,1, p = 0,0024 (10/12 bitki)**;
AUC **+16,1, p = 0,0010 (11/12)**. Fark 5 tekrarın hepsinde pozitif (+8,7 … +17,4). Donmuşta da
EKG (77,9 / AUC 88,4) 5 rastgele başlangıcın hepsinden iyi (63,0–71,7 / 68,9–74,9).
→ **EKG ön eğitiminin etkisi şans değil.**

### 10.2 Zaman karıştırıcısı testleri
| Test (olması gereken) | LightGBM | EKG donmuş | Rastgele donmuş (5 ort.) |
|---|---|---|---|
| Kontrol bitkisi ilk/son gün, AUC (~0,5) | 0,76 (4/4 bitki > 0,5) | **0,95** (4/4, hepsi p < 0,001) | 0,76 |
| Stres modeli → kontrol bitkisi, AUC (~0,5) | 0,81 (4/4) | **0,82** (4/4, hepsi p ≤ 0,02) | 0,83 |
| Son 3 gün sulama/kontrol grubu, cihaz permütasyon p | 0,036 | 0,25 | 0,20 |
| İlk 3 gün (tedavi öncesi, plasebo), p | 0,071 | 0,29 | 0,92 |
| Kendi başlangıcına göre fark, p | 0,29 | 0,57 | 0,74 |

(Cihaz-dışarıda-bırak; 8 cihaz, kontrol bitkileri yalnızca PN8/PN9 → C(8,2) = 28 permütasyon, en
küçük p = 0,036. LODO'da havuzlanmış AUC aşağı yanlı olduğundan karşılaştırma permütasyon p ile yapılır.)

**Sonuç:**
1. Tüm modellerin "stres" kararı büyük ölçüde **deneyin başı/sonu farkından** (zaman: elektrot,
   bitki gelişimi, sera koşulları) geliyor: stres modeli hiç stres görmemiş bitkilerin son günlerine
   de "stresli" diyor (AUC ~0,82, 4/4 bitki, anlamlı).
2. Zaman ve cihaz etkisi ayıklandığında (aynı gün, fark testleri) **hiçbir model stres sinyali
   gösteremiyor** (EKG p = 0,25 / 0,57). Sınır: yalnızca 2 kontrol cihazı → güç düşük.
3. EKG ön eğitimi bu yavaş zamansal değişimi rastgele modelden çok daha iyi yakalıyor
   (kontrol AUC 0,95 / 0,76). Yani aktarılan bilgi gerçek, ama bu veride stresle değil zamanla ilgili.
4. Yazarların etiketi (ilk 3 gün sağlıklı / son 3 gün stresli) bu karıştırıcıyı içeriyor; rastgele
   bölme ve LOPO bunu yakalayamıyor.

## 11. Doz-etki testi (2026-10-07)

`python scripts/doz_etki.py` → `results/tables/doz_etki*_{bitki,ozet}.csv`. Grup eşlemesi: docs/veri_notlari.md.
Zamana karşı dengelenmiş eğitim (kontrol son günleri = sağlıklı), cihaz-dışarıda-bırak; model dozu görmez.
İstatistik: kontrol < 200 mL < 100 mL sırası ile bitki Δ'sı (son 3 gün − ilk 3 gün skor) arasında Spearman r;
cihaz düzeyinde 90 atama, her birinde yeniden eğitim (en küçük p = 0,011).

| Model (6 sa) | r | p (cihaz perm.) | Δ kontrol | Δ 200 mL | Δ 100 mL | Δ aşırı sulama |
|---|---|---|---|---|---|---|
| **HuBERT-ECG donmuş (EKG ağırlıkları)** | **0,65** | **0,011** | 0,30 | 0,29 | **0,59** | 0,51 |
| Aynı model, rastgele ağırlık t1 | 0,18 | 0,26 | 0,28 | 0,18 | 0,33 | 0,44 |
| t2 | −0,44 | 0,82 | 0,34 | 0,07 | 0,21 | 0,31 |
| t3 | 0,03 | 0,36 | 0,32 | 0,09 | 0,34 | 0,51 |
| t4 | −0,38 | 0,80 | 0,37 | 0,02 | 0,22 | 0,30 |
| t5 | −0,18 | 0,57 | 0,35 | 0,15 | 0,27 | 0,30 |
| tsfresh + LightGBM | 0,15 | 0,29 | 0,14 | 0,33 | 0,28 | 0,39 |

**Sonuç:** Doz-etki ilişkisi (daha az su → daha yüksek stres skoru) yalnızca EKG ön eğitimli modelde var ve
olası en küçük p değerine ulaşıyor; aynı mimarinin 5 rastgele başlangıcının hiçbirinde ve LightGBM'de yok.
→ Zamandan bağımsız stres bilgisini çıkaran şey EKG ön eğitimi.
Sınırlar: test, grup eşlemesi bulunduktan sonra tasarlandı (keşif); etkiyi esas olarak 100 mL grubu taşıyor
(200 mL ≈ kontrol). Doğrulama için önceden kayıtlı testler: docs/on_kayit.md.
