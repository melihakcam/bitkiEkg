# İlerleme Raporu — İnsan EKG'sinden Bitkiye

Son güncelleme: 2026-10-06 · Dal: `onDeneme` · Ayrıntılı sayılar: `docs/sonuclar.md`

## 1. Projenin sorusu
Domates bitkisinin gövdesinden ölçülen elektrik sinyaline bakarak bitkinin **sulama stresi**
altında olup olmadığını anlayabilir miyiz? İnsan EKG'siyle önceden eğitilmiş yapay zekâ
modelleri (HuBERT-ECG, ECG-FM) bu işi öğrenebilir mi? Sonuçlar **hiç görülmemiş bitkide**
(bitki-dışarıda-bırak, LOPO) ölçülür.

## 2. Veri
| Veri seti | İçerik | Kullanım |
|---|---|---|
| Domates (Zenodo 18876513) | 16 bitki, 18 gün, 1 Hz; ilk 3 gün sağlıklı, son 3 gün stresli | Ana deneyler (12 sulama bitkisi), 4 kontrol bitkisi zaman testi ve DAPT için |
| Sarmaşık (Zenodo 15095523) | 4 bitki, dış ortam, 4,5 ay, 2 kanal; hava durumu etiketleri | İkinci tür: şişme testi, DAPT verisi |
| Uyaran (Zenodo 7126105) | Zamioculcas, 5 uyaran | Yalnızca tanımlayıcı inceleme (ışık sınıflarında artefakt şüphesi) |

**Veri denetimi** (`scripts/veri_denetimi.py`): Her bitkinin son penceresi kısmen boş ve hepsi
"stresli" sınıfındaydı; modele kısayol olmasın diye %1'den fazla eksik pencereler çıkarıldı.
Bitki 7'de 40 saat düz sinyal, bitki 4 ve 13'te çok sayıda sıçrama bulundu (belgelendi).

## 3. Yapılan deneyler ve sonuçlar

### 3.1 Değerlendirme yöntemi sonucu şişiriyor (iki türde)
| Model | Rastgele bölme | Yazarların 2 bitkilik testi | Hiç görmediği bitki (LOPO) |
|---|---|---|---|
| tsfresh + LightGBM, 1 sa | 90,9 | 87,2 (yazarlar: 84,0) | **72,9** |
| tsfresh + LightGBM, 6 sa | 85,9 | 95,7 | **81,2** |
| MiniRocket, 1 sa | 79,7 | 81,9 | **72,5** |
| kNN / Naive Bayes, 1 sa | — | — | ~50 (şans) |

Sarmaşıkta (makro F1): rastgele bölme, görülmemiş bitkiye göre 3–9 puan, gelecek haftalara
göre 4–8 puan iyimser. → **Literatürdeki %90+ sonuçlar yeni bitkiye genellenmiyor.**

### 3.2 Zaman karıştırıcısı kontrolü
Hiç strese girmeyen kontrol bitkilerinde "ilk günler / son günler" ayrımı 30 dk'da şans düzeyinde
(%53,3), 1 sa'te %63,7. → Sinyalin ana kaynağı sulama stresi; küçük bir zaman etkisi var.

### 3.3 EKG modelleri: zaman ölçeği belirleyici
| Pencere | HuBERT donmuş, EKG ağırlıkları | Aynı model, rastgele ağırlıklar | Fark |
|---|---|---|---|
| 30 dk | 69,2 | 68,3 | +0,9 |
| 1 sa | 70,1 | 68,9 | +1,2 |
| **6 sa** | **77,9** | **72,8** | **+5,1** |

1 sa'te ince ayarlı HuBERT (%71,1) ve ECG-FM (%70,8) rastgele başlatılmış hallerinden farklı
değil; donmuş ECG-FM'de EKG ağırlıkları zararlı (%64,8 / %70,0).

### 3.4 DAPT 2×2 (6 sa, Colab)
Etiketsiz bitki verisiyle (4 400 pencere, test bitkileri hariç) ek ön eğitim denendi.

| Kol | Doğruluk | AUC |
|---|---|---|
| tsfresh + LightGBM | 81,2 | 86,7 |
| Donmuş, EKG + DAPT | 79,7 | 87,8 |
| Donmuş, EKG | 77,9 | 88,4 |
| İnce ayar, EKG + DAPT | 76,1 | 84,3 |
| İnce ayar, EKG | 74,6 | 84,2 |
| Donmuş, rastgele | 72,8 | 80,2 |
| İnce ayar, rastgele | 62,0 | 72,5 |

- **EKG ön eğitimi anlamlı:** ince ayarda +12,7 puan (p = 0,010, 12 bitkiden 10'unda).
- **DAPT ek katkı vermedi:** ~+1,5 puan, anlamsız.
- **En iyi EKG modeli LightGBM ile eşdeğer:** −1,4 puan, p = 0,66.

## 4. Ana bulgular (tek cümleyle)
1. Bitki sinyalinde rastgele bölmeyle raporlanan başarılar şişik; görülmemiş bitkide gerçek
   başarı %73–81.
2. İnsan EKG'si ile eğitilmiş modeller bitkiye **aktarılıyor, ama yalnızca 6 saatlik pencere
   modelin 5 saniyesine sıkıştırıldığında**.
3. EKG modelleri güçlü klasik modelle eşdeğer, onu geçmiyor.

## 5. Teknik notlar
- Tüm ayarlar deney öncesi sabitlendi; test bitkisine bakarak ayar yapılmadı. Tohum 42,
  model sürümleri sabit.
- İstatistik: 12 bitki üzerinde eşleştirilmiş Wilcoxon + bootstrap %95 güven aralığı.
- GPU deneyleri Colab'da (`notebooks/02–04`). Yerelde Windows Akıllı Uygulama Denetimi PyTorch'u
  engellediği için torch ana ortamdan kaldırıldı.

## 6. Belgeler
| Dosya | İçerik |
|---|---|
| `docs/sonuclar.md` | Bütün sonuç tabloları ve istatistikler |
| `docs/veri_notlari.md` | Veri setlerinin yapısı ve tuzakları |
| `docs/girdi_uyarlama_plani.md` | Bitki sinyalinin EKG modeline nasıl verildiği |
| `reports/faz1/faz1_rapor.md` | Faz 1 raporu (EDA, literatür) |
| `reports/faz2/makale_taslak.md` | Makale taslağı (Türkçe) |

## 7. Sıradaki adımlar (hedef: Q1 makale)
**Yeni Colab gerektirmeyenler:**
1. Frekans örtüşmesi analizi: 6 sa'in neden işe yaradığının açıklaması.
2. Hibrit model: EKG gömmeleri + tsfresh öznitelikleri.
3. Öznitelik önemi: susuz bitkide sinyalde ne değişiyor.
4. Zor bitkilerin (ör. bitki 11) incelenmesi.
5. İngilizce makale ve grafikler.

**Colab gerektirenler (makaleyi güçlendirir):**
- Başka ön eğitimli modellerle karşılaştırma (ses: wav2vec2/HuBERT; zaman serisi: MOMENT).
- Ek ölçekler (2, 3, 12 sa) ve farklı tohumlarla tekrar.
- EKG modelinin sarmaşıkta denenmesi.
