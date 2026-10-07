# Dış İnceleme Promptu (başka bir yapay zekâya verilecek)

Aşağıdaki metnin tamamını kopyalayıp ver. Tarih: 2026-10-07.

---

## ROLÜN

Makine öğrenmesi, biyosinyal işleme ve tarımsal sensörler konusunda deneyimli, **Q1 dergi
hakemi** gibi davran. Aşağıdaki araştırmayı acımasız ama yapıcı biçimde değerlendir. Övgüye
değil, **zayıf noktalara, hatalara ve eksiklere** odaklan. Bir iddia verilen sayılarla
desteklenmiyorsa söyle. Bilmediğin bir şeyi uydurma; emin olmadığın yerde "emin değilim" de.
Kaynakları kontrol edebiliyorsan kontrol et; künyesi yanlış ya da var olmayan bir kaynak görürsen
belirt. Cevabını Türkçe ver.

## 1. ARAŞTIRMANIN AMACI

**Soru:** İnsan EKG'si ile bitki gövdesinden ölçülen elektrik potansiyeli arasında, yapay zekâ
modelinin kullanabileceği bir ortaklık var mı? İnsan EKG'siyle önceden eğitilmiş temel modeller
(HuBERT-ECG, ECG-FM), domateste **sulama stresini, eğitimde hiç görülmemiş bitkide** tespit
edebilir mi?

**Hedef:** Q1 dergi makalesi (aday dergiler: Computers and Electronics in Agriculture, Plant
Methods, Biosystems Engineering, Smart Agricultural Technology).

**Öne sürülen katkılar:**
1. İnsan EKG temel modellerinin bitki elektrofizyolojisine aktarılmasının ilk deneysel testi
   (Wahid vd. 2026 derlemesi bunu "doğrulanmamış hipotez" olarak bırakıyor).
2. Aktarımın yalnızca belirli bir zaman ölçeğinde (6 saatlik pencere) çalıştığının gösterilmesi.
3. Rastgele pencere bölmesinin sonuçları ne kadar şişirdiğinin, aynı veride ve iki türde
   (domates, sarmaşık) yan yana ölçülmesi.
4. Zaman karıştırıcısı (time confound) kontrolü: stres görmeyen kontrol bitkileriyle.

## 2. VERİ

| Veri seti | İçerik | Kullanım |
|---|---|---|
| Buss vd. 2026, Zenodo 10.5281/zenodo.18876513 | Sera, 16 domates, 18 gün (04–21.06.2025), cihazda 10 Hz, veri setinde 1 Hz, bitki başına 1 kanal (gövdede iki gümüş elektrot). 4 grup × 4 bitki: kontrol 400 mL/gün, aşırı sulama, orta kuraklık 200 mL, şiddetli kuraklık 100 mL. İlk 5 gün herkes 400 mL. | Ana deneyler |
| Buss vd. 2025, Zenodo 10.5281/zenodo.15095523 | Sarmaşık (Hedera helix), 4 bitki, dış ortam, ~4,5 ay, 1 Hz, 2 kanal; hava durumu verisi | İkinci tür: şişme testi; DAPT için etiketsiz veri |
| Buss vd. 2023, Zenodo 10.5281/zenodo.7126105 | Zamioculcas, 5 uyaran (rüzgâr, ısı, ışık…) | Yalnızca tanımlayıcı inceleme (ışık sınıflarında artefakt şüphesi) |

**Etiket (domates, ikili görev):** yazarların tanımı — her bitkinin **ilk 3 günü "sağlıklı",
son 3 günü "stresli"**. Kullanılan 12 bitki = sulama grubu bitkileri (kontrol grubundaki 4 bitki
stres görmediği için sınıflandırmada değil, zaman karıştırıcısı testinde ve etiketsiz DAPT
verisinde kullanıldı). Sınıflar dengeli. Not: aşırı sulanan grup da "stresli" sayılıyor; bu
yüzden "su stresi" yerine "sulama stresi" diyoruz.

**Veri denetimi:** Her bitkinin son penceresi kısmen boş ve hepsi "stresli" sınıfındaydı
(6 sa'te %34 boş). Kısayol olmasın diye %1'den fazla eksik pencereler çıkarıldı. Filtre öncesi
HuBERT 6 sa sonucu %80,6, sonrası %77,9. Bitki 7'de 40 saat düz sinyal, bitki 4 ve 13'te çok
sayıda sıçrama var.

**Pencere sayıları (12 bitki):** 1 sa → 1 728 (bitki başına 144); 30 dk → 3 456 (bitki başına
288); 6 sa → bitki başına ~23 (filtre sonrası).

## 3. YÖNTEM

### 3.1 Bölmeler
- **LOPO (asıl değerlendirme):** bitki-dışarıda-bırak, 12 katman.
- **Yazar bölmesi:** Buss vd. 2026'daki gibi bitki 0 ve 12 test.
- **Rastgele:** tabakalı 5 katlı pencere bölmesi (karşılaştırma için).
- Sarmaşıkta ayrıca **zaman bloğu** bölmesi.
- Tüm ayarlar deneyden önce sabitlendi; test bitkisine bakılarak ayar yapılmadı. Tohum 42 (tek tohum).

### 3.2 Modeller
| Model | Girdi | Ayarlar |
|---|---|---|
| kNN, Gauss Naive Bayes ("uygun olmayan" taban çizgileri) | robust z-skorlu ham pencere | k = 5 |
| tsfresh + LightGBM | yazarların hazır ~780 tsfresh özniteliği | 300 ağaç, öğrenme oranı 0,05 |
| MiniRocket (aeon) | robust z-skorlu ham pencere | varsayılan |
| HuBERT-ECG small (30,5 M; `Edoardo-Coppola/hubert-ecg-small`) | bkz. 3.3 | donmuş sonda ve ince ayar |
| ECG-FM (90,9 M; wav2vec 2.0 tabanlı) | bkz. 3.3 | donmuş sonda ve ince ayar |
| Aynı iki mimari **rastgele ağırlıklarla** | aynı | kontrol: kazanç ön eğitimden mi? |

### 3.3 Bitki sinyalini EKG modeline uyarlama (zaman sıkıştırma)
- Pencere robust z-skorlanır, modelin sabit girdi uzunluğuna **yeniden örneklenir**: HuBERT-ECG
  için 500 örnek (100 Hz × 5 s), ECG-FM için 2 500 örnek (500 Hz × 5 s).
- Tek kanal **12 derivasyona kopyalanır** (alternatif: pencereyi 12 parçaya bölme — ablasyonda
  denendi).
- Böylece 1 sa pencere ~720 kat, 6 sa pencere ~4 320 kat hızlanmış olur.
- **Donmuş sonda:** model eğitilmez; katmanların zaman ortalaması gömme → StandardScaler +
  lojistik regresyon (C = 1).
- **İnce ayar (Colab T4):** 15 epoch, toplu 32, AdamW (gövde 3e-5, baş 1e-3), ağırlık çürümesi
  0,01, %10 ısınma + doğrusal azalma, CNN öznitelik çıkarıcı donmuş, mask_time_prob 0,05; her
  katmanda 2 eğitim bitkisi iç doğrulama, en iyi epoch iç doğrulama AUC'siyle seçildi.
- **DAPT (alana uyarlamalı ön eğitim):** etiketsiz 4 400 pencere (kontrol domates 572 + sarmaşık
  3 828; test edilen 12 bitki hariç), maskeli yeniden yapılandırma, 10 epoch.

### 3.4 İstatistik
12 bitki üzerinde eşleştirilmiş Wilcoxon + %95 bootstrap güven aralığı. Çoklu karşılaştırma
düzeltmesi sistematik yapılmadı (bir yerde Bonferroni notu var).

## 4. SONUÇLAR

### 4.1 Değerlendirme yöntemi sonucu şişiriyor (domates, doğruluk %)
| Model | Rastgele | Yazar (2 bitki) | LOPO |
|---|---|---|---|
| tsfresh + LightGBM, 1 sa | 90,9 ± 0,9 | 87,2 (yazarların raporu 84,0) | **72,9 ± 11,3** |
| tsfresh + LightGBM, 30 dk | 92,2 ± 1,2 | 83,9 (yazarlar 83,2) | 70,5 ± 11,4 |
| tsfresh + LightGBM, 6 sa | 85,9 | 95,7 | **81,2 ± 17,5** |
| MiniRocket, 1 sa | 79,7 ± 1,5 | 81,9 | 72,5 ± 13,2 |
| kNN / NB, 1 sa | ~52 / ~50 | ~46–51 | ~50,5 |

LOPO'da bitki başına doğruluk %51–94 arası (1 sa LightGBM: bitki 0 %94, bitki 13 %57, bitki 1 %60).

**Sarmaşık (makro F1 %, 6 089 saatlik pencere, bitki ve kanal başına z-skor):**
| Görev | Model | Rastgele | LOPO | Zaman bloğu |
|---|---|---|---|---|
| Gündüz/gece | LightGBM | 76,7 | 67,4 | 68,4 |
| Gündüz/gece | MiniRocket | 76,1 | 70,6 | 71,2 |
| Yağmurlu/kuru | LightGBM | 72,6 | 65,4 | 66,2 |
| Yağmurlu/kuru | MiniRocket | 70,4 | 67,0 | 64,5 |

### 4.2 Zaman karıştırıcısı testi (LightGBM, LOPO, "ilk 3 gün / son 3 gün")
| Grup | 1 sa doğruluk / AUC | 30 dk doğruluk / AUC |
|---|---|---|
| Kontrol (4 bitki, stres yok) | 63,7 / 65,0 | 53,3 / 49,3 |
| Sulama (12 bitki) | 72,9 / 80,7 | 70,5 / 79,3 |

### 4.3 1 saatlik pencere, LOPO (doğruluk / AUC)
| Model | Doğruluk | AUC |
|---|---|---|
| tsfresh + LightGBM | 72,9 | 80,7 |
| MiniRocket | 72,5 | 79,5 |
| HuBERT ince ayar (EKG) | 71,1 ± 8,9 | 79,4 |
| ECG-FM ince ayar (EKG) | 70,8 ± 12,5 | 78,9 |
| HuBERT donmuş (EKG) | 70,1 ± 10,0 | 78,9 |
| ECG-FM ince ayar (rastgele) | 70,4 | 77,2 |
| ECG-FM donmuş (rastgele) | 70,0 | 76,7 |
| HuBERT ince ayar (rastgele) | 69,2 ± 11,6 | 77,0 |
| HuBERT donmuş (rastgele) | 68,9 | 74,4 |
| ECG-FM donmuş (EKG) | 64,8 | 72,8 |

- LightGBM, MiniRocket ve EKG modelleri arasında anlamlı fark yok (ör. HuBERT ince ayar −1,8
  [−6,6; 2,8], p = 0,44).
- Ön eğitimin etkisi 1 sa'te: HuBERT ince ayar +1,9 (p = 0,61); donmuş AUC +4,5 [1,6; 7,7]
  p = 0,016 (düzeltilmemiş); ECG-FM donmuş −5,2 (p = 0,058, EKG ağırlıkları zararlı).
- Ön eğitimli HuBERT daha hızlı yakınsıyor (en iyi epoch ortancası 4'e karşı 12).

### 4.4 Zaman ölçeği taraması (HuBERT donmuş, LOPO, doğruluk)
| Pencere | EKG ağırlıkları | Rastgele | Fark (doğruluk / AUC) |
|---|---|---|---|
| 5 dk | 67,1 | — | — |
| 30 dk | 69,2 | 68,3 | +0,9 / +3,9 |
| 1 sa | 70,1 | 68,9 | +1,2 / +4,5 |
| **6 sa** | **77,9** | **72,8** | **+5,1 / +8,2** |

### 4.5 DAPT 2×2 deneyi, 6 sa, LOPO
| Kol | Doğruluk | AUC |
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

- **EKG ön eğitiminin etkisi (6 sa):** ince ayar +12,7 [+5,1; +19,9], p = 0,010 (12 bitkiden 10'unda);
  donmuş AUC +8,1, p = 0,012.
- **DAPT'ın ek etkisi:** +1,4 (p = 0,65) ve +1,8 (p = 0,52) → anlamsız.
- **En iyi EKG kolu vs LightGBM:** −1,4 [−11,2; +9,4], p = 0,66 → eşdeğer, geçemiyor.

### 4.6 Ana bulgular
1. Rastgele bölme sonucu ~20 puan şişiriyor (domates); sarmaşıkta 3–9 puan.
2. EKG ön eğitimi yalnızca 6 sa pencerede anlamlı katkı sağlıyor; 30 dk – 1 sa'te sağlamıyor.
3. EKG modelleri güçlü klasik modelle (LightGBM) eşdeğer; onu geçmiyor.

### 4.7 Önerilen mekanizma (henüz test edilmedi)
6 sa pencere 5 s'ye sıkıştırıldığında (~4 320 kat), bitkide yaklaşık saatte bir tekrarlayan bir
dalgalanma saniyede ~1,2 tekrara düşer; bu, kalp atış hızına (~72/dk) yakın. 1 sa pencerede
aynı dalgalanma ~0,2 Hz'de kalır. Yani 6 sa'te bitki sinyalinin bir bileşeni EKG modelinin
"tanıdığı" frekans bandına oturuyor olabilir.

## 5. BİLİNEN SINIRLAR (kendimiz farkındayız)
- Yalnızca 12 test bitkisi; güven aralıkları ±10 puan; güç düşük.
- Tek tohum; 6 sa'te bitki başına ~23 pencere.
- Etiket deneyin başı ve sonuyla çakışıyor (zaman karıştırıcısı); kontrol testi 4 bitkiyle.
- Ana bulgu tek veri setinden; sarmaşıkta EKG modeli denenmedi; sarmaşık etiketleri vekil
  (hava durumu).
- Kanal 12'ye kopyalanıyor (EKG'nin uzamsal yapısı yok).
- Çoklu karşılaştırma düzeltmesi sistematik değil.
- Sadece iki EKG modeli; konuşma modeli ya da genel zaman serisi temel modeliyle kaynak alan
  karşılaştırması yok.
- Mekanizma (4.7) ölçülmedi.

## 6. KAYNAKLAR

**Veri setlerinin makaleleri**
- Buss, E. vd. (2026). Early detection of water stress by plant electrophysiology. arXiv 2604.28038 (ön baskı).
- Buss, E., Aust, T., Hamann, H. (2025). When plants respond: Electrophysiology and machine learning for green monitoring systems. Biomimetic and Biohybrid Systems, LNCS, 249–261. doi:10.1007/978-3-032-07448-5_21
- Buss, E., Aust, T., Wahby, M., Rabbel, T.-L., Kernbach, S., Hamann, H. (2023). Stimulus classification with electrical potential and impedance of living plants. Zenodo 7126105.

**Bitki elektrofizyolojisi + makine öğrenmesi**
- Wahid, Baihaqi, Nambo (2026). Derleme, Smart Agricultural Technology 15:102484. doi:10.1016/j.atech.2026.102484
- González i Juclà, D. vd. (2023). Detecting stress caused by nitrogen deficit using deep learning techniques applied on plant electrophysiological data. Sci Rep 13:9633. doi:10.1038/s41598-023-36683-3 — 16 domates, **LOOCV %77,0 ± 12,1**, ardışık 1 000 tahmin birleştirmeyle %87,6.
- Tran, D. vd. (2019). Electrophysiological assessment of plant status outside a Faraday cage using supervised machine learning. Sci Rep 9:17073. doi:10.1038/s41598-019-53675-4 — rastgele %80/20, kuraklık %98,5; girişte insan kalp/beyin ölçümüne benzetme.
- Aust, T. vd. (2024). Automated phytosensing: Ozone exposure classification based on plant electrical signals. arXiv 2412.13312 — rastgele bölmede %94,6, ayrı test kümesinde %71–74.
- Bhadra, N., Chatterjee, S. K., Das, S. (2023). PLOS ONE 18(5):e0285321. doi:10.1371/journal.pone.0285321
- Chatterjee, S. K. vd. (2015). J. R. Soc. Interface. doi:10.1098/rsif.2014.1225
- Chatterjee, S. K. vd. (2018). Biosensors 8(3):83. doi:10.3390/bios8030083
- Najdenovska, E. vd. (2021). Applied Sciences 11(12):5640. doi:10.3390/app11125640
- Diğer (henüz okunmadı): Najdenovska 2021 (Appl. Sci. 11(4):1414, akar); Wen vd. 2025 (AI in Agriculture); NJAS 2025 doi:10.1080/27685241.2025.2534470; Qi vd. 2024 (Biosens. Bioelectron.); Qi vd. 2026 (Physiol. Plant., doi:10.1111/ppl.70954); Biosensors 2025 doi:10.3390/bios15110744 (ImageNet modeli); Sensors 2024, 24(6):1917 (ImageNet modeli).

**EKG temel modelleri**
- Coppola, E. vd. (2024). HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications. medRxiv 10.1101/2024.11.14.24317328
- McKeen, K. vd. ECG-FM: An open electrocardiogram foundation model. arXiv 2408.05178
- Al-Masud, M. A., Strodthoff, N. (2026). Pretraining strategies and scaling for ECG foundation models: A systematic study. arXiv 2605.12241 — S4 + CPC en iyi; HuBERT++ > HuBERT-ECG; DAPT öneriliyor.
- Tang, W. vd. (2026). Extending pretrained 10-second ECG foundation models to longer horizons. arXiv 2605.16975

**Alanlar arası aktarım**
- Yang, C.-H. H., Tsai, Y.-Y., Chen, P.-Y. (2021). Voice2Series: Reprogramming acoustic models for time series classification. ICML, PMLR 139. arXiv 2106.09296
- Lu, K., Grover, A., Abbeel, P., Mordatch, I. (2022). Frozen pretrained transformers as universal computation engines. AAAI 36(7):7628–7636. arXiv 2103.05247
- Zhou, T. vd. (2023). One fits all: Power general time series analysis by pretrained LM. NeurIPS. arXiv 2302.11939
- Jin, M. vd. (2024). Time-LLM: Time series forecasting by reprogramming large language models. ICLR. arXiv 2310.01728
- Tan, M. vd. (2024). Are language models actually useful for time series forecasting? NeurIPS. arXiv 2406.16964
- Li, C. vd. (2026). BioX-Bridge: Model bridging for unsupervised cross-modal knowledge transfer across biosignals. ICLR. arXiv 2510.02276
- İnsandan ata EKG transferi (PMC6957496); DeepMiceTL (insan → fare EKG); insandan fareye uyku EEG transferi (PMC11025629); QSLP-AE (ICLR 2026) — **künyeleri henüz tam doğrulanmadı.**

## 7. SENDEN İSTENENLER

Her başlık için kısa ve net cevap ver; gerekiyorsa "bilmiyorum" de.

1. **Yöntem hataları:** Veri sızıntısı, yanlış istatistik, adil olmayan karşılaştırma ya da
   sonuçları geçersiz kılabilecek bir tasarım hatası görüyor musun? (Özellikle: etiketin zamanla
   çakışması, 6 sa'te bitki başına ~23 pencere, iç doğrulamayla epoch seçimi, DAPT verisinde
   kontrol bitkilerinin kullanılması.)
2. **Ana iddia sağlam mı?** "EKG ön eğitimi 6 sa ölçekte anlamlı katkı sağlıyor (+12,7, p = 0,010)"
   iddiası 12 bitki ve tek tohumla Q1 hakemini ikna eder mi? Çoklu karşılaştırma sorunu var mı?
   Ölçek taraması sonradan seçilmiş (post hoc) görünür mü?
3. **Alternatif açıklamalar:** 6 sa'teki kazancı EKG'ye özgü bilgi dışında ne açıklayabilir?
   (ör. herhangi bir ön eğitimli modelin genel faydası, girdi uzunluğu/yeniden örnekleme etkisi,
   gece-gündüz döngüsü.) Bunları ayırmak için hangi kontrol deneyleri şart?
4. **Yenilik:** İnsan EKG modelini bitkiye aktaran bir çalışma biliyor musun? Katkı listesindeki
   hangi madde gerçekten yeni, hangisi değil?
5. **Q1'e hazır mı?** Bugünkü haliyle hangi dergi düzeyine uygun? Q1 için **olmazsa olmaz** en
   fazla 5 ek iş nedir, öncelik sırasıyla? Her biri için neden gerekli olduğunu yaz.
6. **Çerçeveleme:** EKG modelleri LightGBM'i geçmiyor. Makale "negatif/karışık sonuç" olarak mı,
   "ne zaman işe yarar" (koşullu transfer) olarak mı, yoksa "sızıntı farkındalıklı benchmark"
   olarak mı kurulmalı? Hangisi daha güçlü ve neden?
7. **Eksik literatür:** Atlanan önemli bir çalışma var mı (bitki elektrofizyolojisi, biyosinyal
   temel modelleri, alanlar arası aktarım)? Yukarıdaki kaynaklarda yanlış bir künye var mı?
8. **Sayı tutarlılığı:** Tablolar arasında çelişki ya da şüpheli görünen bir sayı var mı?
