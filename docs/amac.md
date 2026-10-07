# Amaç

Son güncelleme: 2026-10-07

## 1. Projenin amacı (değişmez çerçeve)

**Soru:** İnsan EKG'si ile bitki elektrik sinyali arasında, yapay zekâ modelinin kullanabileceği
bir ortaklık var mı? Varsa, insan EKG'siyle önceden eğitilmiş modeller (HuBERT-ECG, ECG-FM)
domateste sulama stresini **hiç görülmemiş bitkide** tespit edebilir mi?

**Hedef:** Bu soruyu Q1 dergi makalesi düzeyinde cevaplamak. Konunun dışına çıkılmaz; yeni bir
deney, model ya da yön eklemeden önce kullanıcının onayı alınır.

**Şu anki cevap (kısaca):** Ortaklık var, ama yalnızca doğru zaman ölçeğinde. Bitkinin 6 saatlik
sinyali modelin 5 saniyesine sıkıştırıldığında EKG ön eğitimi aynı modelin rastgele başlatılmışına
göre +12,7 puan kazandırıyor (p = 0,010). 30 dk – 1 sa ölçeğinde bu etki yok. Ayrıntı:
`docs/sonuclar.md`.

## 2. Bitki ve insan EKG'sini hangi alanlarda karşılaştırıyoruz?

İki sinyali doğrudan yan yana koymuyoruz; insan EKG'siyle eğitilmiş modeli bitki sinyaline
**aktarıyoruz** (transfer). Bunun çalışması için iki sinyalin ortak yanları olmalı:

| Alan | İnsan EKG'si | Bitki sinyali | Benzer mi? |
|---|---|---|---|
| Sinyal türü | Kalp hücrelerinin elektrik potansiyeli | Gövde hücrelerinin elektrik potansiyeli | ✅ İkisi de biyoelektrik |
| Ölçüm | İki elektrot arası voltaj farkı | Gövdeye batırılmış iki gümüş elektrot arası voltaj farkı | ✅ Aynı mantık |
| Kanal sayısı | 12 derivasyon | 1 kanal | ❌ Kanal 12'ye kopyalanıyor |
| Zaman ölçeği | Kalp atışı ~1 saniye | Dalgalanmalar dakikalar – saatler | ❌ Çok farklı |
| Örnekleme | 100–500 Hz | 1 Hz | ❌ Yeniden örnekleme ile uyduruluyor |
| Değişkenlik | Kişiden kişiye fark | Bitkiden bitkiye büyük fark | ✅ Aynı sorun |

### Zaman ölçeği hipotezi
6 saatlik bitki sinyali modele 5 saniyelik EKG gibi verildiğinde zaman ~4 300 kat hızlanır.
Bitkide yaklaşık saatte bir tekrarlayan bir dalgalanma bu durumda saniyede ~1,2 tekrara düşer;
bu, kalp atışı hızına (~72/dk) yakındır. 1 saatlik pencerede aynı dalgalanma çok yavaş kalır.
Bu, "neden yalnızca 6 sa'te işe yarıyor?" sorusunun olası açıklamasıdır. **Henüz hipotez;**
frekans örtüşmesi analiziyle doğrulanacak.

## 3. Bu karşılaştırmanın avantajları

**Bilimsel (makale için):**
1. **Özgünlük:** Wahid vd. (2026), 182 çalışmalık derlemesinde EKG/EEG'den bitkiye transferi
   "doğrulanmamış hipotez" olarak bırakıyor. İlk deneysel test bu çalışma.
2. **Ne zaman işe yaradığı:** Transferin doğru zaman ölçeğinde çalıştığı gösteriliyor; başka
   sinyallere de uygulanabilecek bir ders.
3. **Sağlam kanıt:** Aynı mimarinin rastgele ağırlıklı haliyle karşılaştırma yapıldı; kazanç
   gerçekten EKG ön eğitiminden geliyor (+12,7 puan, p = 0,010). Tan vd. (2024) bu kontrolü
   yapmayan çalışmaların yanıldığını gösterdi.

**Pratik:**
1. **Veri azlığına çözüm:** 16 bitkiye karşı milyonlarca insan EKG'siyle eğitilmiş model.
2. **Daha hızlı öğrenme:** En iyi tur ortancası 4 (EKG) / 12 (rastgele).
3. **Daha tutarlı sonuç:** Bitkiler arası std ±8,9 / ±11,6 (1 sa, ince ayar).
4. **Sıralama gücü:** En iyi AUC (88,4) EKG modelinde.
5. **Düşük maliyet:** Hazır modeller, Colab, sıfır donanım.

**Dürüst sınır:** Doğrulukta EKG modelleri LightGBM'i geçmiyor (%79,7 / %81,2, p = 0,66).
Bugünkü iddia "eşdeğer başarı, daha tutarlı ve daha iyi sıralama"dır; "daha yüksek başarı" değil.

## 4. Şu anki başarı (LOPO, 12 bitki)

| Algoritma | 6 sa doğruluk | 6 sa AUC | 1 sa doğruluk |
|---|---|---|---|
| tsfresh + LightGBM | **81,2** | 86,7 | **72,9** |
| HuBERT-ECG donmuş + DAPT | 79,7 | 87,8 | — |
| HuBERT-ECG donmuş | 77,9 | **88,4** | 70,1 |
| HuBERT-ECG ince ayar (+DAPT) | 74,6 (76,1) | 84,2 (84,3) | 71,1 |
| MiniRocket | 76,1 | 87,1 | 72,5 |
| ECG-FM ince ayar | — | — | 70,8 |
| HuBERT rastgele ağırlık (kontrol) | 62,0–73,2 | 72,5–80,2 | 68,9–69,2 |
| Naive Bayes / kNN | 62,7 / 54,3 | — | ~50 |

## 5. Benzer çalışmalar

PDF'ler `docs/makaleler/` ve `docs/makaleler/benzer/` altında (telif nedeniyle git'e girmez).
İndirme durumu: ✅ indi (başlığı PDF'ten doğrulandı) · ⬜ elle indirilecek (kullanıcı; bkz. §5.5).
⚠️ İşaretlilerin yalnızca arama sonucu görüldü; künyesi ve içeriği doğrulanmalı. İnen makaleler de
henüz okunmadı; makalede alıntılamadan önce okunacak.

### 5.1 Bitki elektrofizyolojisi ve makine öğrenmesi

| Çalışma | Ne yapmışlar | Bizden farkı | Dosya |
|---|---|---|---|
| Buss vd. 2026 (arXiv 2604.28038) | Domates sulama stresi; tsfresh + HGB, CNN/InceptionTime/Mamba | Ana veri setimiz; 2 bitkilik test, temel model yok | ✅ `Buss2026_Early_Detection_Water_Stress.pdf` |
| Buss vd. 2025 (LNCS) | Sarmaşık, RF/AutoML, F1 %90 | Rastgele bölme; biz 3–9 puan şişik bulduk | ✅ `Buss2025_When_Plants_Respond.pdf` |
| Buss vd. 2023 | Zamioculcas uyaran sınıflandırma | Rastgele bölme | ✅ `Buss2023_Stimulus_Classification.pdf` |
| Chatterjee vd. 2015 (J. R. Soc. Interface) | PLEASED, kimyasal uyaran, diskriminant analizi | Klasik | ✅ `Chatterjee2015.pdf` |
| Chatterjee vd. 2018 (Biosensors) | Eğri uydurma öznitelikleri, LDA/QDA | Klasik | ⬜ |
| Tran vd. 2019 (Sci Rep) | Bitki sinyaliyle stres sınıflandırma | Klasik | ✅ `benzer/Tran2019_SciRep.pdf` |
| Najdenovska vd. 2021 (Appl. Sci., kuraklık) | XGBoost, %85 | Veri kapalı | ⬜ |
| Najdenovska vd. 2021 (Appl. Sci., akar) | Akar tespiti | Klasik | ⬜ |
| Bhadra vd. 2023 (PLOS ONE) | PLEASED, AdaBoost, %71 | Rastgele bölme | ✅ `benzer/Bhadra2023_PLOSONE.pdf` |
| Aust vd. 2024 (arXiv 2412.13312) | Ozon, AutoML | Klasik | ✅ `benzer/Aust2024_Ozon_AutoML.pdf` |
| NJAS 2025 (10.1080/27685241.2025.2534470) | Dış uyaranlar | Klasik | ⬜ |
| Wen vd. 2025 (AI in Agriculture) | Kanola ve yulaf | Klasik | ⬜ |

### 5.2 Bitkide derin öğrenme ve ön eğitimli model

| Çalışma | Ne yapmışlar | Bizden farkı | Dosya |
|---|---|---|---|
| González i Juclà vd. 2023 (Sci Rep) | Azot eksikliği, Encoder, %99 | Görülmemiş bitki testi yok | ✅ `benzer/Gonzalez2023_SciRep_Azot.pdf` |
| Qi vd. 2024 (Biosens. Bioelectron.) | Toprak nemi | Farklı sensör/görev | ⬜ |
| Biosens. Bioelectron. 2025 (PMID 39708493) | Mikro-iğne sensör | Donanım odaklı | ⬜ |
| Qi vd. 2026 (Physiologia Plantarum) | Alkali stres | Ön eğitimli model yok | ⬜ |
| Biosensors 2025 (10.3390/bios15110744) | AD8232 ile ölçüm + ön eğitimli model | ImageNet modeli, tek bitki | ⬜ |
| Sensors 2024 (24/6/1917) | Bitki sinyali + ön eğitimli model | ImageNet modeli | ⬜ |
| Wahid vd. 2026 (Smart Agric. Tech. 15:102484) | 182 çalışmalık derleme | EKG/EEG → bitki "denenmemiş hipotez"; çıkış noktamız | ⬜ |

### 5.3 Kullandığımız EKG modelleri ve EKG temel modeli çalışmaları

| Çalışma | İçerik | Dosya |
|---|---|---|
| HuBERT-ECG (medRxiv 2024) | 9,1 M EKG ile eğitilmiş; kullandığımız model | ⬜ |
| ECG-FM (2024/2025) | Açık EKG temel modeli; kullandığımız ikinci model | ✅ `benzer/ECG_FM_2024.pdf` |
| Extending 10-second ECG FMs to longer horizons (arXiv 2605.16975) | EKG modelini uzun kayıtlara uyarlama; zaman sıkıştırma fikrimize yakın | ✅ `benzer/Extending_10s_ECG_FM_2026.pdf` |
| Pretraining strategies and scaling for ECG FMs (arXiv 2605.12241) | Ön eğitim ölçeği ne zaman işe yarar | ✅ `benzer/Pretraining_Scaling_ECG_FM_2026.pdf` |
| ⚠️ Scaling ECG FMs and a threshold for representation learning (PMC13409230) | Ölçek eşiği | ⬜ |

### 5.4 Bir veri setinden başka bir veri setine aktarım (alanlar arası transfer)

Kaynak ile hedef arasındaki uzaklığa göre, yakından uzağa:

**a) Aynı sinyal, farklı canlı**

| Çalışma | Kaynak → hedef | Sonuç | Dosya |
|---|---|---|---|
| ⚠️ İnsandan ata EKG transferi (PMC6957496) | İnsan EKG (MIT-BIH) → at EKG | %92,6 | ⬜ |
| ⚠️ DeepMiceTL | İnsan EKG → fare EKG | Doğruluk %84,8 | ⬜ |
| ⚠️ İnsandan fareye uyku EEG (PMC11025629) | İnsan uyku EEG → fare uyku evresi | Ön eğitim yardımcı; yanlış tür kaynak (motor imgeleme) zararlı | ⬜ |
| ⚠️ QSLP-AE (ICLR 2026) | İnsan + fare uyku EEG, ortak gizli uzay | Türler arası hizalama | ⬜ |

**b) Farklı sinyal, aynı canlı**

| Çalışma | Kaynak → hedef | Dosya |
|---|---|---|
| BioX-Bridge (arXiv 2510.02276) | EKG modeli → PPG vb. biyosinyaller | ✅ `benzer/BioX_Bridge2025.pdf` |

**c) Tamamen farklı alan (bize en benzer)**

| Çalışma | Kaynak → hedef | Sonuç | Dosya |
|---|---|---|---|
| Voice2Series (Yang vd., ICML 2021) | Konuşma modeli → genel zaman serisi | 31 görevin 22'sinde eşit/üstün; en yakın fikir | ✅ `benzer/Yang2021_Voice2Series.pdf` |
| Frozen Pretrained Transformer (Lu vd., AAAI 2022) | Dil modeli → görüntü, protein, sayısal | Sıfırdan eğitimle eşdeğer, daha hızlı yakınsama | ✅ `benzer/Lu2022_Frozen_Pretrained_Transformers.pdf` |
| GPT4TS / One Fits All (Zhou vd., NeurIPS 2023) | Dil modeli → zaman serisi | Birçok görevde iyi | ✅ `benzer/Zhou2023_GPT4TS_One_Fits_All.pdf` |
| Time-LLM (Jin vd., ICLR 2024) | Dil modeli → zaman serisi tahmini | Girdiyi yeniden programlama | ✅ `benzer/Jin2024_Time_LLM.pdf` |
| Tan vd. 2024 (NeurIPS), "Are LMs actually useful for TS forecasting?" | Eleştiri | Rastgele ağırlıkla değiştirince sonuç değişmiyor → kazanç ön eğitimden değil | ✅ `benzer/Tan2024_Are_LMs_Useful_for_TS.pdf` |

### 5.5 Elle indirilecekler (kullanıcı)
Otomatik indirme başarısız oldu (yayıncı engeli ya da ücretli erişim). Hepsi küçük PDF; 1 GB üzeri
dosya yok. İndirince `docs/makaleler/benzer/` altına aynı adla koyulmalı.

| Dosya adı | Çalışma | Link |
|---|---|---|
| `HuBERT_ECG_2024.pdf` | HuBERT-ECG (medRxiv) | https://www.medrxiv.org/content/10.1101/2024.11.14.24317328 |
| `Scaling_ECG_FM_Threshold.pdf` | Scaling ECG FMs (PMC) | https://pmc.ncbi.nlm.nih.gov/articles/PMC13409230/ |
| `Insan_At_EKG_Transfer.pdf` | İnsandan ata EKG transferi | https://pmc.ncbi.nlm.nih.gov/articles/PMC6957496/ |
| `Insan_Fare_Uyku_EEG.pdf` | İnsandan fareye uyku EEG | https://pmc.ncbi.nlm.nih.gov/articles/PMC11025629/ |
| `DeepMiceTL.pdf` | DeepMiceTL | Link bulunmadı; Google Scholar'da "DeepMiceTL" aranmalı |
| `QSLP_AE_ICLR2026.pdf` | QSLP-AE | https://iclr.cc/virtual/2026/10022303 (OpenReview'dan) |
| `Chatterjee2018_Biosensors.pdf` | Chatterjee vd. 2018 | https://doi.org/10.3390/bios8030083 |
| `Najdenovska2021_Kuraklik.pdf` | Najdenovska vd. 2021 | https://doi.org/10.3390/app11125640 |
| `Najdenovska2021_Akar.pdf` | Najdenovska vd. 2021 | https://www.mdpi.com/2076-3417/11/4/1414 |
| `NJAS2025_Dis_Uyaranlar.pdf` | NJAS 2025 | https://doi.org/10.1080/27685241.2025.2534470 |
| `Wen2025_Kanola_Yulaf.pdf` | Wen vd. 2025 | https://www.sciencedirect.com/science/article/pii/S2589721725000467 |
| `Qi2024_Toprak_Nemi.pdf` | Qi vd. 2024 (ücretli olabilir) | https://www.sciencedirect.com/science/article/abs/pii/S095656632400530X |
| `BiosensBioelectron2025_Mikroigne.pdf` | Mikro-iğne sensör (ücretli olabilir) | https://pubmed.ncbi.nlm.nih.gov/39708493/ |
| `Qi2026_Alkali.pdf` | Qi vd. 2026 (ücretli olabilir) | https://onlinelibrary.wiley.com/doi/10.1111/ppl.70954 |
| `Biosensors2025_AD8232_ImageNet.pdf` | Biosensors 2025 | https://doi.org/10.3390/bios15110744 |
| `Sensors2024_Bitki_Onegitimli.pdf` | Sensors 2024 | https://www.mdpi.com/1424-8220/24/6/1917 |
| `Wahid2026_Derleme.pdf` | Wahid vd. 2026 | https://www.sciencedirect.com/science/article/pii/S2772375526007094 |

### 5.6 Bu tablodan çıkanlar
1. "Bir alanda eğitilmiş modeli çok farklı bir alana taşımak" bilinen bir araştırma çizgisi;
   hakem fikri yadırgamaz.
2. Bulunan en uzak canlı atlaması insan → hayvan; **insan → bitki** atlaması bulunamadı.
3. Rastgele ağırlık kontrolü baştan yapıldı (Tan vd. 2024'ün eleştirdiği eksik bizde yok).
4. Kaynak ile hedef aynı tür süreci yansıttığında transfer çalışıyor (uyku → uyku); bizim
   "doğru zaman ölçeği" bulgumuz bununla tutarlı.

## 6. Q1'e giden yol (onaylanmadan başlanmaz)

Hepsi bu çerçevenin içinde; sırası ve kapsamı kullanıcıyla kararlaştırılır.
1. Frekans örtüşmesi analizi (zaman ölçeği hipotezinin testi).
2. Hibrit model: 6 sa EKG gömmeleri + tsfresh (gömmeler `data/processed/` altında hazır).
3. Bitkinin kendi başlangıcına göre ayarlama + kontrol bitkisi testi.
4. Birkaç pencereye birlikte karar ve erken tespit eğrisi.
5. Colab: 12/24 sa ölçek, tohum tekrarı, başka ön eğitimli model.
6. Daha çok bitki: PLEASED veri seti (link kontrolü kullanıcıda).
7. Benzer makalelerin okunup İngilizce makaleye işlenmesi.
