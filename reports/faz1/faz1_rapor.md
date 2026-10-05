# Faz 1 İlerleme Raporu (Taslak)

**Başlık:** İnsan EKG'sinden Bitkiye: Biyosinyal Temel Modelleri ile Bitki Elektrofizyolojisinde
Su Stresi Tespiti
**Proje türü:** Bireysel · Mehmet Melih Akçam
**Kategori:** Çok değişkenli zaman serileri
**Depo:** https://github.com/melihakcam/bitkiEkg (dal: `onDeneme`)

> Bu taslak, dersin Faz 1 şablonu paylaşıldığında o şablona aktarılacaktır.
> Şekiller `results/figures/`, tablolar `results/tables/` altındadır; üreten kodlar `scripts/` ve
> `notebooks/` klasörlerindedir.

---

## 1. Problem Tanımı

Bitkiler su eksikliği veya fazlalığı gibi olumsuz sulama koşullarında gövdelerinde ölçülebilir
elektrik potansiyeli değişimleri üretir. Bu çalışmada problem, gövdeden ölçülen zaman serisi
pencerelerinden bitkinin **sağlıklı mı, sulama stresi altında mı** olduğunu **ikili
sınıflandırma** ile tespit etmektir. Ana veri setinde (Buss vd., 2026) "stresli" sınıfı hem
eksik sulanan (100 mL, 200 mL) hem de aşırı sulanan bitkileri kapsadığından hedef etiket
"sulama stresi" olarak tanımlanmıştır.

Temel güçlükler: (i) veri azlığı (16 bitki), (ii) bitkiden bitkiye değişen sinyal, (iii)
literatürde yaygın olan rastgele pencere bölmesinin sonuçları iyimser göstermesi. Proje, insan
EKG'siyle önceden eğitilmiş biyosinyal temel modellerinin (ECG-FM, HuBERT-ECG) bitki sinyaline
aktarılıp aktarılamayacağını, eğitimde görülmemiş bitkiler üzerinde test eder.

## 2. Veri Setleri

| Veri seti (literatürdeki adı, sürüm) | İçerik | Link | Projedeki rolü |
|---|---|---|---|
| Early Detection of Water Stress by Plant Electrophysiology (Buss vd., 2026), Zenodo v2 | Domates, 16 bitki, 18 gün; cihazda 10 Hz, veri setinde 1 Hz; CC-BY | https://doi.org/10.5281/zenodo.18876513 | Ana veri seti (eğitim/test) |
| When Plants Respond: Electrophysiology and ML for Green Monitoring Systems (Buss vd., 2025), Zenodo v1 | Sarmaşık, 4 bitki, 5 ay dış ortam, 1 Hz, 2 kanal; CC-BY | https://doi.org/10.5281/zenodo.15095523 | Türler/ortamlar arası test |
| Stimulus classification with electrical potential and impedance of living plants (Buss vd., 2023), Zenodo v1 | Zamioculcas (potansiyel) ve domates (empedans); rüzgâr, ısı, ışık; CC-BY | https://doi.org/10.5281/zenodo.7126105 | Ek veri / yardımcı görev |

Kullanılan dosyalar ve boyutları: domates `AdditionalMaterial.zip` 9,7 GB (açılmış 22,8 GB,
260 483 dosya); sarmaşık `Plant_data.zip` 423 MB + hava durumu; uyaran `SupplementaryCode.zip`
124 MB. Uyaran kaydındaki `DeepClassifier.zip` (4,2 GB) yalnızca yazarların eğittiği modelleri
içerdiğinden indirilmemiştir. Tüm dosyalar `scripts/download_data.py` ile indirilmiş ve Zenodo'nun
MD5 özetleriyle doğrulanmıştır. Veriler yapay zekâ ile üretilmemiştir; yazarların yayımladığı
ölçümlerdir.

## 3. Veri Seti Analizi

### 3.1 Domates sulama stresi (ana veri seti)
- **Düzenek:** Serada 16 domates; 8 PhytoNode cihazı, her cihazın CH1 ve CH2 kanalı **ayrı bir
  bitkiye** bağlı (bitki başına tek kanal: gövdeye batırılmış iki gümüş elektrot arası potansiyel
  farkı, tabanda ve ≥30 cm yukarıda).
- **Süre / örnekleme:** 04–21.06.2025 (18 gün); cihazda 10 Hz, veri setinde 1 Hz; birim mV.
- **Sulama protokolü:** 04–08.06 tüm bitkiler 400 mL/gün; ardından 4 grup × 4 bitki: kontrol
  (400 mL), aşırı sulanmış (sürekli suda), orta kuraklık (200 mL), şiddetli kuraklık (100 mL).
- **Dosya yapısı:** hazır pencereler (`00_time_windows`, `datetime, CH1, CH2`), tsfresh
  öznitelikleri (`01_features`), yazarların bölmesi (`02_test_train_val_split`) ve sonuçları
  (`03_results`); 5 pencere süresi.

  | Pencere | 1 dk | 5 dk | 30 dk | 1 sa | 6 sa |
  |---|---|---|---|---|---|
  | Pencere dosyası (cihaz başına 2 bitki) | 207 352 | 41 472 | 6 912 | 3 456 | 584 |

- **Etiketler (veride hazır, `class` sütunu):** ilk 3 gün = sağlıklı (0), son 3 gün = stresli (1),
  aradaki günler ve kontrol bitkileri = 3 (eğitimde kullanılmıyor, geçiş analizinde kullanılıyor).
- **Sınıf dağılımı (1 sa pencere):** sağlıklı 864 · stresli 864 · kullanılmayan 5 184 →
  ikili görev **dengeli**; 12 bitki eğitim/test için kullanılabilir (Şekil 4).
- **Eksik veri:** 15 bitkide %0; PN9 cihazının iki bitkisinde %0,004 (1 dk çözünürlükte).
  Kopuk elektrot kaynaklı uzun boşluk yok (`results/tables/domates_bitki_ozet.csv`).
- **Genlik:** bitki ortalamaları −63 ile +20 mV, standart sapmaları 6–100 mV arasında → bitkiler
  arası ölçek farkı büyük; bitki/pencere bazlı normalizasyon gerekli.
- **Gözlemler:**
  - Sulaması değiştirilen bitkilerde sinyal ilk günlerde daha oynak, son günlerde daha düzgün
    (Şekil 1–3). Aynı bitkinin sağlıklı ve stresli bir günü gözle ayırt edilebiliyor (Şekil 3).
  - Kontrol bitkilerinde de büyük sıçramalar var (ör. 17–18.06) → sinyal sulama dışındaki
    etkenlerden de etkileniyor.
  - **Olası karıştırıcı etken:** "sağlıklı" etiketi deneyin ilk 3 gününe, "stresli" etiketi son
    3 gününe denk geliyor. İlk günlerdeki oynaklığın bir kısmı elektrotların yeni takılmasından
    kaynaklanabilir. Bu, kontrol bitkilerinin ilk ve son günleri ayırt edilebiliyor mu diye
    Faz 2'de test edilecektir.

| Şekil | Dosya |
|---|---|
| 1. 18 günlük sinyal: sulama grubu (PN10) ve kontrol (PN8) | `results/figures/domates_18_gun.png` |
| 2. Bitki başına günlük dalgalanma (gün içi std) | `results/figures/domates_gunluk_dalgalanma.png` |
| 3. Aynı bitkinin sağlıklı ve stresli günü | `results/figures/domates_saglikli_vs_stresli.png` |
| 4. Sınıf dağılımı | `results/figures/domates_sinif_dagilimi.png` |

### 3.2 Sarmaşık dış ortam
- **Ölçüm:** PhytoNode, gümüş kaplı elektrotlar; gövde alt ucu ile 30–60 cm yukarısı (gövde veya
  yaprak sapı) arası potansiyel farkı. Bitki başına **2 kanal** (CH1, CH2).
- **Bitki / süre:** 4 sarmaşık (P1, P2, P3, P5), 05.07–18.11.2024, Konstanz botanik bahçesi.
- **Örnekleme / birim:** cihazda ~200 Hz, veri setinde 1 sn ortalama ile 1 Hz; değerler ham
  cihaz sayımı (~8×10⁶), mV karşılığı verilmemiş → z-skor kullanılacak.
- **Dosya yapısı:** bitki başına 12 saatlik CSV parçaları (`datetime, CH1, CH2`), toplam 575
  dosya; hava durumu 10 dk aralıklı (sıcaklık, nem, ışınım, yağış, rüzgâr, çiy noktası).
- **Eksik veri:** kapsama P1 %59 · P2 %50 · P3 %59 · P5 %53 (donanım iletişim sorunları,
  Şekil 6). İki dosyada (P1/P3, 10.08.2024) ~19 milyon boş satır var; okurken atılıyor.
  Hava durumunda buharlaşma sütunu %99 boş, diğerleri eksiksiz.
- **Etiketler:** veride hazır etiket yok; makaledeki eşiklerle hava durumundan türetiliyor.
  Doğrudan su stresi etiketi yok → türler arası testte vekil etiket **yağmurlu/kuru**.
- **Sınıf dağılımı (1 sa pencere, verinin ≥%80'i dolu saatler):**

  | Bitki | Geçerli saat | Gündüz / gece | Yağmurlu / kuru | Sıcak / soğuk (08–20) | Rüzgârlı / sakin (08–20) |
  |---|---|---|---|---|---|
  | P1 | 1 900 | 749 / 1 151 | 149 / 1 751 | 224 / 712 | 57 / 879 |
  | P2 | 1 541 | 558 / 983 | 147 / 1 394 | 101 / 651 | 47 / 705 |
  | P3 | 1 441 | 509 / 932 | 120 / 1 321 | 101 / 603 | 45 / 659 |
  | P5 | 1 214 | 430 / 784 | 122 / 1 092 | 63 / 535 | 42 / 556 |
  | **Toplam** | **6 096** | **2 246 / 3 850** | **538 / 5 558** | **489 / 2 501** | **191 / 2 799** |

  Yağmurlu/kuru (~1:10) ve rüzgârlı/sakin (~1:15) **çok dengesiz** → dengeli metrikler (makro F1,
  dengeli doğruluk) ve sınıf ağırlıklandırma gerekli.
- **Gözlem:** Yağış anlarında sinyalde belirgin sıçramalar görülüyor (ör. P2, 05.09.2024 akşamı,
  Şekil 7).

| Şekil | Dosya |
|---|---|
| 5. Tüm kayıt (10 dk ortalama) | `results/figures/sarmasik_tum_kayit.png` |
| 6. Günlük veri kapsaması | `results/figures/sarmasik_kapsama.png` |
| 7. Bir hafta: sinyal, sıcaklık, ışınım, yağış | `results/figures/sarmasik_bir_hafta.png` |
| 8. Hava durumu | `results/figures/sarmasik_hava.png` |

### 3.3 Uyaran sınıflandırma
- **Ölçüm:** Zamioculcas zamiifolia, CYBRES phytosensor, ~0,58 Hz (≈1,7 s/örnek), 2 kanal;
  ek olarak ortam sensörleri (sıcaklık, ışık, nem, toprak nemi vb.). Domates verisi yalnızca
  doku empedansı içeriyor.
- **Hazır veri:** UCR formatı — her satır etiket + 512 örnek (uyaran öncesi 256 + sonrası 256).
  Mavi/kırmızı ışık pencereleri yazarların kodunda her ikinci örnek alınarak oluşturulmuş
  (zaman ölçeği farklı). Eğitim 1 302, test 558 pencere (yazarların rastgele %70/%30 bölmesi);
  eksik değer yok.
- **Etiketler (yazarların kodundan doğrulandı):** 0 rüzgâr, 1 ısı, 2 uyaran yok, 3 mavi ışık,
  4 kırmızı ışık.

  | Sınıf | Eğitim | Test |
  |---|---|---|
  | Rüzgâr | 381 | 163 |
  | Isı | 353 | 151 |
  | Uyaran yok | 381 | 163 |
  | Mavi ışık | 92 | 40 |
  | Kırmızı ışık | 95 | 41 |

- **Gözlemler:** Rüzgâr ve ısı tepkileri uyarandan ~80–100 örnek (~2,5 dk) sonra başlıyor.
  Mavi/kırmızı ışık pencerelerinin hepsinde aynı noktada keskin bir basamak ve kırmızı ışıkta
  düzenli zikzak görülüyor (Şekil 10) → ölçüm veya veri hazırlama **artefaktı** olabilir;
  yazarların bu sınıflardaki çok yüksek doğrulukları kısmen bununla açıklanabilir. Bu veri seti
  bu nedenle yardımcı rolde kullanılacaktır.

| Şekil | Dosya |
|---|---|
| 9. Sınıf dağılımı | `results/figures/uyaran_sinif_dagilimi.png` |
| 10. Sınıf başına ortanca tepki | `results/figures/uyaran_sinif_tepkisi.png` |
| 11. Her sınıftan örnek pencereler | `results/figures/uyaran_ornek_pencereler.png` |

## 4. Ön İşleme

Ayrıntılı gerekçe: `docs/girdi_uyarlama_plani.md`.

1. **Okuma ve temizlik:** parça dosyalarını birleştirme, boş satırları atma, çakışan zaman
   damgalarını tekilleştirme (`src/bitki_ekg/data.py`).
2. **Kalite süzgeci:** verinin <%80'i dolu pencereleri atma (sarmaşık).
3. **Pencereleme:** domateste yazarların pencereleri (1 dk–6 sa) doğrudan kullanılabilir;
   karşılaştırma için ağırlıklı olarak 30 dk ve 1 sa.
4. **Normalizasyon:** pencere başına robust z-skor, `(x − medyan) / IQR` (Buss vd., 2026 ile
   uyumlu); öznitelik tabanlı modellerde min-max (yalnızca eğitim verisinden).
5. **EKG modelleri için:** pencereyi modelin girdi uzunluğuna yeniden örnekleme (ECG-FM:
   2 500 örnek; HuBERT-ECG: 100 Hz) ve tek kanalı 12 derivasyona eşleme.
6. **Bölme:** bitki bazlı — 12 domates bitkisiyle **bitki-dışarıda-bırak (LOPO)** çapraz
   doğrulama; karşılaştırma için yazarların bölmesi ve rastgele bölme de raporlanacak.
   Ölçekleyiciler yalnızca eğitim katmanında öğrenilecek (veri sızıntısını önlemek için).

## 5. Literatür Taraması Özeti

| Çalışma | Veri | Yöntem | Bölme | En iyi sonuç |
|---|---|---|---|---|
| Buss vd. (2026) | Domates sulama stresi (16 bitki, 4 sulama grubu) — **bu projenin ana veri seti** | 1 dk–6 sa pencere, ~700 tsfresh özniteliği + NaiveAutoML (HGB), MI + SBS öznitelik seçimi, sıcaklık ölçekleme; DL: CNN, InceptionTime, Mamba (Optuna) | Test: 2 görülmemiş bitki; eğitim/doğrulama: kalan 10 bitkide rastgele 80/20 | Test doğruluğu HGB %62,6 (1 dk) – %89,6 (6 sa); 30 dk %83,2. En iyi DL: CNN 6 sa %97,0. Doğrulama %92'ye kadar. Stres 4. günde tespit |
| Buss vd. (2025) | Sarmaşık, dış ortam (4 bitki, 5 ay) — **bu projenin veri seti** | 1 sa pencere, z-skor, 700+ tsfresh özniteliği, SMOTE; NB, kNN, doğrusal SVM, MLP, RF, AutoML | Rastgele %80/%20 (bitki/zaman bazlı değil) | Ort. makro F1: RF %90,7, AutoML %89,6; en iyi yağmurlu/kuru %95,5. NB %31,9–63,4, kNN %49,1–69,4 |
| Buss vd. (2023) | Zamioculcas potansiyeli, domates empedansı — **bu projenin veri seti** | 9 istatistiksel öznitelik + SFS ile 5 diskriminant analizi; 10 derin sınıflandırıcı (MLP, FCN, ResNet, Inception, Encoder vb.) | Rastgele %70/%30 (pencere düzeyinde) | DA: 2 sınıf %100, 5 sınıf %99,1 (QDA). DL: 2 sınıf Inception %89,7; 3 sınıf %92,2; 5 sınıf FCN/ResNet %83,5 |
| Najdenovska vd. (2021) | 36 ticari domates; kuraklık, besin eksikliği, örümcek akarı (PhytlSigns) | 34 istatistiksel öznitelik, çoklu pencere, XGBoost | — | 1 dk pencerede %85'e kadar doğruluk |
| González i Juclà vd. (2023) | 16 domates, azot eksikliği, 15 gün (PhytlSigns) | 4 derin öğrenme mimarisi, 1–30 s pencere | — | Encoder modeli %99'a kadar doğruluk; sınıflandırıcı güveninde normal→stresli geçişi |
| Chatterjee vd. (2015) | Domates ve salatalık; NaCl, H₂SO₄, O₃ (PLEASED) | 11 istatistiksel öznitelik, 5 diskriminant analizi | Birini-dışarıda-bırak (örnek düzeyinde) | En iyi ikili: H₂SO₄–O₃ >%94 (QDA, dalgacık entropisi); ortalamalar %62–70 |
| Chatterjee vd. (2018) | Aynı PLEASED verisi | Eğri uydurma (polinom, Gauss, Fourier, üstel) katsayıları öznitelik; LDA, QDA | — | >%90 doğruluk |
| Bhadra vd. (2023) | Domates ve lahana; NaCl, H₂SO₄, O₃ (PLEASED, ~38 000 blok, dengesiz) | 15 istatistiksel öznitelik, Monte Carlo alt örnekleme, 10 sınıflandırıcı | Rastgele %50/%50, 1000 tekrar | AdaBoost dengeli doğruluk 0,71, F1 0,71 |
| Wahid vd. (2026) | 182 çalışmalık derleme | — | — | EKG/EEG → bitki transferi doğrulanmamış hipotez; rastgele bölme yaygın |

**Çıkarımlar:**
- Bu alanda klasik öznitelik tabanlı yöntemler (HGB, RF, diskriminant analizi) derin öğrenmeyle
  rekabet ediyor veya onu geçiyor; temel sebep veri azlığı.
- Rastgele/örnek düzeyinde bölme çok yaygın ve çok yüksek doğruluklar (%95–100) üretiyor;
  görülmemiş bitkilerde (Buss vd., 2026) doğruluk belirgin biçimde düşüyor (ör. 1 dk'da
  doğrulama %92 → test %63).
- Taranan çalışmaların hiçbiri EKG/EEG ile önceden eğitilmiş model kullanmamış.
- Bu proje üç boşluğu hedefler: biyosinyal temel modeli transferi, kapsamlı bitki bazlı
  değerlendirme (12 bitkiyle LOPO), türler arası genelleme (domates ↔ sarmaşık).

## 6. Kullanılacak Yöntemler

| Grup | Model | Veriye uygunluk gerekçesi |
|---|---|---|
| Uygun olmayan | kNN, Naive Bayes | kNN ham pencerede zamansal kaymaya duyarlı ve bitkiler arası genlik farkından etkilenir; NB öznitelik bağımsızlığı varsayar (tsfresh öznitelikleri güçlü biçimde ilişkili). Literatürde bu veride en düşük sonuçlar (NB %31,9–63,4, kNN %49,1–69,4) |
| Klasik | tsfresh + Gradient Boosting (LightGBM) | Ana veri setindeki en iyi yöntemle (HGB) aynı aile |
| Derin | ROCKET, InceptionTime | Zaman serisi sınıflandırmada güçlü taban çizgileri; InceptionTime ana veri setinde raporlanmış |
| Transfer | HuBERT-ECG, ECG-FM (+ rastgele başlatılmış aynı mimari kontrolü) | Projenin yenilik iddiası; kontrol, kazancın ön eğitimden geldiğini ayırmak için zorunlu |

**Metrikler:** doğruluk, F1 (sarmaşıkta makro F1), ROC-AUC; bitki bazlı katmanlar üzerinden
ortalama ± standart sapma.

## 7. Kısıtlar ve Açık Konular
- Ana veri setinde yalnızca 16 bitki (eğitim/test için 12) → sonuçlar güven aralığıyla verilecek.
- Sağlıklı/stresli etiketleri deneyin başı ve sonuyla çakışıyor (zaman karıştırıcısı, §3.1).
- Sarmaşıkta su stresi etiketi yok; vekil etiket (yağmurlu/kuru) ve sınıf dengesizliği.
- Uyaran verisindeki ışık sınıflarında olası artefakt (§3.3).
- Toprak nemi ana veri setinin Zenodo kaydında yok.

## Kaynakça

Bhadra, N., Chatterjee, S. K., & Das, S. (2023). Multiclass classification of environmental
chemical stimuli from unbalanced plant electrophysiological data. *PLOS ONE, 18*(5), e0285321.
https://doi.org/10.1371/journal.pone.0285321

Buss, E., Aust, T., & Hamann, H. (2025). When plants respond: Electrophysiology and machine
learning for green monitoring systems. In *Biomimetic and Biohybrid Systems* (LNCS, s. 249–261).
Springer. https://doi.org/10.1007/978-3-032-07448-5_21

Buss, E., Aust, T., Wahby, M., Rabbel, T.-L., Kernbach, S., & Hamann, H. (2023). Stimulus
classification with electrical potential and impedance of living plants: Comparing discriminant
analysis and deep-learning methods. *Bioinspiration & Biomimetics, 18*(2), 025003.
https://doi.org/10.1088/1748-3190/acbad2

Buss, E., Aust, T., & Hamann, H. (2026). Early detection of water stress by plant
electrophysiology: Machine learning for irrigation management. arXiv:2604.28038.
https://doi.org/10.48550/arXiv.2604.28038

Chatterjee, S. K. vd. (2015). Exploring strategies for classification of external stimuli using
statistical features of the plant electrical response. *Journal of the Royal Society Interface,
12*, 20141225. https://doi.org/10.1098/rsif.2014.1225

Chatterjee, S. K., Malik, O., & Gupta, S. (2018). Chemical sensing employing plant electrical
signal response—Classification of stimuli using curve fitting coefficients as features.
*Biosensors, 8*(3), 83. https://doi.org/10.3390/bios8030083

Coppola, E. vd. (2024). HuBERT-ECG: A self-supervised foundation model for broad and scalable
cardiac applications. medRxiv. https://doi.org/10.1101/2024.11.14.24317328

González i Juclà, D., Najdenovska, E., Dutoit, F., & Raileanu, L. E. (2023). Detecting stress
caused by nitrogen deficit using deep learning techniques applied on plant electrophysiological
data. *Scientific Reports, 13*(1), 9633. https://doi.org/10.1038/s41598-023-36683-3

McKeen, K. vd. (2025). ECG-FM: An open electrocardiogram foundation model. *JAMIA Open, 8*(5),
ooaf122. https://doi.org/10.1093/jamiaopen/ooaf122

Najdenovska, E., Dutoit, F., Tran, D. vd. (2021). Identifying general stress in commercial
tomatoes based on machine learning applied to plant electrophysiology. *Applied Sciences,
11*(12), 5640. https://doi.org/10.3390/app11125640

Wahid, A. M., Baihaqi, W. M., & Nambo, H. (2026). Transformer adaptation for plant bioelectric
potential classification: A scoping review and analogical transfer framework. *Smart
Agricultural Technology, 15*, 102484. https://doi.org/10.1016/j.atech.2026.102484
