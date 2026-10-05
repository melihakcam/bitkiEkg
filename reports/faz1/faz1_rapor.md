# Faz 1 İlerleme Raporu (Taslak)

**Başlık:** İnsan EKG'sinden Bitkiye: Biyosinyal Temel Modelleri ile Bitki Elektrofizyolojisinde
Su Stresi Tespiti
**Proje türü:** Bireysel · Mehmet Melih Akçam
**Kategori:** Çok değişkenli zaman serileri

> Bu taslak, dersin Faz 1 şablonu paylaşıldığında o şablona aktarılacaktır.
> `[DOLDURULACAK]` işaretli yerler veri indirildikten sonra EDA çıktılarıyla doldurulacaktır.

---

## 1. Problem Tanımı

Bitkiler su stresi gibi olumsuz koşullarda gövdelerinde ölçülebilir elektrik potansiyeli
değişimleri üretir. Bu çalışmada problem, gövdeden ölçülen zaman serisi pencerelerinden su
stresinin varlığını **ikili sınıflandırma** olarak tespit etmektir. Temel güçlükler veri azlığı,
bitkiden bitkiye değişen sinyal ve literatürde yaygın olan rastgele veri bölmesinin sonuçları
iyimser göstermesidir.

## 2. Veri Setleri

| Veri seti (literatürdeki adı, sürüm) | İçerik | Link | Projedeki rolü |
|---|---|---|---|
| Early Detection of Water Stress by Plant Electrophysiology (Buss vd., 2026), Zenodo v2 | Domates, 16 bitki, 18 gün; cihazda 10 Hz, veri setinde 1 Hz; CC-BY | https://doi.org/10.5281/zenodo.18876513 | Ana veri seti (eğitim/test) |
| When Plants Respond: Electrophysiology and ML for Green Monitoring Systems (Buss vd., 2025), Zenodo v1 | Sarmaşık, 5 ay dış ortam; CC-BY | https://doi.org/10.5281/zenodo.15095523 | Veri setleri/türler arası test |
| Stimulus classification with electrical potential and impedance of living plants (Buss vd., 2023), Zenodo v1 | Domates, Zamioculcas; rüzgâr, ısı, ışık; CC-BY | https://doi.org/10.5281/zenodo.7126105 | Ek veri / ön eğitim adayı |

İndirme boyutları: domates 10,2 GB, sarmaşık 0,44 GB, uyaran 4,3 GB.
Veriler `scripts/download_data.py` ile indirilir ve MD5 ile doğrulanır.

## 3. Veri Seti Analizi

Her veri seti için aşağıdaki bilgiler raporlanacaktır.

### 3.1 Domates su stresi
- Dosya formatı ve yapısı: hazır pencereler (`datetime, CH1, CH2` CSV), tsfresh öznitelikleri,
  yazarların eğitim/doğrulama/test bölmesi ve sonuçları; 5 pencere süresi (1 dk, 5 dk, 30 dk, 1 sa, 6 sa)
- Bitki sayısı / kayıt süresi / örnekleme hızı: 16 bitki, 18 gün (04–21.06.2025);
  cihazda 10 Hz, veri setinde 1 Hz; birim mV
- Kanal sayısı ve kanalların anlamı: 8 PhytoNode cihazı, her cihazın CH1 ve CH2'si ayrı bir bitki
  (bitki başına tek kanal, gövdeye batırılmış 2 elektrot arası potansiyel farkı)
- Etiket tanımı: 04–08.06 tüm bitkiler 400 mL/gün; sonra 4 grup (kontrol 400 mL, aşırı sulanmış,
  200 mL, 100 mL). İlk 3 gün = sağlıklı (0), son 3 gün = stresli (1), aradaki günler ve kontrol
  bitkileri = 3 (eğitimde kullanılmıyor). **Aşırı sulanan grup da "stresli" sayıldığından etiket
  "sulama stresi"dir.**
- Toplam örnek / pencere sayısı: 1 sa pencerede 6 912 (16 bitki × 432); 1 dk'da 207 352 dosya
- Sınıf dağılımı (1 sa): sağlıklı 864 · stresli 864 · kullanılmayan 5 184 → ikili görev dengeli
- Eksik veri oranı, kopuk elektrot süreleri: [DOLDURULACAK]
- Şekiller: ham sinyal örnekleri, sınıf bazlı ortalama sinyal, bitki bazlı dağılımlar [DOLDURULACAK]

### 3.2 Sarmaşık dış ortam
- Ölçüm: PhytoNode cihazı, gümüş kaplı elektrotlar; gövde alt ucu ile 30–60 cm yukarısı
  (gövde veya yaprak sapı) arasındaki potansiyel farkı (Buss vd., 2025)
- Bitki sayısı / süre: 4 sarmaşık (P1, P2, P3, P5), 05.07–18.11.2024, Konstanz botanik bahçesi
- Örnekleme: cihazda ~200 Hz, veri setinde 1 sn ortalama ile 1 Hz; 2 kanal (CH1, CH2)
- Dosya yapısı: bitki başına 12 saatlik CSV parçaları (`datetime, CH1, CH2`), toplam 575 dosya
- Veri kapsaması: P1 %59 · P2 %50 · P3 %59 · P5 %53 (donanım iletişim sorunları);
  iki dosyada ~19 milyon boş satır var
- Etiket: veri setinde hazır etiket yok; hava durumu verisinden (10 dk) eşikle türetiliyor.
  **Su stresi etiketi yok** → veri setleri arası testte vekil etiket olarak yağmurlu/kuru
  kullanılması planlanıyor
- Sınıf dağılımı, pencere sayıları, şekiller: [DOLDURULACAK]

### 3.3 Uyaran sınıflandırma
- [DOLDURULACAK]

## 4. Ön İşleme

Ayrıntılı gerekçe: `docs/girdi_uyarlama_plani.md`.

1. Kalite süzgeci (kopuk elektrot, doygunluk, eksik aralıklar)
2. Pencere başına kayma giderme (detrend)
3. Pencereleme (10 dk, 1 sa, 6 sa)
4. Modelin girdi uzunluğuna yeniden örnekleme (EKG modelleri için)
5. Pencere başına z-skor ölçekleme
6. Bitki bazlı bölme (`GroupKFold`, grup = bitki kimliği)

Uygulanan adımların sonuçları (atılan pencere sayısı, son sınıf dağılımı): [DOLDURULACAK]

## 5. Literatür Taraması Özeti

| Çalışma | Veri | Yöntem | Bölme | En iyi sonuç |
|---|---|---|---|---|
| Buss vd. (2026) | Domates sulama stresi (16 bitki, 4 sulama grubu) | 1 dk–6 sa pencere, ~700 tsfresh özniteliği + NaiveAutoML (HGB), MI + SBS öznitelik seçimi, sıcaklık ölçekleme; DL: CNN, InceptionTime, Mamba (Optuna) | Test: 2 görülmemiş bitki; eğitim/doğrulama: kalan 10 bitkide rastgele 80/20 | Test doğruluğu HGB 62,6 (1 dk) – 89,6 (6 sa); 30 dk 83,2. En iyi DL: CNN 6 sa 97,0. Doğrulama %92'ye kadar. Stres 4. günde tespit |
| Buss vd. (2025) | Sarmaşık, dış ortam (4 bitki, 5 ay) | 1 sa pencere, z-skor, 700+ tsfresh özniteliği, SMOTE; NB, kNN, doğrusal SVM, MLP, RF, AutoML. Etiketler hava durumundan eşikle: gündüz/gece, yağmurlu/kuru, soğuk/sıcak, rüzgârlı/sakin | Rastgele %80/%20 (bitki/zaman bazlı değil) | Ort. makro F1: RF %90,7, AutoML %89,6; en iyi yağmurlu/kuru RF + öznitelik seçimi %95,5. NB %31,9–63,4, kNN %49,1–69,4 |
| Buss vd. (2023) | Zamioculcas (elektrik potansiyeli, ~0,58 Hz) ve domates (doku empedansı); rüzgâr, ısı, mavi/kırmızı ışık | 9 istatistiksel öznitelik + SFS ile 5 diskriminant analizi (LDA, QDA, naive Bayes LDA/QDA, Mahalanobis); 10 derin sınıflandırıcı (MLP, FCN, ResNet, Inception, Encoder vb.) | Rastgele %70 eğitim / %30 test (pencere düzeyinde) | DA: 2 sınıf %100, 5 sınıf %99,1 (QDA + SFS); empedans 2 sınıf %100. DL: 2 sınıf Inception %89,7; 3 sınıf Inception %92,2; 5 sınıf FCN/ResNet %83,5; MLP %57,4–73,2 |
| Chatterjee vd. (2015) | PLEASED | İstatistiksel öznitelik + diskriminant analizi | [DOLDURULACAK] | [DOLDURULACAK] |
| Chatterjee vd. (2018) | PLEASED | Eğri uydurma katsayıları öznitelik olarak | [DOLDURULACAK] | [DOLDURULACAK] |
| Bhadra vd. (2023) | PLEASED | 15 istatistiksel öznitelik, alt örnekleme, 10 sınıflandırıcı | [DOLDURULACAK] | 0,71 dengeli doğruluk (AdaBoost) |
| Wahid vd. (2026) | 182 çalışmalık derleme | — | — | EKG/EEG → bitki transferi doğrulanmamış hipotez; rastgele bölme yaygın |

**Çıkarımlar:**
- Bu alanda klasik öznitelik tabanlı yöntemler derin öğrenmeden güçlü; sebep veri azlığı.
- Taranan çalışmaların hiçbiri EKG/EEG ile önceden eğitilmiş model kullanmamış.
- Bitki bazlı bölme ve veri setleri arası test nadir; rastgele bölme sonuçları şişirebilir.
- Bu proje üç boşluğu hedefler: biyosinyal temel modeli transferi, bitki bazlı değerlendirme,
  türler arası genelleme.

## 6. Kullanılacak Yöntemler

| Grup | Model | Veriye uygunluk gerekçesi |
|---|---|---|
| Uygun olmayan | kNN, Naive Bayes | Ham pencerede zamansal yapıyı yok sayar; NB öznitelik bağımsızlığı varsayar |
| Klasik | tsfresh + Gradient Boosting (LightGBM) | Literatürdeki en güçlü yaklaşımla aynı aile |
| Derin | ROCKET, InceptionTime | Zaman serisi sınıflandırmada güçlü taban çizgileri |
| Transfer | ECG-FM, HuBERT-ECG (+ rastgele başlatılmış kontrol) | Projenin yenilik iddiası |

**Metrikler:** Doğruluk, F1, ROC-AUC (bitki bazlı katmanlar üzerinden ortalama ± std).

## Kaynakça

Bhadra, N., Chatterjee, S. K., & Das, S. (2023). Multiclass classification of environmental
chemical stimuli from unbalanced plant electrophysiological data. *PLOS ONE, 18*(5), e0285321.
https://doi.org/10.1371/journal.pone.0285321

Buss, E., Aust, T., & Hamann, H. (2025). When plants respond: Electrophysiology and machine
learning for green monitoring systems. In *Biomimetic and Biohybrid Systems* (LNCS, s. 249–261).
Springer. https://doi.org/10.1007/978-3-032-07448-5_21

Buss, E. vd. (2023). Stimulus classification with electrical potential and impedance of living
plants: Comparing discriminant analysis and deep-learning methods. *Bioinspiration &
Biomimetics, 18*(2), 025003. https://doi.org/10.1088/1748-3190/acbad2

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

McKeen, K. vd. (2025). ECG-FM: An open electrocardiogram foundation model. *JAMIA Open, 8*(5),
ooaf122. https://doi.org/10.1093/jamiaopen/ooaf122

Wahid, A. M., Baihaqi, W. M., & Nambo, H. (2026). Transformer adaptation for plant bioelectric
potential classification: A scoping review and analogical transfer framework. *Smart
Agricultural Technology, 15*, 102484. https://doi.org/10.1016/j.atech.2026.102484
