# Veri Notları

Zip içerikleri `scripts/download_data.py --peek` ile, dosyalar indirilmeden incelendi
(2026-10-02). Tam listeler: `data/interim/peek_*.txt` (git dışı).

## Domates su stresi (Zenodo 18876513)
- `AdditionalMaterial.zip`: 9,7 GB, MD5 doğrulandı (2026-10-05). Açılmış: 260 483 dosya,
  22,8 GB → `data/raw/domates_su_stresi/AdditionalMaterial/Tomato_Zenodo/`

### Klasör yapısı
| Klasör | İçerik | Boyut |
|---|---|---|
| `00_time_windows/Exp1/{1min,5min,30min,1h,6h}/` | Sinyal pencereleri, `<cihaz>_<tarih>_<saat>.csv` (ör. `PN10_2025-06-04_00-00.csv`) | 3,3 GB |
| `01_features/Exp1/<pencere>/` | tsfresh öznitelikleri: `all_features`, `features_with_class`, `filtered_features_with_class` | 14,1 GB |
| `02_test_train_val_split/Exp1/<pencere>/` | Yazarların `train/val/test.csv` bölmesi (1min train 2,1 GB) | 4,1 GB |
| `03_results/Exp1/<pencere>/` | AutoML / NaiveAutoML modelleri (`.joblib`), raporlar, karışıklık matrisleri | 1,3 GB |

Pencere sayıları (`00_time_windows`, dosya): 1min 207 352 · 5min 41 472 · 30min 6 912 ·
1h 3 456 · 6h 584.

### Sinyal
- Her pencere CSV'si: `datetime, CH1, CH2`, **1 Hz** (öneride yazılan 10 Hz değil — düzeltilecek).
  Değerler ~−50 civarı, büyük ihtimalle mV (doğrulanacak).
- **8 PhytoNode cihazı** (PN2, PN5, PN8, PN9, PN10, PN11, PN12, PN16); her cihazın CH1 ve
  CH2'si **iki ayrı bitki** → **16 bitki** (`plant_id` 0–15; ör. PN2 → 0 ve 1).
- Süre: 2025-06-04 – 2025-06-21 (18 gün).

### Etiketler (`01_features/.../features_with_class.csv`, 1 sa pencere)
Meta sütunlar: `plant_id, node, day, datetime_start, datetime_end, class`.

| Sınıf | Bitkiler / günler | Pencere (1 sa) | Yorum (çıkarım — makaleden doğrulanacak) |
|---|---|---|---|
| 0 | 12 bitki, ilk 3 gün (04–06.06) | 864 | Sulanmış, stressiz |
| 1 | Aynı 12 bitki, son 3 gün (19–21.06) | 864 | Su stresi |
| 3 | Aradaki 12 gün + PN8/PN9'daki 4 bitkinin (id 4–7) tamamı | 5 184 | Geçiş dönemi / kontrol bitkileri, ikili görevde kullanılmıyor |

- İkili görev dengeli: 864 / 864. Bitki bazlı bölme için 12 bitki kullanılabilir.
- Makale ile doğrulandı (aşağıda): PN8/PN9 bitkileri = 400 mL kontrol grubu.

### Yazarların sonucu (1 sa, AutoML, `03_results/Exp1/1h/`)
- Doğrulama: doğruluk 0,901, makro F1 0,901 (282 pencere)
- **Test doğruluğu 0,823** (288 pencere), eğitim doğruluğu 1,0

### Makaleden (Buss, Aust & Hamann, 2026 — arXiv 2604.28038)
- **Düzenek:** 16 domates, sera, 04–22.06.2025. PhytoNode cihazı başına 2 bitki; gümüş kaplı
  2 elektrot gövdeye batırılmış (biri tabanda, diğeri ≥30 cm yukarıda).
- **Örnekleme:** cihazda **10 Hz**, yazarlar **1 Hz'e** indirmiş (veri setindeki hal).
  Birim **mV** (EDP). Toprak nemi 0,1 Hz ölçülmüş ama indirdiğimiz zip'te yok.
- **Sulama:** 04–08.06 tüm bitkiler 400 mL/gün; sonra 4 grup × 4 bitki:

  | Grup | Sulama | Veri setinde |
  |---|---|---|
  | Kontrol | 400 mL/gün | PN8, PN9 (id 4–7), hep sınıf 3 |
  | Aşırı sulanmış | Sürekli kovada | sınıf 0 → 1 |
  | Orta kuraklık | 200 mL/gün | sınıf 0 → 1 |
  | Şiddetli kuraklık | 100 mL/gün | sınıf 0 → 1 |

  Hangi cihazın hangi gruba ait olduğu makalede yazmıyor (`02_test_train_val_split`'ten çıkarılabilir).
- **Etiket:** ilk 3 gün = sağlıklı (0), son 3 gün = stresli (1); aradaki günler (3) eğitimde
  kullanılmaz, geçişin zamanlamasını görmek için kullanılır.
  ⚠️ **Aşırı sulanmış grup da "stresli"** sayılıyor → etiket "su eksikliği" değil, "sulama stresi".
- **Bölme:** Test = **yalnızca 2 bitki** (1 aşırı sulanmış + 1 100 mL), kontroller dışarıda;
  kalan 10 bitki **rastgele 80/20** eğitim/doğrulama (aynı bitkiler her ikisinde) → doğrulama
  iyimser. Öznitelik seçiminde 5 katlı GroupKFold (her katta 2 bitki).
- **Yöntem:** tsfresh ~700 öznitelik, varyans < 0,01 atılır, min-max; NaiveAutoML → HGB
  (Histogram Gradient Boosting); MI ile ilk 200 + ardışık geriye eleme (SBS); sıcaklık ölçekleme
  ile kalibrasyon. DL: CNN, InceptionTime, Mamba (Optuna, 100 deneme, robust z-skor
  `(x − medyan) / IQR`, 5 tohum).
- **Pencere başına örnek (eğitim / doğrulama / test):** 1 dk 69 112 / 17 278 / 17 278 ·
  5 dk 13 822 / 3 458 / 3 456 · 30 dk 2 302 / 578 / 576 · 1 sa 1 158 / 282 / 288 · 6 sa 195 / 45 / 48.
- **Test doğruluğu (%):**

  | Pencere | HGB | HGB + SBS | CNN | InceptionTime | Mamba |
  |---|---|---|---|---|---|
  | 1 dk | 62,6 | 61,6 | 63,6 ± 1,2 | 62,3 ± 1,3 | 62,3 ± 0,8 |
  | 5 dk | 76,3 | 75,6 | 69,0 ± 2,0 | 71,0 ± 1,6 | 68,8 ± 2,0 |
  | 30 dk | 83,2 | 82,5 | 73,6 ± 2,8 | 71,1 ± 2,4 | 80,2 ± 3,9 |
  | 1 sa | 84,0 | 82,3 | 84,8 ± 2,9 | 77,3 ± 5,5 | 83,8 ± 3,0 |
  | 6 sa | 89,6 | 87,5 | 97,0 ± 1,1 | 88,3 ± 5,1 | 58,3 ± 11,9 |

  Doğrulama HGB: 92,2 / 92,6 / 91,0 / 90,1 / 77,8. Test AUPRC: 0,581 / 0,728 / 0,871 / 0,871 / 0,937.
- **Önerilen pencere:** 30 dk. **Erken tespit:** 100 ve 200 mL gruplarında 4. gün, aşırı
  sulanmışta ~6. gün; kontrol grubu eşiğin altında kalıyor.
- **Çok sınıflı (sağlıklı / aşırı / eksik sulama):** doğrulama %95, **test %48** → stres türü ayrılamıyor.
- **Yazarların kısıtları:** 16 bitki, tek sera/mevsim, yalnızca belirli sulama rejimleri;
  %50 eşiği fizyolojik başlangıç anlamına gelmiyor.

### Bizim proje için çıkarımlar
- Test yalnızca 2 bitki → **12 bitkiyle bitki-dışarıda-bırak (LOPO)** değerlendirme güçlü bir katkı.
- InceptionTime zaten raporlanmış → doğrudan kıyas noktası.
- Sürekli 1 Hz pencereler elimizde → kendi pencere sürelerimizi ve yeniden örneklemeyi seçebiliriz.
- Ek literatür (farklı veri setleri, aynı problem): Najdenovska vd. (2021) Applied Sciences
  11(12):5640 — 36 domates, XGBoost, 1 dk %85; González i Juclà vd. (2023) Sci Rep 13:9633 —
  16 domates azot eksikliği, derin öğrenme %99.

## Sarmaşık dış ortam (Zenodo 15095523)
- `Plant_data.zip`: 423 MB, 579 öğe, açılmış 1,9 GB.
  - Bitki klasörleri: P1 (179 dosya), P2 (147), P3 (138), P5 (115) → **4 bitki** (P4 yok).
  - Dosyalar 12 saatlik CSV parçaları: `P1_2024-07-27_12:00:00.csv` (~2,3 MB).
  - Dosya adlarında `:` var; Windows'ta açarken `-` ile değiştiriliyor.
- `weather_data.zip`: aylık CSV (2024-07 … 2024-11), Almanca başlıklar, 3 satırlık üst bilgi
  (Station / Kanal / ME). Kanallar: rüzgâr hızı ve yönü, hava sıcaklığı, bağıl nem,
  küresel ışınım, **yağış (Niederschlag)**, çiy noktası, buharlaşma.
- `feature_selection.zip`: yazarların 4 ikili görevi — `cold_warm`, `day_night`,
  `rain_dry`, `wind_calm`. tsfresh öznitelikleri + RandomForest; en iyi test makro F1:
  rain_dry 0,955 · cold_warm 0,916 · day_night 0,915 · wind_calm 0,878.
- ⚠️ **Doğrudan su stresi etiketi yok.** Veri setleri arası test için en yakın görev
  `rain_dry`; ya da yağış + buharlaşmadan bir "kuraklık" etiketi türetilebilir.

### İndirildikten sonra (Plant_data)
- Her CSV: `datetime, CH1, CH2`, 1 Hz, 12 saatlik parça (~43 200 satır). Değerler ham
  cihaz sayımı (~8 milyon civarı).
- ⚠️ `P1/P1_2024-08-10_12-00-00.csv` ve `P3/P3_2024-08-10_12-00-00.csv` ~400 MB:
  2024-01-01'den başlayan ~19 milyon **boş satır** içeriyor (Zenodo'dan böyle geliyor).
  Okurken boş satırlar atılmalı; Excel ile açılmamalı.
- Veri kapsaması (1 dk çözünürlükte, ilk–son ölçüm arası):
  P1 %59 (05.07–18.11) · P2 %50 (11.07–18.11) · P3 %59 (07.08–18.11) · P5 %53 (13.08–18.11).
- Hava durumu: 10 dk aralık, 01.07–30.11.2024; gece yarısı "24:00:00" olarak yazılmış;
  `Verdunstung Haude` (buharlaşma) %99 boş → kuraklık etiketi için kullanılamaz.

### Makaleden (Buss, Aust & Hamann, 2025 — arXiv 2506.23872)
- **Cihaz:** PhytoNode, gümüş kaplı elektrotlar. Bir elektrot gövdenin alt ucunda (toprağın
  hemen üstü), diğeri 30–60 cm yukarıda gövdede veya bir yaprak sapında.
- **Kanallar:** Makale sonuçları "STEM" (gövde) ve "LEAF" (yaprak) olarak ayrı veriyor;
  CH1/CH2 ile eşleşmesi açıkça yazılmamış (muhtemelen bunlar).
- **Örnekleme:** Cihaz ~200 Hz; yazarlar 1 sn ortalama ile 1 Hz'e indirmiş.
- **Birim:** Ham değerin mV karşılığı verilmemiş; yazarlar z-skor kullanmış.
- **Yer/süre:** Konstanz Üniversitesi botanik bahçesi, 05.07–18.11.2024, 4 sarmaşık
  (P4'ün yokluğu açıklanmamış).
- **Boşluklar:** Donanım iletişim sorunları; %80'den az dolu günler atılmış → 216 gün.
- **Etiketler (yazarlara göre keyfi eşikler):**

  | Görev | Kural |
  |---|---|
  | Gündüz / gece | Işınım > 50 W/m² → gündüz |
  | Yağmurlu / kuru | Yağış > 0 mm |
  | Soğuk / sıcak | 25 °C eşiği, yalnızca 08:00–20:00 |
  | Rüzgârlı / sakin | 1,25 m/s eşiği, yalnızca 08:00–20:00 |

- **Yöntem:** 1 sa pencere, z-skor, 700+ tsfresh özniteliği (min-max), SMOTE (k=5).
  **Rastgele %80/%20 bölme** (+ eğitimin %20'si doğrulama), 10 tabakalı karıştırmalı bölme.
  Bitki bazlı veya zaman bazlı bölme yok.
- **Makro F1 (%), gövde:**

  | Model | Rüzgârlı/sakin | Gündüz/gece | Yağmurlu/kuru | Soğuk/sıcak |
  |---|---|---|---|---|
  | Naive Bayes | 31,9 | 63,4 | 60,0 | 63,4 |
  | kNN (k=5) | 49,1 | 69,4 | 53,7 | 59,3 |
  | MLP | 80,5 | 82,1 | 88,5 | 82,4 |
  | Doğrusal SVM | 73,8 | 79,6 | 84,9 | 77,3 |
  | Random Forest (256 ağaç) | 87,9 | 92,8 | 93,8 | 88,5 |
  | AutoML | 90,8 | 93,8 | 89,3 | 84,7 |

  En iyi: yaprak, yağmurlu/kuru, RF + 49 öznitelik → **%95,5 ± 0,5**.
- **Su stresi / toprak nemi / kuraklık makalede hiç geçmiyor.**
- **Yazarların belirttiği kısıtlar:** yalnızca 4 bitki, tek mevsim ve konum, keyfi eşikler.

## Uyaran sınıflandırma (Zenodo 7126105)
- `DeepClassifier.zip` (4,2 GB): **yalnızca yazarların eğittiği modeller ve sonuçları**
  (`.hdf5`, `history.csv`, `y_pred.npy`). Ham sinyal yok → **indirilmeyecek.**
  Görevler: 2 sınıf (rüzgâr/uyaran yok), 3 sınıf (+ısı), 5 sınıf (+mavi/kırmızı ışık);
  modeller FCN, ResNet, Inception, 5 tekrar.
- `SupplementaryCode.zip` (124 MB, açılmış 515 MB): **ham sinyal burada** —
  `datasets/train.tsv`, `test_stat.tsv`, `impedance_post.tsv`, `UzL/Temp/lrpi0_*.csv` …
- `classification_results.xlsx`: sonuç tablosu (okumak için `openpyxl` gerekiyor).

### İndirildikten sonra (SupplementaryCode)
- `datasets/train.tsv`, `test_ANN.tsv` vb.: UCR formatı — 1. sütun etiket, ardından
  **512 örneklik pencere**. Eğitim 1302, test 558 pencere.
  Etiket dağılımı (eğitim): 0:381 · 1:353 · 2:381 · 3:92 · 4:95 (5 sınıf; 3 ve 4 azınlık,
  muhtemelen mavi/kırmızı ışık — doğrulanacak). `*_stat.tsv`: 680 sütunluk öznitelik sürümü.
- `datasets/UzL/{NoStimulus,Temp,Wind,BlueRedDatasets}/lrpi0_<zaman>.csv`: **ham ölçüm**,
  ~2 sn aralıkla (≈0,5 Hz), dosya başına ~4500 satır (~2,5 saat).
  Sütunlar: `differential_potential_CH1`, `differential_potential_CH2` (2 kanal bitki
  potansiyeli), `temp-external`, `light-external`, `humidity-external`, `transpiration`,
  `air_pressure`, **`soil_moisture`**, `soil_temperature`, `mag_X/Y/Z`, `RF_power_emission`.
  Değerler ham ADC sayımı (ör. 510625) → mV dönüşümü cihaz koduna (`mu_interface`) bakılarak yapılacak.
### Makaleden (Buss vd., 2023 — Bioinspir. Biomim. 18:025003, açık erişim)
- Elektrik potansiyeli: **Zamioculcas zamiifolia**, CYBRES phytosensor, ~0,58 Hz; uyaranlar
  rüzgâr, ısı, mavi ve kırmızı ışık. Doku empedansı: 3 **domates**, ~0,08 Hz, yalnızca ışık.
- Örnekler: uyaran başlangıcından itibaren 340 örnek (~9,8 dk), ön-uyaran dönemiyle arka plan
  çıkarma; empedansta 295 örnek (~60 dk).
- Bölme: **rastgele %70 eğitim / %30 test**, sınıf oranları korunarak (bitki/deney bazlı değil).
- DA: 9 öznitelik (ortalama, varyans, çarpıklık, basıklık, IQR, Hjorth hareketlilik/karmaşıklık,
  WPE, ASP) + SFS → 2 sınıf %100, 5 sınıf %99,1 (QDA), empedans 2 sınıf %100.
- DL (5 tekrar ortalaması, test): 2 sınıf Inception 89,7 · ResNet 89,4 · FCN 89,3 · MLP 73,2;
  3 sınıf Inception 92,2; 5 sınıf FCN/ResNet 83,5 · MLP 57,4.
- `classification_results.xlsx`: bu sonuçların ham tabloları (DA-BioPot, DA-Imp, DL-2/3/5classes).
- `soil_moisture` sütunu önemli: aynı laboratuvarın (Buss/Hamann) cihazı domates verisinde de
  kullanıldıysa su stresi etiketi toprak nemiyle doğrulanabilir.

### Domates grup eşlemesi (2026-10-07)
Kaynak: Buss vd., Zenodo 22081982 (2026-08-24 sürümü) `All_Plants_smoothed.png` alt grafik başlıkları;
dosya `data/raw/domates_v2/`. Aynı grafikte toprak nemi eğrileri de var (ham değerleri paylaşılmamış).

| Grup | Cihaz | plant_id |
|---|---|---|
| Aşırı sulanmış | PN2, PN5 | 0, 1, 2, 3 |
| Kontrol 400 mL | PN8, PN9 | 4, 5, 6, 7 |
| Orta kuraklık 200 mL | PN10, PN11 | 8, 9, 10, 11 |
| Şiddetli kuraklık 100 mL | PN12, PN16 | 12, 13, 14, 15 |

Her grup tam olarak 2 cihaz → grup ile cihaz birebir çakışıyor (cihaz düzeyinde değerlendirme şart).
Kuraklık gruplarında toprak nemi 9–11.06'dan itibaren düşüyor; aşırı sulamada ~%100 sabit.
