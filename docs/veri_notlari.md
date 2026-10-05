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
- Sınıf 3'ün ve PN8/PN9 bitkilerinin anlamı (kontrol grubu mu?) makaleden netleşecek.

### Yazarların sonucu (1 sa, AutoML, `03_results/Exp1/1h/`)
- Doğrulama: doğruluk 0,901, makro F1 0,901 (282 pencere)
- **Test doğruluğu 0,823** (288 pencere), eğitim doğruluğu 1,0
- Bölme stratejisi (bitki bazlı mı?) makaleden ve `02_test_train_val_split` içeriğinden doğrulanacak.

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
- `soil_moisture` sütunu önemli: aynı laboratuvarın (Buss/Hamann) cihazı domates verisinde de
  kullanıldıysa su stresi etiketi toprak nemiyle doğrulanabilir.
