# Veri Notları

Zip içerikleri `scripts/download_data.py --peek` ile, dosyalar indirilmeden incelendi
(2026-10-02). Tam listeler: `data/interim/peek_*.txt` (git dışı).

## Domates su stresi (Zenodo 18876513)
- `AdditionalMaterial.zip`: 9,5 GB, **260 512 öğe**, içerik listesi 35,4 MB.
- İçerik listesi henüz çekilmedi (yavaş bağlantıda ~1 saat).

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

## Uyaran sınıflandırma (Zenodo 7126105)
- `DeepClassifier.zip` (4,2 GB): **yalnızca yazarların eğittiği modeller ve sonuçları**
  (`.hdf5`, `history.csv`, `y_pred.npy`). Ham sinyal yok → **indirilmeyecek.**
  Görevler: 2 sınıf (rüzgâr/uyaran yok), 3 sınıf (+ısı), 5 sınıf (+mavi/kırmızı ışık);
  modeller FCN, ResNet, Inception, 5 tekrar.
- `SupplementaryCode.zip` (124 MB, açılmış 515 MB): **ham sinyal burada** —
  `datasets/train.tsv`, `test_stat.tsv`, `impedance_post.tsv`, `UzL/Temp/lrpi0_*.csv` …
- `classification_results.xlsx`: sonuç tablosu (okumak için `openpyxl` gerekiyor).
