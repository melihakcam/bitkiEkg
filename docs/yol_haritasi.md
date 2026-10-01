# Yol Haritası

Durum işaretleri: ✅ bitti · ⏳ sıradaki · ⬜ bekliyor

## Aşama 0 — Altyapı ✅
- ✅ Klasör yapısı, README, .gitignore, requirements, config
- ✅ Sanal ortam ve paketler (`.venv`)
- ✅ İndirme betiği (`scripts/download_data.py`): devam ettirme, MD5, `--peek`, `--extract`
- ✅ GitHub: `melihakcam/bitkiEkg`, dal `onDeneme`
- ✅ Uyarlama planı (`docs/girdi_uyarlama_plani.md`), Faz 1 taslağı, veri notları
- ✅ Küçük dosyalar indirildi (hava durumu, feature_selection, classification_results, kod arşivleri)

## Aşama 1 — Veri indirme ⏳ (internet gerekir)

`D:\ekg` klasöründe, sırayla çalıştırılır. Kesilen komut aynen tekrar çalıştırılırsa kaldığı
yerden devam eder.

| # | Komut | Boyut | Not |
|---|---|---|---|
| 1 | `.venv\Scripts\python -m pip install openpyxl` | ~0,3 MB | Excel sonuç tablosu için |
| 2 | `.venv\Scripts\python scripts\download_data.py uyaran_siniflandirma --files SupplementaryCode.zip --extract` | 124 MB | Uyaran ham verisi |
| 3 | `.venv\Scripts\python scripts\download_data.py sarmasik_dis_ortam --files Plant_data.zip --extract` | 423 MB | Sarmaşık, 4 bitki |
| 4 | `.venv\Scripts\python scripts\download_data.py domates_su_stresi --extract` | 10,2 GB | Ana veri; en uzun süren |

- **İndirilmeyecek:** `DeepClassifier.zip` (4,2 GB, yalnızca yazarların modelleri).
- Domates (4) çok uzun sürerse alternatif: Colab'a doğrudan Zenodo'dan çektirmek
  (aynı betik Colab'da da çalışır).
- Disk: zip'ler + açılmış hali ≈ 25–30 GB (boş yer 620 GB).

## Aşama 2 — Veriyi tanıma ⬜
- ⬜ Her veri setinde dosya formatı, sütunlar, kanal sayısı, örnekleme hızı
- ⬜ Domates: su stresi etiketi nasıl tanımlı, bitki kimliği nerede
- ⬜ Uyaran: `train.tsv` / `UzL/*.csv` yapısı
- ⬜ Sarmaşık: sinyal + hava durumu zaman eşlemesi
- ⬜ `classification_results.xlsx` → Faz 1 literatür tablosundaki boşluklar
- ⬜ Model girdi özelliklerini doğrula (ECG-FM, HuBERT-ECG model kartları)

## Aşama 3 — EDA (Faz 1 raporu) ⬜
- ⬜ Yükleyiciler: `src/bitki_ekg/data.py` (her veri seti için ortak tablo formatı)
- ⬜ `notebooks/01_eda_domates.ipynb`, `02_eda_sarmasik`, `03_eda_uyaran`
- ⬜ Örnek/pencere sayıları, sınıf dağılımı, eksik veri, bitki bazlı dağılımlar
- ⬜ Şekiller → `results/figures/`, tablolar → `results/tables/`
- ⬜ Faz 1 raporundaki `[DOLDURULACAK]` alanları

## Aşama 4 — Ön işleme ⬜
- ⬜ `src/bitki_ekg/preprocessing.py`: kalite süzgeci, detrend, pencereleme, yeniden örnekleme, z-skor
- ⬜ `src/bitki_ekg/splits.py`: bitki bazlı GroupKFold + karşılaştırma için rastgele bölme
- ⬜ Sarmaşık için ortak etiket kararı: `rain_dry` mi, yağış/buharlaşmadan türetilmiş kuraklık mı?
- ⬜ İşlenmiş pencereler → `data/processed/` (parquet / npy)

## Aşama 5 — Temel modeller ⬜ (yerelde)
- ⬜ kNN, Naive Bayes (uygun olmayan)
- ⬜ tsfresh + LightGBM (klasik)
- ⬜ ROCKET (aeon)
- ⬜ Ortak değerlendirme: doğruluk, F1, ROC-AUC; katman ortalaması ± std

## Aşama 6 — Derin ve transfer modeller ⬜ (Colab/Kaggle)
- ⬜ InceptionTime
- ⬜ HuBERT-ECG, ECG-FM: doğrusal sonda + tam ince ayar
- ⬜ Aynı mimariler rastgele başlatılmış (kontrol)
- ⬜ Ablasyonlar: pencere süresi, kanal eşleme

## Aşama 7 — Genelleme ve raporlama ⬜
- ⬜ Rastgele bölme vs bitki bazlı bölme farkı
- ⬜ Domates → sarmaşık, sarmaşık → domates
- ⬜ Literatürle karşılaştırma, görselleştirme
- ⬜ Faz 2 makale formatında rapor, sunum

## Açık kararlar
- Sarmaşık ortak etiketi (Aşama 4)
- Literatür: aynı veri setlerini kullanan 5 kaynak şartı (şu an 3 Buss çalışması) — ek kaynak
  aranacak ya da hocaya danışılacak
- `main` dalının GitHub'a gönderilip gönderilmeyeceği
