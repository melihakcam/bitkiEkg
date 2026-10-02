# Yol Haritası

Durum işaretleri: ✅ bitti · 🟡 kısmen · ⏳ sıradaki · ⬜ bekliyor
Son güncelleme: 2026-10-02

## Aşama 0 — Altyapı ✅
- ✅ Klasör yapısı, README, .gitignore, requirements, config
- ✅ Sanal ortam ve paketler (`.venv`, `openpyxl` dahil)
- ✅ İndirme betiği (`scripts/download_data.py`): devam ettirme, MD5, `--peek`, `--extract`,
  bağlantı kopmalarında otomatik yeniden deneme, Windows için `:` → `-` dosya adı düzeltmesi
- ✅ GitHub: `melihakcam/bitkiEkg`, dal `onDeneme`
- ✅ Uyarlama planı (`docs/girdi_uyarlama_plani.md`), Faz 1 taslağı, veri notları

## Aşama 1 — Veri indirme 🟡

| Veri seti | Dosya | Boyut | Durum |
|---|---|---|---|
| Uyaran | `SupplementaryCode.zip` + küçük dosyalar | 124 MB | ✅ İndi, doğrulandı, açıldı |
| Sarmaşık | `Plant_data.zip` + hava durumu + feature_selection | 424 MB | ✅ İndi, doğrulandı, açıldı |
| Domates | `AdditionalMaterial.zip` | 10,2 GB | ⏸️ Durduruldu (85 MB indi, `.part` olarak duruyor) |

- **İndirilmeyecek:** `DeepClassifier.zip` (4,2 GB, yalnızca yazarların modelleri).
- Hat hızı ~200–900 KB/s; domates bu hızla 3–14 saat sürer. Seçenekler:
  1. İçerik listesine bakmak (35 MB) → yalnızca ham sinyal dosyalarını zip'ten seçerek çekmek
  2. Colab'a doğrudan Zenodo'dan indirtmek, orada işleyip küçültmek
  3. Kaldığı yerden devam: `.venv\Scripts\python scripts\download_data.py domates_su_stresi --extract`

## Aşama 2 — Veriyi tanıma 🟡
- ✅ Uyaran: UCR formatı (512 örneklik pencere, 5 sınıf) + ham `UzL/*.csv` (2 kanal, ~0,5 Hz,
  toprak nemi sütunu var) → `docs/veri_notlari.md`
- ✅ Sarmaşık: `datetime, CH1, CH2`, 1 Hz; kapsama %50–59; iki dosyada ~19 milyon boş satır
- ✅ Sarmaşık makalesi okundu (Buss vd., 2025): elektrot yerleşimi, etiket eşikleri, rastgele
  bölme, tüm sonuçlar → notlar ve Faz 1 raporu
- ⬜ Domates: dosya yapısı, su stresi etiketi, bitki kimliği (veri gelince)
- ⬜ `classification_results.xlsx` → Buss 2023 satırındaki boşluklar
- ⬜ Uyaran ve domates makalelerini okumak (Buss 2023, Buss 2026)
- ⬜ Model girdi özelliklerini doğrula (ECG-FM, HuBERT-ECG model kartları)

## Aşama 3 — EDA (Faz 1 raporu) 🟡
- 🟡 Yükleyiciler: `src/bitki_ekg/data.py` — sarmaşık bitki + hava durumu ✅; uyaran ⬜; domates ⬜
- 🟡 Not defterleri: `01_sarmasik_ilk_bakis.ipynb` yazıldı (henüz çalıştırılmadı);
  uyaran ⬜; domates ⬜
- ⬜ Örnek/pencere sayıları, sınıf dağılımı, eksik veri, bitki bazlı dağılımlar
- ⬜ Şekiller → `results/figures/`, tablolar → `results/tables/`
- ⬜ Faz 1 raporundaki `[DOLDURULACAK]` alanları

## Aşama 4 — Ön işleme ⬜
- ⬜ `src/bitki_ekg/preprocessing.py`: kalite süzgeci, detrend, pencereleme, yeniden örnekleme, z-skor
- ⬜ `src/bitki_ekg/splits.py`: bitki bazlı GroupKFold + karşılaştırma için rastgele bölme
- ⬜ Sarmaşık için ortak etiket kararı: `rain_dry` (öneri) — buharlaşma sütunu %99 boş olduğu
  için türetilmiş kuraklık etiketi pek mümkün değil
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
- Domates verisi nasıl alınacak (Aşama 1: seçerek indirme / Colab / tamamını indirme)
- Sarmaşık ortak etiketi (Aşama 4) — öneri: yağmurlu/kuru
- Literatür: aynı veri setlerini kullanan 5 kaynak şartı (şu an 3 Buss çalışması) — ek kaynak
  aranacak ya da hocaya danışılacak
- `main` dalının GitHub'a gönderilip gönderilmeyeceği
