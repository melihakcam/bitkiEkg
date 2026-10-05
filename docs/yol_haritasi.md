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
| Domates | `AdditionalMaterial.zip` | 9,7 GB (açılmış 22,8 GB) | ✅ İndi, doğrulandı, açıldı (2026-10-05) |

- **İndirilmeyecek:** `DeepClassifier.zip` (4,2 GB, yalnızca yazarların modelleri).

## Aşama 2 — Veriyi tanıma 🟡
- ✅ Uyaran: UCR formatı (512 örneklik pencere, 5 sınıf) + ham `UzL/*.csv` (2 kanal, ~0,5 Hz,
  toprak nemi sütunu var) → `docs/veri_notlari.md`
- ✅ Sarmaşık: `datetime, CH1, CH2`, 1 Hz; kapsama %50–59; iki dosyada ~19 milyon boş satır
- ✅ Sarmaşık makalesi okundu (Buss vd., 2025): elektrot yerleşimi, etiket eşikleri, rastgele
  bölme, tüm sonuçlar → notlar ve Faz 1 raporu
- ✅ Domates: yapı, 16 bitki / 8 cihaz, 1 Hz, sınıflar 0/1/3, yazarların sonuçları → notlar
- ✅ Domates makalesi (Buss 2026) okundu: 4 sulama grubu, etiketler, 2 bitkilik test, tüm sonuçlar
- ✅ Buss 2023 makalesi okundu → Faz 1 literatür tablosu dolduruldu
- ✅ Domates EDA ilk grafikleri (`scripts/eda_domates.py` → `results/figures/domates_*.png`)
- 🟡 Model girdi özellikleri: ECG-FM tam doğrulandı (500 Hz, 5 s, 12 derivasyon, z-skor);
  HuBERT-ECG kısmen (100 Hz, 12 derivasyon; segment süresi ve derivasyon dizilişi Colab'da koddan)

## Aşama 3 — EDA (Faz 1 raporu) ✅
- ✅ Yükleyiciler: `src/bitki_ekg/data.py` — domates, sarmaşık, hava durumu
- ✅ Domates: `scripts/eda_domates.py` (4 şekil + bitki özet tablosu)
- ✅ Sarmaşık: `notebooks/01_sarmasik_ilk_bakis.ipynb` (4 şekil), `scripts/eda_sarmasik_etiket.py`
  (makale eşikleriyle sınıf dağılımı)
- ✅ Uyaran: `scripts/eda_uyaran.py` (3 şekil; etiket eşlemesi yazar kodundan doğrulandı;
  ışık sınıflarında olası artefakt)
- ✅ Faz 1 raporu dolduruldu (`reports/faz1/faz1_rapor.md`); ders şablonu gelince aktarılacak

## Aşama 4 — Ön işleme 🟡
- ✅ `src/bitki_ekg/preprocessing.py`: domates pencere kümeleri (1 sa, 30 dk), robust z-skor,
  LOPO ve yazar bölmesi
- ⬜ `src/bitki_ekg/preprocessing.py`: kalite süzgeci, detrend, pencereleme, yeniden örnekleme, z-skor
- ⬜ `src/bitki_ekg/splits.py`: bitki bazlı GroupKFold + karşılaştırma için rastgele bölme
- ⬜ Sarmaşık için ortak etiket kararı: `rain_dry` (öneri) — buharlaşma sütunu %99 boş olduğu
  için türetilmiş kuraklık etiketi pek mümkün değil
- ⬜ İşlenmiş pencereler → `data/processed/` (parquet / npy)

## Aşama 5 — Temel modeller ✅ (yerelde) → `docs/sonuclar.md`
- ✅ kNN, Naive Bayes (uygun olmayan): ~%50
- ✅ tsfresh + LightGBM: LOPO %72,9 · yazar bölmesi %87,2 (yazarlar %84,0) · rastgele %90,9
- ✅ MiniRocket: LOPO %72,5–73,1
- ✅ Zaman karıştırıcısı testi: kontrol bitkilerinde 30 dk'da şans düzeyi
- ⬜ MiniRocket AUC düzeltmesi ve hızlandırma (n_jobs, dönüşüm önbelleği)

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
- ~~Öneride 10 Hz yazıyor~~ → cihaz 10 Hz, veri seti 1 Hz; raporda ikisi de belirtildi
- Etiket ifadesi: "su stresi" yerine "sulama stresi" (aşırı sulanan grup da stresli sayılıyor)
- Sarmaşık ortak etiketi (Aşama 4) — öneri: yağmurlu/kuru
- Literatür: aynı veri setlerini kullanan 5 kaynak şartı (şu an 3 Buss çalışması) — ek kaynak
  aranacak ya da hocaya danışılacak
- `main` dalının GitHub'a gönderilip gönderilmeyeceği
