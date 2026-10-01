# İnsan EKG'sinden Bitkiye: Biyosinyal Temel Modelleri ile Bitki Elektrofizyolojisinde Su Stresi Tespiti

Uygulamalı Yapay Zeka dersi dönem projesi · Bireysel · Mehmet Melih Akçam
Kategori: Çok değişkenli zaman serileri

## Amaç

Bitki gövdesinden ölçülen elektrik potansiyeli zaman serisi pencerelerinden su stresinin
varlığını ikili olarak sınıflandırmak. İnsan EKG verisiyle önceden eğitilmiş biyosinyal temel
modelleri (ECG-FM, HuBERT-ECG) bitki sinyaline aktarılarak klasik ve derin yöntemlerle
karşılaştırılır. Değerlendirme, eğitimde görülmemiş bitkiler (bitki bazlı bölme) ve veri setleri
arası testlerle yapılır.

## Veri setleri

| Veri seti | İçerik | Link |
|---|---|---|
| Early Detection of Water Stress by Plant Electrophysiology (Buss vd., 2026), Zenodo v2 | Domates, 16 bitki, 18 gün, 10 Hz | https://doi.org/10.5281/zenodo.18876513 |
| When Plants Respond: Electrophysiology and ML for Green Monitoring Systems (Buss vd., 2025), Zenodo v1 | Sarmaşık, 5 ay dış ortam | https://doi.org/10.5281/zenodo.15095523 |
| Stimulus classification with electrical potential and impedance of living plants (Buss vd., 2023), Zenodo v1 | Domates, Zamioculcas; rüzgâr, ısı, ışık | https://doi.org/10.5281/zenodo.7126105 |

Veri dosyaları depoya eklenmez; `data/raw/` altına indirme betikleriyle indirilir.

## Modeller

| Grup | Modeller |
|---|---|
| Uygun olmayan | kNN, Naive Bayes |
| Klasik | tsfresh öznitelikleri + Gradient Boosting |
| Derin | ROCKET, InceptionTime |
| Transfer | ECG-FM, HuBERT-ECG (ince ayar) |

Metrikler: doğruluk, F1, ROC-AUC.

İlerleme ve yapılacaklar: [`docs/yol_haritasi.md`](docs/yol_haritasi.md)

## Klasör yapısı

```
configs/          Proje ayarları (config.yaml)
data/raw/         İndirilen ham veri (git dışı)
data/interim/     Ara çıktılar (git dışı)
data/processed/   Modele hazır veri (git dışı)
notebooks/        Keşifsel analiz not defterleri
scripts/          İndirme, ön işleme ve eğitim betikleri
src/bitki_ekg/    Ortak Python kodu
models/           Eğitilmiş modeller (git dışı)
results/          Şekiller ve tablolar
reports/          Faz raporları ve makale
```

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
```

Derin öğrenme aşaması için (GPU önerilir, Colab/Kaggle): `pip install -r requirements-dl.txt`
