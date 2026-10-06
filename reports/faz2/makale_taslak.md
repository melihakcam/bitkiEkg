# İnsan EKG'sinden Bitkiye: Biyosinyal Temel Modelleriyle Bitki Elektrofizyolojisinde Sulama Stresi Tespiti ve Değerlendirme Yanlılığı

**Mehmet Melih Akçam**
Uygulamalı Yapay Zeka Dersi, Dönem Projesi (Faz 2 taslağı)

> Taslak. Ders şablonu paylaşıldığında aktarılacak. `[SARMAŞIK]` işaretli yerler sarmaşık
> deneyi tamamlanınca doldurulacak. Tüm sayılar `docs/sonuclar.md` ve `results/` altındaki
> dosyalardan; kodlar `scripts/`, `src/bitki_ekg/`, `notebooks/`.

---

## Öz

Bitkiler sulama stresine gövdelerindeki elektrik potansiyeli değişimleriyle tepki verir; bu
sinyallerden stres tespiti, akıllı sulama için umut vericidir. Ancak veri azdır ve literatürdeki
yüksek başarılar çoğunlukla aynı bitkinin pencerelerinin hem eğitimde hem testte bulunduğu
rastgele bölmeyle elde edilmiştir. Bu çalışmada (i) değerlendirme yönteminin sonuçlara etkisi
ve (ii) insan EKG'siyle önceden eğitilmiş biyosinyal temel modellerinin (HuBERT-ECG, ECG-FM)
bitki sinyaline aktarılabilirliği sistematik olarak incelenmiştir. Ana veri setinde (16 domates,
18 gün) görülmemiş bitki üzerinde (bitki-dışarıda-bırak, 12 bitki) en iyi doğruluk %81,2'dir
(6 saatlik pencere, tsfresh + LightGBM); aynı model rastgele bölmede %85,9–92,2, yazarların
2 bitkilik testinde %95,7 vermektedir. EKG ön eğitimi 30 dk–1 sa pencerelerde katkı sağlamazken,
6 saatlik pencere modelin 5 saniyelik girdisine sıkıştırıldığında aynı mimarinin rastgele
başlatılmışına göre doğruluğu 12,7 puan (p = 0,010) artırmakta ve en iyi EKG tabanlı model
(%79,7) öznitelik tabanlı temel modelle istatistiksel olarak eşdeğer olmaktadır. Etiketsiz bitki
verisiyle ek ön eğitim (DAPT) anlamlı katkı sağlamamıştır. Bulgular, EKG temel modellerinin bitkiye
**doğru zaman ölçeğinde** aktarılabildiğini ve bitki elektrofizyolojisinde bitki bazlı
değerlendirmenin zorunlu olduğunu göstermektedir.

**Anahtar kelimeler:** bitki elektrofizyolojisi, sulama stresi, transfer öğrenme, biyosinyal temel
modeli, EKG, veri sızıntısı, bitki-dışarıda-bırak

## 1. Giriş

Tarımsal su kullanımının verimliliği, bitkinin fizyolojik durumunu doğrudan ölçen sensörlerle
artırılabilir. Bitki elektrofizyolojisi, gövdeye yerleştirilen elektrotlarla ölçülen potansiyel
farkı üzerinden stres tepkilerini izlemeyi sağlar (Buss vd., 2026; Najdenovska vd., 2021).
Makine öğrenmesiyle bu sinyallerden uyaran ve stres sınıflandırması yapan çalışmalar %85–100
arası başarılar raporlamaktadır (Chatterjee vd., 2015; Buss vd., 2023, 2025). Ancak bu
çalışmaların büyük kısmı pencereleri rastgele bölmekte, aynı bitkinin birbirine çok benzeyen
ardışık pencereleri hem eğitim hem test kümesine düşmektedir. Wahid vd. (2026) 182 çalışmalık
derlemelerinde bu sorunu vurgulamış ve insan EKG/EEG'si için geliştirilen transformer temelli
modellerin bitki sinyaline aktarılabileceğini **doğrulanmamış bir hipotez** olarak önermiştir.

Bu çalışmanın katkıları:
1. **Değerlendirme yanlılığının ölçülmesi:** Aynı modeller rastgele, yazar ve bitki-dışarıda-bırak
   (LOPO) bölmeleriyle değerlendirilerek iki türde (domates, sarmaşık) şişme miktarı gösterilmiştir.
2. **EKG temel modeli transferinin ilk deneysel testi:** İki model (HuBERT-ECG, ECG-FM), donmuş ve
   ince ayarlı, her biri aynı mimarinin rastgele başlatılmış kontrolüyle.
3. **Zaman ölçeğinin belirleyiciliği:** EKG ön eğitiminin yalnızca bitki penceresi uygun zaman
   ölçeğinde sıkıştırıldığında (6 sa → 5 s) katkı sağladığının gösterilmesi.
4. **Kontroller:** zaman karıştırıcısı testi (kontrol bitkileri), veri kalite denetimi ve
   etiketsiz bitki verisiyle alana uyarlama (DAPT).

## 2. İlgili Çalışmalar

| Çalışma | Veri | Yöntem | Bölme | Sonuç |
|---|---|---|---|---|
| Buss vd. (2026) | Domates sulama stresi, 16 bitki | tsfresh + AutoML (HGB); CNN, InceptionTime, Mamba | 2 bitki test, kalan rastgele | Test %62,6 (1 dk) – %89,6 (6 sa) |
| Buss vd. (2025) | Sarmaşık, dış ortam, 4 bitki | tsfresh + RF, AutoML | Rastgele %80/20 | Makro F1 %95,5'e kadar |
| Buss vd. (2023) | Zamioculcas, uyaranlar | Diskriminant analizi, 10 derin model | Rastgele %70/30 | DA %99–100, DL %83–92 |
| Najdenovska vd. (2021) | 36 domates, çoklu stres | 34 öznitelik + XGBoost | — | %85 (1 dk) |
| González i Juclà vd. (2023) | 16 domates, azot eksikliği | 4 derin mimari | — | %99'a kadar |
| Chatterjee vd. (2015, 2018) | PLEASED | İstatistik/eğri uydurma + DA | LOOCV (örnek) | %62–94 |
| Bhadra vd. (2023) | PLEASED | 15 öznitelik, 10 sınıflandırıcı | Rastgele 50/50 | Dengeli doğruluk 0,71 |
| Wahid vd. (2026) | 182 çalışmalık derleme | — | — | EKG/EEG → bitki transferi hipotezi |

EKG temel modelleri: HuBERT-ECG (Coppola vd., 2024; 9,1 M EKG, kendi kendine öğrenme) ve ECG-FM
(McKeen vd., 2025; 1,5 M EKG, wav2vec 2.0). Taranan bitki çalışmalarının hiçbiri EKG/EEG ile
önceden eğitilmiş model kullanmamıştır.

## 3. Yöntem

### 3.1 Veri setleri
- **Domates (ana veri; Buss vd., 2026; Zenodo 10.5281/zenodo.18876513):** 16 bitki, 8 PhytoNode
  cihazı (cihaz başına 2 bitki), 04–21.06.2025, cihazda 10 Hz, veri setinde 1 Hz (mV). İlk 4 gün
  400 mL/gün; ardından 4 grup × 4 bitki: kontrol 400 mL, aşırı sulanmış, 200 mL, 100 mL. Etiket:
  ilk 3 gün sağlıklı (0), son 3 gün stresli (1); aşırı sulanan grup da stresli sayıldığından görev
  **sulama stresi** tespitidir. Ara günler ve 4 kontrol bitkisi ikili görevde kullanılmaz.
- **Sarmaşık (Buss vd., 2025; 10.5281/zenodo.15095523):** 4 bitki, dış ortam, 05.07–18.11.2024,
  1 Hz, 2 kanal; etiketler hava durumundan eşikle (gündüz/gece: ışınım > 50 W/m²; yağmurlu/kuru:
  yağış > 0). Kapsama %50–59.
- **Uyaran (Buss vd., 2023; 10.5281/zenodo.7126105):** Zamioculcas, 5 uyaran; ışık sınıflarında
  olası artefakt nedeniyle yalnızca tanımlayıcı analizde kullanılmıştır.

### 3.2 Ön işleme ve veri kalitesi
- Pencereler: yazarların 30 dk, 1 sa ve 6 sa pencereleri; ikili görevde 12 sulama bitkisi
  (1 sa: bitki başına 144, 6 sa: 23).
- **Veri denetimi** (`scripts/veri_denetimi.py`): her bitkinin son penceresi kayıt bittiği için
  kısmen boştur ve hepsi "stresli" sınıftadır (6 sa'te %34 boş). Boşluğun sıfırla doldurulması
  "sıfır = stresli" kısayolu yaratabileceğinden %1'den fazla eksik pencereler çıkarılmıştır
  (6 sa'te sonucu %80,6'dan %77,9'a düşürmüştür; 1 sa/30 dk'da etki ihmal edilebilir).
- Normalizasyon: pencere başına robust z-skor `(x − medyan)/IQR`; sarmaşıkta bitki ve kanal
  başına z-skor (seviye bilgisini korumak için).

### 3.3 Modeller
- **Uygun olmayan:** kNN (k = 5), Gauss Naive Bayes (ham pencere).
- **Öznitelik tabanlı:** yazarların ~780 tsfresh özniteliği + LightGBM (300 ağaç, öğrenme oranı 0,05).
- **Derin, sıfırdan:** MiniRocket (aeon).
- **EKG temel modelleri:** HuBERT-ECG small (30,5 M) ve ECG-FM (90,9 M). Girdi dönüşümü:
  pencere modelin bir derivasyon uzunluğuna yeniden örneklenir (HuBERT: 500 örnek = 5 s @100 Hz;
  ECG-FM: 2 500 = 5 s @500 Hz) ve 12 derivasyona kopyalanır. Böylece pencere süresi modelin
  5 saniyesine sıkıştırılır; bitkideki dakika–saat ölçekli değişimler EKG'nin öğrendiği frekans
  bandına taşınır.
  - Donmuş: katman ortalamalı gömme + lojistik regresyon.
  - İnce ayar: ortalama havuzlama + doğrusal baş; AdamW (gövde 3e-5, baş 1e-3), 15 epoch, CNN
    öznitelik çıkarıcı dondurulmuş; en iyi epoch iç doğrulama bitkileriyle (2 bitki) seçilir.
  - **Kontrol:** aynı mimari rastgele ağırlıklarla (EKG ön eğitiminin katkısını ayırmak için).
  - **DAPT:** EKG (ve rastgele) ağırlıklarından başlayıp etiketsiz bitki verisiyle (kontrol
    domates + sarmaşık, 4 400 pencere; test bitkileri hariç) maskeli yeniden yapılandırma, 10 epoch.
- Tüm ayarlar deney öncesi sabittir; test bitkisine bakılarak değiştirilmemiştir. Rastgelelik
  tohumu 42; model sürümleri sabitlenmiştir.

### 3.4 Değerlendirme
- **LOPO:** her katmanda bir bitki test (12 katman) — asıl ölçüt.
- **Yazar bölmesi:** bitki 0 ve 12 test (Buss vd., 2026 ile karşılaştırma).
- **Rastgele:** tabakalı 5 katlı pencere bölmesi (literatürdeki uygulama).
- **Sarmaşıkta zaman bloğu:** takvim haftalarına göre 5 ardışık blok.
- Ölçütler: doğruluk, F1/makro F1, ROC-AUC. Karşılaştırmalar 12 bitkide eşleştirilmiş Wilcoxon
  işaretli sıralar testi ve 10 000 tekrarlı bootstrap %95 güven aralığı.
- **Zaman karıştırıcısı testi:** hiç strese girmeyen kontrol bitkilerinin ilk/son 3 günü aynı
  modelle ayırt edilmeye çalışılır.

## 4. Bulgular

### 4.1 Değerlendirme yöntemine göre şişme
| Model (domates) | Rastgele bölme | Yazar bölmesi | LOPO |
|---|---|---|---|
| tsfresh + LightGBM, 1 sa | 90,9 | 87,2 | **72,9** |
| tsfresh + LightGBM, 6 sa | 85,9 | 95,7 | **81,2** |
| MiniRocket, 1 sa | 79,7 | 81,9 | **72,5** |

Rastgele bölme görülmemiş bitkiye göre 5–18 puan, yazarların 2 bitkilik testi 3–14 puan daha
yüksek sonuç vermektedir. Yazar bölmesinde LightGBM 1 sa %87,2, yazarların raporu %84,0 →
boru hattımız yayımlanmış sonucu yeniden üretmektedir. Sarmaşıkta (gündüz/gece, makro F1):
rastgele %76,7 → bitki-dışarıda-bırak %67,4 → zaman bloğu %68,4 (öznitelik + LightGBM).
`[SARMAŞIK: MiniRocket ve yağmurlu/kuru sonuçları]`

### 4.2 Uygun olmayan modeller
kNN ve Naive Bayes LOPO'da şans düzeyindedir (1 sa: %50,5 ve %50,6; LightGBM'den 22 puan düşük,
p < 0,001); ham pencerede bitkiler arası genlik/faz farklarını aşamamaktadırlar.

### 4.3 EKG temel modelleri ve zaman ölçeği
| Pencere | HuBERT donmuş, EKG | HuBERT donmuş, rastgele | Fark |
|---|---|---|---|
| 30 dk | 69,2 | 68,3 | +0,9 |
| 1 sa | 70,1 | 68,9 | +1,2 |
| **6 sa** | **77,9** | **72,8** | **+5,1** (AUC +8,1, p = 0,012) |

1 saatlik pencerede ince ayar sonuçları: HuBERT EKG %71,1 / rastgele %69,2 (p = 0,61); ECG-FM EKG
%70,8 / rastgele %70,4 (p = 0,90); donmuş ECG-FM'de EKG ön eğitimi zararlıdır (%64,8 / %70,0).
6 saatlik pencerede (DAPT deneyi):

| Kol (6 sa, LOPO) | Doğruluk | AUC |
|---|---|---|
| tsfresh + LightGBM | 81,2 | 86,7 |
| Donmuş, EKG + DAPT | 79,7 | 87,8 |
| Donmuş, EKG | 77,9 | 88,4 |
| İnce ayar, EKG + DAPT | 76,1 | 84,3 |
| İnce ayar, EKG | 74,6 | 84,2 |
| Donmuş, rastgele | 72,8 | 80,2 |
| İnce ayar, rastgele | 62,0 | 72,5 |

- EKG − rastgele (ince ayar): **+12,7 puan [+5,1; +19,9], p = 0,010, 10/12 bitki.**
- DAPT'ın ek katkısı: +1,4 (ince ayar, p = 0,65), +1,8 (donmuş, p = 0,52) — anlamsız.
- En iyi EKG kolu − LightGBM: −1,4 [−11,2; +9,4], p = 0,66 — eşdeğer.

### 4.4 Zaman karıştırıcısı
Kontrol bitkilerinde ilk/son 3 gün ayrımı 30 dk'da şans düzeyinde (%53,3, AUC 0,49), 1 sa'te
%63,7; sulama bitkilerinde %70,5–72,9 → sinyalin ana kaynağı sulama stresidir, küçük bir zaman
etkisi vardır.

## 5. Tartışma
- **Literatürle karşılaştırma:** Yazarların bölmesinde sonuçları yeniden üretiyoruz (%87,2 vs
  %84,0); ancak 12 bitkinin tamamında LOPO ortalaması 7–15 puan daha düşüktür. Yazarların 2 test
  bitkisi (0 ve 12) LOPO'da en kolay bitkiler arasındadır. Rastgele bölmeyle raporlanan %90+
  değerler görülmemiş bitkiye genellenmemektedir; bu, Wahid vd. (2026)'nın uyarısını nicel olarak
  doğrulamaktadır.
- **EKG transferi neden 6 saatte işe yarıyor?** Model "saniye" değil "örnek başına frekans"
  öğrenir. 6 sa penceresi 500 örneğe sıkıştırıldığında bitkideki dakika–saat ölçekli değişimler
  EKG'nin baskın bandına (≈0,005–0,4 döngü/örnek) düşer; 1 sa'te ise bu değişimler çok düşük
  frekansta kalır. Donmuş ECG-FM'in 1 sa'te rastgeleden kötü olması da yanlış ölçekte EKG
  özelliklerinin uyumsuz olduğunu destekler.
- **DAPT neden katkı vermedi?** 4 400 etiketsiz pencere, EKG ön eğitiminin ölçeğine göre çok
  küçüktür; ayrıca DAPT verisinin çoğu farklı tür/ortamdan (sarmaşık) gelmektedir.
- **EKG modeli neden temel modeli geçmiyor?** 6 sa'te bitki başına 23 pencere vardır; 12 bitkiyle
  güven aralıkları ±10 puandır. Az veride iyi tasarlanmış öznitelikler (tsfresh) güçlü bir alt
  sınırdır; Buss vd. (2026) de derin modellerin HGB'yi geçmediğini bildirmiştir.
- **Pratik anlam:** 6 saatlik karar penceresi, günler içinde gelişen sulama stresi için
  kabul edilebilir bir gecikmedir.

## 6. Kısıtlar
- Ana veri setinde 16 bitki (değerlendirmede 12); istatistiksel güç sınırlı.
- Sağlıklı/stresli etiketleri deneyin başı ve sonuna denk gelir (zaman karıştırıcısı kısmen
  kontrol edildi, 4 kontrol bitkisi).
- Sarmaşıkta su stresi etiketi yoktur; hava durumu vekil etiketleri kullanılmıştır.
- EKG modellerinde tek bir girdi eşlemesi (derivasyona kopyalama) ve sabit ince ayar ayarları
  kullanılmıştır; ECG-FM 6 sa ölçekte denenmemiştir.
- Uyaran veri setinin ışık sınıflarında olası artefakt.

## 7. Sonuç
Bitki elektrofizyolojisinde sulama stresi görülmemiş bitkilerde %73–81 doğrulukla tespit
edilebilmektedir; literatürdeki %90+ sonuçlar değerlendirme yöntemi nedeniyle şişiktir. İnsan
EKG'siyle önceden eğitilmiş temel modeller bitki sinyaline aktarılabilmektedir, ancak yalnızca
pencere uygun zaman ölçeğinde verildiğinde; bu durumda EKG ön eğitimi aynı mimarinin rastgele
başlatılmışına göre anlamlı katkı sağlamakta ve güçlü öznitelik tabanlı modelle eşdeğer
performansa ulaşmaktadır. Gelecek çalışmalar daha fazla bitki, farklı türler, EEG ve genel zaman
serisi temel modelleri ve çoklu zaman ölçekli girdi eşlemelerini incelemelidir.

## Kaynakça
(APA; tam liste `reports/faz1/faz1_rapor.md` ile aynı — Bhadra vd. 2023; Buss vd. 2023, 2025,
2026; Chatterjee vd. 2015, 2018; Coppola vd. 2024; González i Juclà vd. 2023; McKeen vd. 2025;
Najdenovska vd. 2021; Wahid vd. 2026.)

## Ek: Tekrarlanabilirlik
Kod ve sonuçlar: https://github.com/melihakcam/bitkiEkg (dal `onDeneme`). Yerel: Python 3.10,
`requirements.txt`; GPU deneyleri Colab T4, `notebooks/02–04_*.ipynb`. Model sürümleri: HuBERT-ECG
`eca1c5a…`, ECG-FM `584219e…`, fairseq-signals `f8f0ff1…`.
