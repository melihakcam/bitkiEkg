# Bitki Sinyalini EKG Temel Modellerine Uyarlama Planı

Bu belge projenin en kritik tasarım kararını tanımlar: 10 Hz örneklenen, dakikalar–saatler
ölçeğinde değişen bitki elektrik potansiyeli, saniyeler ölçeğinde kalp atımı görmeye alışmış
EKG modellerine nasıl verilecek?

> ⚠️ **Doğrulanacak:** Aşağıdaki model girdi özellikleri yazarın bilgisine dayanır, internet
> erişimi olmadan yazılmıştır. Ağırlıklar indirildiğinde model kartı ve kodundan teyit edilecek.

## 1. Uyuşmazlık

| Özellik | Bitki sinyali (domates, Buss 2026) | ECG-FM | HuBERT-ECG |
|---|---|---|---|
| Örnekleme hızı | 10 Hz | 500 Hz *(doğrulanacak)* | 100 Hz *(doğrulanacak)* |
| Girdi uzunluğu | Pencereye bağlı (1 dk = 600, 1 sa = 36 000 örnek) | 5 s → 2 500 örnek *(doğrulanacak)* | 5 s → 500 örnek/kanal *(doğrulanacak)* |
| Kanal sayısı | Veri gelince belirlenecek | 12 derivasyon | 12 derivasyon |
| İlgili olaylar | Yavaş kaymalar, dakikalar–saatler süren tepkiler | QRS ~0,1 s, kalp döngüsü ~1 s | Aynı |
| Gürültü | Elektrot kayması, sulama/ışık döngüsü | Taban çizgisi kayması, kas gürültüsü | Aynı |

## 2. Temel fikir: model "Hz" bilmez, örnek bilir

Model ağırlıkları saniye değil **örnek başına frekans** (döngü/örnek) öğrenir. Bu yüzden
amaç, bitki sinyalindeki bilgili bandı modelin eğitimde gördüğü normalize frekans bandına
taşımaktır.

- EKG'de bilgi taşıyan bant kabaca 0,5–40 Hz'dir. 500 Hz'de bu bant
  **~0,001–0,08 döngü/örnek** aralığına düşer.
- Bitki sinyalinde ilgili değişimlerin periyodu dakikalar–saatlerdir. Pencere `T` saniye,
  modelin girdi uzunluğu `L` örnek ise yeni örnekleme hızı `fs' = L / T` olur.
  Periyodu `P` saniye olan bir bileşen `1 / (P · fs')` döngü/örnek'e düşer.
- Örnek: ECG-FM (`L = 2500`), 1 saatlik pencere → `fs' ≈ 0,69 Hz`. 10 dk periyotlu bir
  dalgalanma → `1 / (600 · 0,69) ≈ 0,0024` döngü/örnek, yani EKG bandının içinde.

**Sonuç:** Bitki penceresi, modelin sabit girdi uzunluğuna **yeniden örneklenir** (zaman
sıkıştırma). Pencere süresi `T`, hangi bitki olaylarının "kalp atımı ölçeğine" düşeceğini
belirleyen hiperparametredir.

## 3. Ön işleme hattı

1. **Kalite süzgeci:** Kopuk elektrot, doygunluk ve uzun eksik aralıklar işaretlenip atılır.
2. **Kayma giderme:** Pencere başına doğrusal eğilim çıkarma (detrend). Gerekirse çok düşük
   kesimli yüksek geçiren süzgeç (kesim frekansı pencere süresine göre seçilir).
3. **Pencereleme:** Süre `T ∈ {10 dk, 1 sa, 6 sa}` (Buss 2026 ile karşılaştırılabilir aralık),
   örtüşme yalnızca **aynı bitki içinde** (bölme bitki bazlı olduğu için sızıntı yaratmaz).
4. **Yeniden örnekleme:** Kenar yumuşatmalı (anti-aliasing) `scipy.signal.resample_poly`
   ile pencere → `L` örnek.
5. **Ölçekleme:** Pencere başına z-skor. EKG modelleri mV ölçeğinde eğitildiği için
   alternatif olarak genliği EKG'nin tipik aralığına (±1–2 mV) eşleyen ölçekleme de denenir.
6. **Kanal eşleme** (bkz. §4).

Aynı pencereler **tüm modellere** (kNN'den ECG-FM'e) verilir; böylece fark yalnızca modelden
kaynaklanır.

## 4. Kanal eşleme seçenekleri

Bitki verisinde birkaç kanal, EKG modellerinde 12 derivasyon var.

| Seçenek | Açıklama | Artı | Eksi |
|---|---|---|---|
| A. Tekrar | Bitki kanalı 12 derivasyona kopyalanır | Basit, ek parametre yok | Derivasyonlar arası ilişki yapay |
| B. Sıfır doldurma | Bitki kanalı Derivasyon II'ye, diğerleri 0 | EKG'de tek derivasyon kaydına benzer | Model eksik derivasyonla eğitilmemişse dağılım dışı |
| C. Öğrenilen adaptör | 1×1 konvolüsyon: `C → 12` | Esnek, az parametre | Az veride aşırı öğrenme riski |

Varsayılan: **A**. B ve C ablasyon olarak denenir.

## 5. Transfer stratejileri

| Strateji | Ne eğitilir | Amaç |
|---|---|---|
| Doğrusal sonda (linear probe) | Yalnızca sınıflandırma başlığı | Önceden öğrenilmiş temsil bitkide işe yarıyor mu? |
| Kısmi ince ayar | Üst katmanlar + başlık | Az veriyle uyum |
| Tam ince ayar | Tüm ağ, düşük öğrenme oranı | Üst sınır |
| **Rastgele başlatma (kontrol)** | Aynı mimari, ön eğitim yok | Kazancın ön eğitimden mi mimariden mi geldiğini ayırır |

Rastgele başlatılmış kontrol, makalenin ana iddiası için zorunludur: "EKG ön eğitimi bitki
sinyaline aktarılır" demek için önceden eğitilmiş model, **aynı mimarinin sıfırdan eğitilmiş
halinden** anlamlı biçimde iyi olmalıdır.

## 6. Değerlendirme

- **Bitki bazlı bölme:** `GroupKFold` (grup = bitki kimliği). 16 domates bitkisiyle
  4 katman → her katmanda 4 görülmemiş bitki. Alternatif: bitki-dışarıda-bırak (LOPO).
- **Rastgele bölme ile karşılaştırma:** Aynı modeller rastgele pencere bölmesiyle de
  eğitilip sonuçların ne kadar şiştiği raporlanır (Wahid vd. 2026'nın eleştirisi).
- **Veri setleri arası:** Domatesle eğit → sarmaşıkta test ve tersi (ortak etiket tanımı
  veri gelince belirlenecek).
- **Metrikler:** Doğruluk, F1, ROC-AUC; katmanlar arası ortalama ± standart sapma,
  bitki bazında önyükleme (bootstrap) güven aralığı.
- **Erken tespit (isteğe bağlı):** Sulama kesildikten sonra modelin stresi kaç saat içinde
  yakaladığı.

## 7. Ablasyon matrisi (öncelik sırasıyla)

1. Ön eğitim var / yok (rastgele başlatma)
2. Pencere süresi `T`: 10 dk, 1 sa, 6 sa
3. Transfer stratejisi: doğrusal sonda, tam ince ayar
4. Kanal eşleme: A, B, C
5. Ölçekleme: z-skor, mV eşleme

Hesap bütçesi sınırlı olduğu için 1–3 zorunlu, 4–5 zaman kalırsa yapılır.

## 8. Veri gelince netleşecek sorular

- [ ] Domates verisinde kanal sayısı ve dosya formatı
- [ ] Su stresi etiketi nasıl tanımlı (sulama takvimi mi, ayrı etiket dosyası mı)?
- [ ] Bitki kimliği her kayıtta var mı?
- [ ] Eksik veri / kopuk elektrot oranı
- [ ] Sarmaşık verisinde su stresiyle eşlenebilecek bir etiket (ör. yağış, toprak nemi) var mı?
- [ ] ECG-FM ve HuBERT-ECG girdi özelliklerinin doğrulanması (§1)
