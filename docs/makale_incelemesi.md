# Benzer Makalelerin İncelemesi — Q1 İçin Çıkarımlar

Tarih: 2026-10-07 · İncelenen: `docs/makaleler/benzer/` altındaki 13 PDF (ana metin; ekler
taranmadı). İndirilemeyen 17 makale `docs/amac.md` §5.5'te; onlar gelince bu belgeye eklenecek.

## 1. Makale makale ne öğrendik

### 1.1 Bitki elektrofizyolojisi

**González i Juclà vd. 2023 (Sci Rep 13:9633) — en önemli bulgu, düzeltme gerektiriyor**
- 16 domates, azot eksikliği, ham sinyal + derin öğrenme.
- **Bitki-dışarıda-bırak (LOOCV, 16 bitki) yapmışlar:** test %77,0 ± 12,1. Üç bitki %60'ın
  altında (B3, B4, C6). Bu bizim tablomuzla neredeyse aynı (LOPO %72,9 ± 11,3; zor bitkiler 1 ve 13).
- **Ardışık tahminleri birleştirme:** son 1 000 tahminin ortalaması → %87,6 ± 16,0 (bazı bitkilerde
  %99,9; zor bitkilerde değişmiyor, C6'da %50).
- Özetteki "%88, birleştirmeyle %96" iki bitkilik testten; LOOCV ortalaması %77.
- **Sonuç:** "Bitki bazlı değerlendirmeyi ilk biz yaptık" diyemeyiz. Bizim farkımız: rastgele
  bölmeyle aynı veride yan yana ölçüp şişmeyi sayıyla göstermek, iki türde tekrar etmek ve zaman
  karıştırıcısı testi. `faz1_rapor.md` ve `makale_taslak.md` tablolarında González için bölme "—" ve
  sonuç "%99'a kadar" yazıyor → **düzeltilmeli** (bölme: LOOCV; sonuç: %77,0, birleştirmeyle %87,6).

**Aust vd. 2024 (arXiv 2412.13312, sarmaşık + ozon, AutoML)**
- Rastgele %80/20 bölmeyle %94,6 raporluyorlar; kendi ayrı test kümelerinde doğruluk
  **%71–74'e düşüyor**. Aynı grup (Buss/Hamann) → şişme bulgumuzu destekleyen bağımsız bir örnek.
- Kendileri "küçük veride meta aşırı uyum" riskini yazıyor.

**Tran vd. 2019 (Sci Rep 9:17073, domates, kuraklık + gece/gündüz)**
- Giriş bölümü doğrudan **insan kalbi/beyni ölçümüne benzetmeyle** başlıyor: "insan tıbbındaki
  teknolojinin aktarılması bitkinin durumunu anlamamızı sağlayabilir". Fikrimizin kökü için iyi alıntı.
- Kuraklık %98,5, gece/gündüz %94,6 (GBT) — **rastgele %80/20 bölme.**
- Bitkiler arası taban seviye farkını gidermek için sinyali **24 saatlik döngülere bölüp
  normalize ediyorlar**. Günlük ritim sinyalin en belirgin bileşeni.

**Bhadra vd. 2023 (PLOS ONE)** — 15 istatistiksel öznitelik, rastgele alt örnekleme, AdaBoost
dengeli doğruluk 0,71. Rastgele bölme. Yeni bilgi yok.

### 1.2 Başka alandan aktarım (yöntem olarak bize en yakın)

**Tan vd. 2024 (NeurIPS) — "Dil modelleri zaman serisinde gerçekten işe yarıyor mu?"**
- Üç popüler yöntemde dil modelini çıkarınca ya da **rastgele ağırlıkla değiştirince** sonuç
  değişmiyor, çoğu zaman iyileşiyor. Önceki makalelerin iddiası bu kontrolle çöküyor.
- Kullandıkları testler: (1) modeli çıkar, (2) rastgele başlat, (3) **girdiyi karıştır (shuffle)**
  — model zamansal sırayı kullanıyor mu?, (4) hesaplama maliyeti, (5) az veri (%10) durumu,
  (6) bootstrap %95 GA.
- **Bize dersi:** Bizde (2) var ve 6 sa'te kazanç anlamlı. Hakem diğerlerini de sorabilir.
  Özellikle (3) karıştırma testi ve (4) maliyet tablosu ucuz ve iddiayı sağlamlaştırır.

**Lu vd. 2022 (AAAI) — Frozen Pretrained Transformer**
- Dil modeli donmuş halde başka alanlarda sıfırdan eğitimle eşdeğer. **Rastgele başlatılmış
  transformer da şaşırtıcı derecede iyi** (MNIST %91,7'ye karşı %98,0).
- Ön eğitimin asıl kazancı **daha hızlı yakınsama** (4–40 kat).
- **Kaynak alan karşılaştırması yapmışlar:** dil vs görüntü (ViT) vs yapay görev. Kaynağın
  hedefe "benzerliği" önemli (protein ↔ dil).
- **Bize dersi:** Bizim bulgular bununla birebir uyumlu (rastgele HuBERT %69, ön eğitimli daha
  hızlı yakınsıyor). "EKG'ye özgü mü, yoksa herhangi bir ön eğitim mi?" sorusu için kaynak alan
  karşılaştırması (EKG vs konuşma) gerekli.

**Yang vd. 2021 (ICML) — Voice2Series**
- Konuşma modelini zaman serisi sınıflandırmaya "yeniden programlıyor"; 30 UCR görevinin 19'unda
  rekabetçi.
- **Kuram:** hedef hatası ≤ kaynak hatası + kaynak ve hedef temsilleri arasındaki
  **Wasserstein uzaklığı**. Yani transfer, bitki temsilleri EKG temsillerine ne kadar yakınsa o
  kadar iyi çalışır.
- **Kaynak verinin süresi önemli:** kısa (≤1 s) komut sesleriyle eğitilen model, kısa zaman
  serilerinde daha iyi. Bizim "doğru zaman ölçeği" bulgumuzun bir benzeri.
- **Bize dersi:** 6 sa bulgusunu açıklamak için ölçülebilir bir yol: her pencere süresinde bitki
  gömmelerinin gerçek EKG gömmelerine uzaklığını hesaplamak. 6 sa'te uzaklık en küçükse
  mekanizma gösterilmiş olur.

**Zhou vd. 2023 (NeurIPS) — GPT4TS / One Fits All** — Donmuş dil/görüntü modeli zaman serisinin
tüm görevlerinde (sınıflandırma dahil) rekabetçi; öz-dikkatin PCA'ya benzer davrandığını öne
sürüyor. Tan vd. 2024 bu yöntemi de eleştiriyor.

**Jin vd. 2024 (ICLR) — Time-LLM** — Dil modelini tahmin için yeniden programlama. Tan vd. 2024'te
en çok çöken yöntem (26/26 durumda basit sürümü daha iyi). Bizim için uyarı örneği.

**Li vd. 2026 (ICLR) — BioX-Bridge** — İki biyosinyal temel modelinin ara katmanlarını hafif bir
"köprü" ağıyla bağlayarak etiketsiz aktarım (ör. PPG ↔ EKG). Katman seçimi için temsillerin
benzerliğini ölçüyorlar. Bize doğrudan yöntem değil, ama "temsil benzerliği ölçümü" fikrini
destekliyor.

### 1.3 EKG temel modelleri

**McKeen vd. — ECG-FM (arXiv 2408.05178)** — 1,5 M EKG, 500 Hz, z-skor, **5 s parça** (bizim
uyarlamamızla tutarlı). Ön eğitimde komşu EKG parçaları "aynı" sayılıyor (CMSC) → model zamansal
kararlılığa göre eğitilmiş.

**Al-Masud & Strodthoff 2026 (arXiv 2605.12241) — EKG temel modellerinde ön eğitim ve ölçek**
- En iyi mimari **S4 (durum uzayı modeli)**, transformer ve CNN'den iyi. En iyi ön eğitim
  yöntemi **CPC**. Kod ve ağırlıklar açık.
- **HuBERT++ tüm görevlerde HuBERT-ECG'den iyi.**
- **Alana uyarlamalı ön eğitimi (DAPT) güçlü biçimde öneriyorlar.** Bizde DAPT ek katkı vermedi
  (+1,5, anlamsız) → tartışmada açıklanması gereken bir çelişki (olası neden: bizde etiketsiz
  veri az, 4 400 pencere; onlarda hedef alan da EKG).
- Temsil benzerliği için **CKA analizi** kullanıyorlar.

**Tang vd. 2026 (arXiv 2605.16975) — 10 s EKG modellerini uzun kayıtlara genişletme**
- 10 s modelleri uzun kayda uygulamanın iki yolu: kayan pencere + birleştirme (temel yöntem) ya da
  kendi önerdikleri hafif ek modül. Konumsal gömmeyi **aradeğerlemiyorlar**, çünkü biyosinyalde
  zaman ölçeği fizyolojik anlam taşıyor.
- **Bize dersi:** Biz tam tersini yapıyoruz: uzun bitki kaydını 5 s'ye **sıkıştırıyoruz**.
  Makalede bu iki yaklaşım karşılaştırılmalı; "neden sıkıştırma?" sorusuna cevap 6 sa bulgusu.

## 2. Q1 açısından ne değişti

### 2.1 Düzeltilmesi gereken iddialar
1. **"Kapsamlı bitki bazlı değerlendirme" yenilik değil** — González vd. 2023 16 bitkiyle LOOCV
   yapmış. Yeniliğimiz: aynı veride rastgele bölme / yazar bölmesi / LOPO / zaman bloğunu yan
   yana koyup şişmeyi ölçmek, iki türde tekrar, zaman karıştırıcısı testi.
2. **"Birkaç pencereyi birleştirme" yeni fikir değil** — González vd. yapmış (+10 puan). Yaparsak
   onlara atıf verip kendi verimizde tekrar olarak sunulmalı; tek başına katkı değil.
3. Literatür tablolarındaki González satırı yanlış (bkz. §1.1).

### 2.2 Güçlenen iddialar
1. **İnsan EKG modeli → bitki** aktarımı hâlâ yok; Tran vd. 2019 benzetmeyi yapmış, deneyi
   yapmamış.
2. **Rastgele ağırlık kontrolü** alanın en sert eleştirisine (Tan vd. 2024) baştan cevap veriyor;
   bizde kontrol yapılmış ve 6 sa'te kazanç anlamlı. Bu, makalenin merkezine konmalı.
3. **Şişme bulgusu** Aust vd. 2024'ün kendi verisinde de görülüyor (%94,6 → %71–74).
4. **Zaman ölçeği bulgusu** literatürle uyumlu (Voice2Series: kaynak süre benzerliği;
   Tang vd. 2026: EKG modelleri zaman ölçeğine duyarlı).

### 2.3 Hakemin büyük ihtimalle soracakları (makalelerden çıkan)
| Soru | Kaynak | Bizde durum |
|---|---|---|
| Kazanç ön eğitimden mi, mimariden mi? | Tan 2024, Lu 2022 | ✅ Rastgele ağırlık kontrolü var |
| EKG'ye özgü mü, herhangi bir ön eğitim mi? | Lu 2022 (kaynak karşılaştırması) | ⬜ Konuşma modeli (HuBERT/wav2vec2) ile karşılaştırma yok |
| Model zamansal sırayı kullanıyor mu? | Tan 2024 (karıştırma testi) | ⬜ Yok |
| Neden 6 sa? Mekanizma ne? | Voice2Series (temsil uzaklığı), Strodthoff (CKA) | ⬜ Hipotez var, ölçüm yok |
| Hesaplama maliyeti buna değer mi? | Tan 2024 | 🟡 Süreler kayıtlı, tablo yok |
| Daha güçlü EKG modeli (S4/CPC, HuBERT++) ne yapar? | Al-Masud 2026 | ⬜ Yok |
| DAPT neden işe yaramadı? | Al-Masud 2026 | 🟡 Sonuç var, açıklama yok |
| Tohum tekrarı, güven aralığı | Tan 2024 | 🟡 GA var, tek tohum |

## 3. Önerilen iş listesi (onay bekliyor; hiçbiri başlatılmadı)

Hepsi mevcut çerçevenin içinde: EKG → bitki aktarımını daha sağlam göstermek.

**Yerelde, Colab gerekmez:**
1. Literatür tablolarını düzeltmek (González, Aust) ve yenilik iddialarını §2.1'e göre yeniden yazmak.
2. Maliyet tablosu (mevcut sürelerden).

**Colab gerekir (tek oturumda toplanabilir):**
3. **Karıştırma testi:** 6 sa pencereyi zamanda karıştırıp EKG modeline vermek; başarı düşmüyorsa
   model sırayı kullanmıyor demektir.
4. **Temsil uzaklığı / CKA:** her pencere süresinde (30 dk, 1 sa, 6 sa) bitki gömmelerinin gerçek
   EKG gömmelerine (ör. PTB-XL'den küçük bir örnek) uzaklığı. 6 sa bulgusunun mekanizması.
5. **Kaynak alan karşılaştırması:** aynı düzende konuşma modeli (HuBERT/wav2vec2). EKG daha iyiyse
   "EKG'ye özgü" iddiası kurulur.
6. Tohum tekrarı (3–5 tohum).
7. (İsteğe bağlı) Daha güçlü EKG modeli: HuBERT++ ya da S4/CPC.

Önceki önerilerden "hibrit model" ve "bitkinin kendi başlangıcına göre ayarlama" hâlâ geçerli;
"birkaç pencereyi birleştirme" ise ancak González'e atıfla, ikincil analiz olarak.
