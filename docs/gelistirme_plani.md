# Geliştirme Planı — Stres Sinyalini Güçlendirmek (donanımsız)

Tarih: 2026-10-07 · Durum: §10 (docs/sonuclar.md) ve dengelenmiş eğitim testleri sonrası.

## Sorun (bir cümle)
Modeller "susuzluk" yerine "deneyin başı/sonu" farkını öğreniyor. Zamanı engelleyince susuzluk izi
görünüyor (3 testte de doğru yönde) ama 4 kontrol bitkisiyle kanıtlanamıyor (p = 0,05–0,18).

## Başarı ölçütü (bundan sonra hep bu)
Eski doğruluk (%81) zamanla şişik olduğu için artık ana ölçüt değil. Yeni ana ölçüt:
**Zamandan arındırılmış başarı** — kontrol bitkilerinin son günleri de "sağlıklı" etiketlenir,
cihaz-dışarıda-bırak değerlendirilir; susuz bitkilerin skor artışı (Δ) ile kontrol bitkilerinin
skor artışı karşılaştırılır.
- Şu an: Δ farkı 0,16–0,19, p = 0,05–0,18 (`results/tables/stres_dengeli_*`).
- Hedef: p < 0,05 (bitki düzeyi) ve tüm testlerde tutarlı yön.
Eski doğruluk da raporlanır ama tek başına iddia kurulmaz.

## Kurallar (sonuçları zorlamamak için)
1. Denenecek yöntemler aşağıda önceden listelenmiştir; sonradan eklenen her şey "keşif" diye ayrı yazılır.
2. Her yöntemin sonucu (kötü olsa da) raporlanır; çoklu test için Holm düzeltmesi.
3. Ayarlar 1 sa pencerede seçilir, 6 sa ve diğer pencerelerde doğrulanır.

## A. Daha fazla veri (en etkili)
| # | İş | Neden | Süre |
|---|---|---|---|
| A1 | Açık veri taraması: Zenodo, Figshare, Mendeley Data, Dryad, GitHub, PLEASED | Kontrol bitkisi olan ikinci bir susuzluk verisi bulgunun bağımsız tekrarı olur | 1–2 sa |
| A2 | Domates verisinde 18 günün hepsini kullanmak (şimdiye kadar yalnızca ilk/son 3 gün) | 3 kat veri; susuzluk günler geçtikçe artmalı, zaman etkisi kontrolde de aynı → "ayrışma eğrisi" | 0,5 gün |
| A3 | Daha kısa pencereler (30 dk, 5 dk) | Daha çok örnek, daha dar güven aralığı | 0,5 gün |

## B. Sinyali güçlendiren yöntemler (her biri yukarıdaki ölçütle değerlendirilir)
| # | Yöntem | Dayanak | Yer |
|---|---|---|---|
| B1 | Her bitkiyi kendi tedavi öncesi 5 gününe (04–08.06, herkes 400 mL) göre normalize etmek | Bitkiler/cihazlar arası sabit farkı siler | Yerel |
| B2 | Günlük ritim öznitelikleri: 24 sa pencere, gündüz/gece genlik oranı, tepe saati, ritim gücü | Tran vd. 2019: su stresi günlük ritmi bozuyor; zaman kayması ritmi bozmaz | Yerel |
| B3 | Kontrol grubuna göre fark: her gün için susuz bitkinin skoru − kontrol bitkilerinin aynı gün ortalaması | Zaman etkisini doğrudan çıkarır | Yerel |
| B4 | Hibrit model: EKG gömmeleri + tsfresh öznitelikleri | EKG AUC'si yüksek, LightGBM doğruluğu yüksek → birbirini tamamlayabilir | Yerel (6 sa gömmeler var) |
| B5 | Ardışık pencereleri birleştirip günlük karar | González vd. 2023: +10 puan; gürültüyü azaltır | Yerel |
| B6 | EKG gömmelerini 18 günün tamamı için çıkarmak, B1–B5'i EKG ile tekrarlamak | Asıl iddia EKG modeli için | Colab, GPU'suz ~10 dk, tek sefer |

## C. Sonuçta ne olacak?
- **Sinyal p < 0,05 ile doğrulanırsa:** "EKG modeli zamandan arındırılmış değerlendirmede de susuzluğu
  yakalıyor" → ana iddia geri gelir; zaman tuzağı bulgusu ve EKG ön eğitimi bulgusuyla birlikte
  güçlü bir Q1 makalesi.
- **İkinci veri seti bulunursa:** bulgular iki bağımsız veride sınanır → Q1 için en güçlü durum.
- **Hiçbiri tutmazsa:** veri sınırı kesinleşir; makale yine zaman tuzağı + EKG aktarımı bulgularıyla
  yazılır, susuzluk izi "ipucu" olarak kalır.

## Sıra
A1 → A2 + B1 + B3 (aynı analizde) → B2 → B4 → B5 → B6 → sonuçlara göre makale.
