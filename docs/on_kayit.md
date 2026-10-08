# Ön kayıt: 18 günlük ayrışma testi (zamandan arındırılmış su stresi testi)

Yazıldığı tarih: 2026-10-08, **EKG gömmeleri çıkarılmadan ve hiçbir sonuç görülmeden önce.**
Bu belgedeki yöntem sonuç görüldükten sonra değiştirilmez. Sonradan eklenen her analiz "keşif" diye ayrı yazılır.

## Soru
EKG ile ön eğitilmiş HuBERT-ECG'nin donmuş gömmeleri, susuz bitkileri **aynı günlerdeki** kontrol bitkilerinden
ayırabiliyor mu, ve bu ayrışma (a) su farkı başlamadan önce yok, (b) su azaldıkça artıyor mu?
Aynı günler karşılaştırıldığı için "deneyin başı/sonu" (zaman) ayrışmayı açıklayamaz.

## Veri
`data/colab/domates_18gun_6h_500.npz` (`scripts/colab_veri_18gun.py`): 16 bitki × 71 pencere (6 sa),
04–21.06.2025, eksik oranı ≤ %1. Gruplar cihaz düzeyinde (docs/veri_notlari.md):
aşırı PN2, PN5 · kontrol 400 mL PN8, PN9 · 200 mL PN10, PN11 · 100 mL PN12, PN16 (cihaz başına 2 bitki).

- **Önce dönemi P0:** 04–08.06 (tüm bitkiler 400 mL; docs/gelistirme_plani.md B1).
- **Sonra dönemi P1:** 14–21.06 (kuraklık gruplarında toprak nemi 9–11.06'dan itibaren düşüyor).
- 09–13.06 geçiş: yalnızca günlük eğri grafiğinde gösterilir.

## Model
- Gömmeler: `hubert_gomme` (tüm katmanların zaman ortalaması), girdi `np.tile(L, (1, 12))`;
  kollar: `ekg` (ön eğitimli) ve `rastgele_t1…t5` (aynı mimari, rastgele ağırlık, tohum 1–5).
- Karşılaştırma: tsfresh + LightGBM (`scripts/stres_dengeli.py` içindeki `model`).
- Sonda: `stres_dengeli.ekg_sonda` (StandardScaler + LogisticRegression C = 1, class_weight = "balanced") —
  §11 doz-etki testiyle aynı (eğitimde 4 susuz cihaza karşı 2 kontrol cihazı olduğu için sınıf ağırlıklı).

## Yöntem
1. Eğitim etiketleri yalnızca **P1** pencerelerinden: 200 mL ve 100 mL cihazları = 1, kontrol cihazları = 0.
   Aşırı sulama cihazları eğitime girmez.
2. **Cihaz-dışarıda-bırak** (6 cihaz): her cihaz için model diğer 5 cihazın P1 pencereleriyle eğitilir,
   dışarıdaki cihazın **tüm günlerdeki** pencereleri skorlanır.
3. Bitki başına **Δ = ortalama skor(P1) − ortalama skor(P0)**. Cihazın sabit farkı Δ'da birbirini götürür.
4. **Ana istatistik:** 12 bitkide (kontrol, 200, 100) doz sırası (0, 1, 2) ile Δ arasındaki Spearman r.
5. **p değeri:** 6 cihazın gruplara 90 farklı atanışı (6!/(2!)³); her atanışta etiketler değiştiği için
   model **yeniden eğitilir**. p = (1 + #{r_perm ≥ r_gerçek}) / 90 (tek yönlü; en küçük olası p ≈ 0,011).

## Başarı ölçütü (üçü birden)
- EKG kolunda r > 0 ve p < 0,05;
- **Plasebo** (aynı yöntem, P1 yerine 07–08.06, P0 yerine 04–05.06; hepsi tedavi öncesi) anlamlı değil (p ≥ 0,05);
- EKG kolunun r'si 5 rastgele kolun hepsinden büyük.

## Ek raporlar (karar ölçütü değil)
- 5 rastgele kol ve LightGBM için aynı r ve p.
- Günlük ayrışma eğrisi: her gün için (susuz grup ort. skor − kontrol ort. skor), kollara göre.
- Aşırı sulama cihazları: 6 cihazın hepsiyle eğitilen modelle skorlanır, yalnızca betimsel.
- Tutarlılık kontrolü: 04–06 ve 19–21.06 pencerelerinin yeni EKG gömmeleri, Aşama 1 gömmeleriyle
  (`gomme_ekg_6h_asama1.npz`) aynı olmalı (aynı girdi, aynı model).

## Bilinen sınırlar (önceden yazıldı)
- Grup başına 2 cihaz: cihaz düzeyinde 90 atanış, güç düşük.
- 100 mL cihazları tedavi öncesinde de ham sinyalde daha çok kayıyor (docs/sonuclar.md §12);
  Δ sabit farkı siler ama cihaza özgü kaymayı silmez. Plasebo testi bunu kısmen yakalar.
