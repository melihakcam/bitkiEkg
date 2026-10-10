# Ön kayıt 2: EKG temel modelleri bitki sinyalinde — çok veri setli karşılaştırma

Yazıldığı tarih: 2026-10-10, **Madariaga ve HIPB-MM sinyal değerlerine hiç bakılmadan, hiçbir gömme çıkarılmadan önce.**
Bu iki veri için yalnızca açıklama, makale yöntemi, dosya adları ve sütun başlıkları incelendi (docs/veri_taramasi.md,
"Karşılaştırma çalışması için tasarım kontrolü"). Yöntem sonuç görüldükten sonra değiştirilmez; her sapma ve sonradan
eklenen her analiz "sapma" ya da "keşif" diye ayrı yazılır.

## Soru
İnsan EKG'siyle ön eğitim, aynı mimarinin rastgele başlatılmış haline göre bitki elektrik sinyalinden daha iyi bilgi
çıkarıyor mu? Bunu daha önce görülmemiş iki veri setinde, iki EKG modeliyle, tek ve sabit bir yöntemle test ederiz.

## Veri setleri ve görevler

### A. Madariaga vd. 2024 — uyaran öncesi / sonrası (figshare 24161100, "Plant E.Phys. Source Data.zip")
- Plant SpikerBox, 10 kHz .wav; uyaran başı ("1") ve sonu ("2") işaretleri .txt dosyalarında.
- **Dahil etme (mekanik):** hem "1" hem "2" işareti olan, "1"den önce ≥ 60 s ve "2"den sonra ≥ 30 s kaydı olan kayıtlar.
  Kullanılabilir kaydı 3'ten az olan türler çıkarılır. **Kalan tür sayısı 8'den azsa test "güç yetersiz" diye raporlanır,
  karar verilmez.**
- Ön işleme: 10 kHz → 100 Hz (`resample_poly`). Kayıt başına ölçek, uyaran öncesi pencerenin medyanı ve IQR'ı ile:
  x' = (x − medyan(Ö)) / IQR(Ö); aynı ölçek o kaydın tüm pencerelerine uygulanır (seviye değişimi korunur).
- Pencereler (30 s = 3 000 örnek): **Ö** = [t₁ − 30 s, t₁), **S** = [t₂, t₂ + 30 s), **Ö₀** = [t₁ − 60 s, t₁ − 30 s).
- Ana görev: Ö (0) / S (1). **Plasebo görevi:** Ö₀ (0) / Ö (1) — ikisi de uyaran öncesi.
- Değerlendirme: **tür-dışarıda-bırak** (her katmanda bir tür test). Birim = tür. Tür başına AUC; ana ölçü türlerin ortalama AUC'si.
- Bilinen uyaran–tür çakışması (dokunma yalnızca sinekkapan ve küstüm otu) nedeniyle alev/dokunma ayrımı yapılmaz.
  Yazarların eşikle türettiği "tepki var/yok" etiketi kullanılmaz.

### B. HIPB-MM (Hugging Face JM1122/HIPB-MM, `HIPB_MM_signal_16s_SEED42_STAND`)
- *Arabidopsis*, Keithley 2401, 16 s parçalar. Yalnızca **M (*H. armigera*, 17 kayıt) / T (*S. exigua*, 21 kayıt)**;
  ikisi de 13–17.05 arasında aynı günlerde kaydedildi. N (tek gün) ve X (4 kayıt) tarihle çakışık olduğundan kullanılmaz.
- `train` ve `test` klasörleri birleştirilir (yazar bölmesi kayıt sızıntısı içeriyor). Kayıt anahtarı = dosya adındaki
  `<tarih>_<kayıt>` (ör. `0513_M2`). Parçalar verildiği gibi alınır, sonra parça başına robust z (medyan / IQR).
- Değerlendirme: **kayıt-dışarıda-bırak** (38 katman). Kayıt skoru = parça olasılıklarının ortalaması. Birim = kayıt;
  ana ölçü 38 kayıt üzerinden AUC.

## Kollar (her iki veri, aynı)
- **HuBERT-ECG:** `hubert_girdisi(Z, "tekrar")` → `hubert_gomme` (tüm katmanların zaman ortalaması, 512 boyut);
  ön eğitimli + rastgele başlatılmış tohum 1–5 (`scripts/ayrisma_testi.py` / Colab not defterleriyle aynı başlatma).
- **ECG-FM:** `ecgfm_girdisi(Z)` → `ecgfm_gomme` (son katman, 768 boyut); ön eğitimli + `ecgfm_yukle(False, tohum)` tohum 1–5.
- Sonda (tüm gömmeler): StandardScaler + LogisticRegression(C = 1, class_weight = "balanced", max_iter = 5000), tohum 42.
- **İkincil karşılaştırma kolları** (karar ölçütü değil): MiniRocket (aeon varsayılan), tsfresh `MinimalFCParameters`
  + LightGBM (300 ağaç, öğrenme oranı 0,05), Chronos-Bolt-small kodlayıcı çıktısının zaman ortalaması + aynı sonda.
  Bir kol kurulamaz ya da çalıştırılamazsa yerine başka model konmaz; "yapılamadı" diye raporlanır.

## Ana hipotez ve başarı ölçütü
Her (veri, EKG modeli) çifti için — 4 ana test: {A, B} × {HuBERT-ECG, ECG-FM}:
1. Ön eğitimli kolun ana ölçüsü, **5 rastgele kolun her birinden** büyük;
2. ΔAUC = AUC(ön eğitimli) − ortalama AUC(5 rastgele) > 0 ve tek yönlü p < 0,05, **4 ana test üzerinde Holm düzeltmesiyle**:
   - A: türler üzerinde eşleştirilmiş Wilcoxon (tür başına ΔAUC);
   - B: kayıtlar üzerinde 2 000 bootstrap, p = oran(ΔAUC* ≤ 0);
3. Yalnızca A için: ön eğitimli kolun **plasebo** görevindeki ortalama AUC'si 0,5'ten anlamlı büyük değil
   (türler üzerinde Wilcoxon, tek yönlü, p ≥ 0,05).

Üçü (B'de 1 ve 2) sağlanırsa o çift için "EKG ön eğitimi avantajı var" denir; sağlanmazsa "yok" denir. İki sonuç da aynen raporlanır.

## Ek raporlar (karar ölçütü değil)
- A ve B için tüm kolların ana ölçüsü; B için ön eğitimli kolun kayıt etiketleri tarih içinde karıştırılarak
  (her tarihte M/T sayısı korunarak) 999 permütasyonla, her birinde yeniden eğitilerek elde edilen p değeri.
- İkincil kollarla aynı ΔAUC karşılaştırması (ön eğitimli EKG − Chronos, − MiniRocket, − LightGBM).
- Domates su stresi: ön kayıt 1'in sonuçları (docs/sonuclar.md §15, §15a) değiştirilmeden aktarılır.
- **Keşif:** Matić külleme verisi (daha önce incelendi, ön kayıtlı sayılmaz) — aynı kollar, deney içinde bitki-dışarıda-bırak.

## Bilinen sınırlar (önceden yazıldı)
- A: uyaran elle işaretlendi; S penceresi uyarandan hemen sonra başladığı için el/alev hareket izi içerebilir. Plasebo bunu
  yakalamaz; S'nin ilk 5 s'i atılmış sürüm keşif olarak raporlanır. Örnekler öğrenciler tarafından birkaç ülkede toplandı.
- A: ECG-FM girdisi pencere başına z-skorlandığı için seviye değişimini göremez; HuBERT-ECG görebilir. Bu, modellerin
  kendi girdi biçimidir ve değiştirilmez.
- B: parçalar yazarlarca "sinyalde belirgin dalgalanma" şartıyla seçildi (iki sınıfa da uygulanıyor); örnekleme hızı
  belirtilmemiş, dosyadan okunur. 38 kayıt → güç orta.
- Pencere süreleri (A: 30 s, B: 16 s) modelin 5 s'ine sıkıştırılır; domatesteki zaman ölçeği bulgusu bu sürelerin seçiminde
  kullanılmadı (A: yazarların bildirdiği 3–6 s gecikmeyi ve tepkiyi kapsayacak süre; B: verinin kendi parça süresi).
