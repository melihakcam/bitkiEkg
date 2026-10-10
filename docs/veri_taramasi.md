# Bağımsız test için üç açık veri seti bulundu

**Kısa cevap:** Açık ve indirilebilir bitki elektrofizyolojisi verisi azdır. Elimizdeki iki veri dışında yaklaşık 15 kayıt bulundu. Zaman tuzağı testi için gereken şey şudur: stresli bitkiler ve kontrol bitkileri **aynı günlerde, yan yana** kaydedilmiş olmalı. Bu şartı açıkça karşılayan tek bağımsız veri **Matić vd. 2025 domates külleme verisidir** (Mendeley, 3 hasta + 3 sağlıklı bitki, 15 gün, CC BY 4.0) ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12557507/fullTextXML)). İkinci aday **Vitis bağ verisidir**. Bu veride 8 asma bir yıl boyunca tek cihazla aynı anda kaydedilmiş. Sağlıklı, hasta, iyileşen asmalar ve ölü kütükler bir arada ([Zenodo 16270285](https://zenodo.org/records/16270285)). Üçüncü aday **HIPB-MM** böcek verisidir. Bu verinin kendi resmî bölmesinde sızıntı var: test kayıtlarının hepsi eğitimde de geçiyor ([HF API](https://huggingface.co/api/datasets/JM1122/HIPB-MM/tree/main/HIPB-MM)). Literatürde de sızıntı kanıtı güçlü. 2026 tarihli bir sistematik derleme 57 çalışmayı inceledi. Yalnızca 4'ünde (%7) test tam bağımsızdı ([Zenodo 21317390](https://zenodo.org/records/21317390)). Ticari, iyi tasarımlı veriler (Vivent) ise yalnızca istek üzerine veriliyor. Aşağıdaki sonraki adımlar **öneridir**. Hiçbiri senin onayın olmadan başlamaz.

Terimler (tek cümleyle):
- **Zaman tuzağı (temporal confound):** Model stresi değil, kaydın hangi gün/saatte yapıldığını öğrenir.
- **Sızıntı (leakage):** Aynı bitkinin veya aynı kaydın parçaları hem eğitimde hem testte bulunur.
- **Paralel kontrol:** Stresli bitkiyle aynı anda, aynı ortamda kaydedilen stressiz bitki.

---

## Paralel kontrollü bağımsız veri yalnızca bir tane

Zaman tuzağını test etmek için en önemli soru şudur: kontrol bitkileri stresli bitkilerle aynı zaman aralığında kaydedildi mi? Çoğu veri bu şartı sağlamıyor. Birçok çalışmada "kontrol" aynı bitkinin stres öncesi saatleridir. Bu durumda stres ile zaman tamamen iç içe geçer. Örneğin Tran 2019 ve González 2023 çalışmalarında stres her zaman ilk günlerden sonra gelir ([PMC6864072](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6864072/fullTextXML); [PMC10267180](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10267180/)). Bu iki çalışma veri olarak kullanılamaz. Ama makalede **tuzağın örneği** olarak anılabilir.

### Tablo 1. Açık veri setleri: temel bilgiler

| # | Veri seti | Bağlantı / DOI | Tür | Uyaran / stres | Bitki / cihaz | Örnekleme hızı | Süre | Boyut | Lisans | Erişim | Makale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Matić vd. 2025, domates külleme** | [10.17632/yr8zhsc6mh.1](https://data.mendeley.com/datasets/yr8zhsc6mh/) | Domates | *Oidium neolycopersici* (külleme) | 3 aşılanmış + 3 kontrol; 30 kanal; Agilent 34970A; deney 2 kez yapıldı | ~200 s'de bir tarama (çok yavaş) | 15 gün | 60,6 MB, 795 dosya | CC BY 4.0 | Doğrudan | Data in Brief 112165; Comput. Electron. Agric. 237 (2025) 110585 |
| 2 | **Vitis vinifera bağ elektromu** | [Zenodo 16270285](https://zenodo.org/records/16270285) | Asma | Sağlık durumu (sağlıklı / Flavescence dorée hastası / iyileşen / ölü kütük) + kesik kütük | 8 bitki, tek DATAQ DI-710 cihaz; ~16 CSV | 1 Hz | 01.01.2023–18.12.2023 hava verisi | 170,6 MB | CC BY 4.0 | Doğrudan | Biomimetics 2025, 10.3390/biomimetics10090636 |
| 3 | **Marul asit/tuz stresi** | [Zenodo 19386720](https://zenodo.org/records/19386720) | Marul | Kontrol, pH 3, pH 2, 300 mM NaCl, 500 mM NaCl, pH 2 + NaCl | 60 bitki, 6 grup (10'ar) | 0,1 Hz | Bitki başına 8 saat | 8,9 MB | CC BY 4.0 | Doğrudan | "ML-enabled implantable plant biomarker sensor" (Zhou) |
| 4 | **PhytoNodes uyaran sınıflama** (Buss vd.) | [Zenodo 6618131](https://zenodo.org/records/6618131) | *Zamioculcas zamiifolia* | Rüzgâr, sıcaklık, mavi ışık, kırmızı ışık, **uyaran yok** | 2 cihaz (lrpi0, lrpi1) | Belirtilmemiş | 2 sa 10 dk'lık bloklar, 23.01–18.03.2022 | 4,4 GB (ham 111 MB) | CC BY 4.0 | Doğrudan | GoodIT 2022 |
| 5 | **Potansiyel + empedans** (Buss vd.) | [Zenodo 7126105](https://zenodo.org/records/7126105) | *Zamioculcas* + domates | Rüzgâr, ısı, kırmızı/mavi ışık | Belirtilmemiş | Belirtilmemiş | Belirtilmemiş | ~4,3 GB | CC BY 4.0 | Doğrudan | Bioinspir. Biomim. 17(6) 2022 |
| 6 | **PhytoNode Upgraded** (Buss vd.) | [Zenodo 11400865](https://zenodo.org/records/11400865) | Belirtilmemiş | Ozon, gece/gündüz, yaprak sıcaklığı | Belirtilmemiş | Belirtilmemiş | Deney 1: 05–14.05.2024 | 1,65 GB | CC BY 4.0 | Doğrudan | FICC 2025 |
| 7 | **Sarmaşık ısı/ozon** (Aust vd.) | [Zenodo 15696845](https://zenodo.org/records/15696845) | Sarmaşık | Isı (~+6 °C), ozon (~1400 ppb) | 8 eğitim + 2 yeni test bitkisi; 151 ısı deneyi, 2'şer bitki paralel | 100 Hz, 2 kanal | 30 dk uyaran, günde 5 kez | 15,7 GB | CC BY 4.0 | Doğrudan | [arXiv 2509.24992](https://arxiv.org/html/2509.24992) |
| 8 | **HIPB-MM böcek otlaması** | [HF JM1122/HIPB-MM](https://huggingface.co/datasets/JM1122/HIPB-MM) | Belirtilmemiş | 3 zararlı türü, 4 sınıf (M, N, T, X) | 53 kayıt (M 17, N 11, T 21, X 4) | Belirtilmemiş | 16 s pencereler, 4.023 parça | ~143 MB | CC BY 4.0 | Doğrudan | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/833) |
| 9 | **LIB veri seti** (Tao vd.) | [Zenodo 14557734](https://zenodo.org/records/14557734) | Doğrulanmadı | Tuz stresi, ışıkla uyarılan sinyal | Derlemeye göre 1.600 fide | Belirtilmemiş | Belirtilmemiş | 6,2 MB | CC BY 4.0 | Doğrudan | Measurement 2026, 10.1016/j.measurement.2025.119238 |
| 10 | **Madariaga kütüphanesi** | [Figshare 25425920](https://doi.org/10.6084/m9.figshare.25425920.v2), [24161100](https://doi.org/10.6084/m9.figshare.24161100.v1) | 16 tür (sinek kapan, küstüm otu, domates…) | Alev, dokunma | 89 bitki, 398 kayıt | 10 kHz (WAV) | Saniye–dakika | 68 + 84 MB | CC BY 4.0 | Doğrudan | [PMC10950275](https://pmc.ncbi.nlm.nih.gov/articles/PMC10950275/) |
| 11 | **Sinek kapan stres** (Bergstrom) | [Zenodo 21048107](https://zenodo.org/records/21048107) | *Dionaea* | Tetik tüyü, ısıyla yaralama | 10 kayıt | 44,1 kHz | Kısa | 19,7 MB | CC BY 4.0 | Doğrudan | — |
| 12 | **Gloor: çuha çiçeği** | [figshare 32180721](https://api.figshare.com/v2/articles/32180721) | *Primula vulgaris* | Gün döngüsü, ziyaretçi | 3 istasyon | 100 Hz | Mart–Nisan 2026 | 142 MB | CC BY 4.0 | Doğrudan (dosya adları açıklamayla uyuşmuyor) | — |
| 13 | **Gloor: jest deneyi** | [figshare 30227083](https://api.figshare.com/v2/articles/30227083) | Fesleğen, marul, domates | İnsan hareketi | Deney + "yalıtılmış" kontrol bitkileri | 142 Hz | Kısa WAV parçaları | 4,8 MB | CC BY 4.0 | Doğrudan | — |
| 14 | **Gloor: tek Kalanchoe** | [figshare 32112583](https://doi.org/10.6084/m9.figshare.32112583.v1) | *Kalanchoe* | İnsan duygusu, CO2 | 1 bitki | 1 Hz | 7 oturum | 2,6 MB | CC BY 4.0 | Doğrudan | — |
| 15 | **Ladin, güneş tutulması** (Chiolerio) | [Dryad 10.5061/dryad.brv15dvhr](https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.brv15dvhr) | *Picea abies* | Güneş tutulması (gözlemsel) | 5 ağaç + 5 kütük | Belirtilmemiş | Belirtilmemiş | 15 MB | CC0 | **Belirsiz** (aşağıya bak) | [PMC12040458](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12040458/fullTextXML) |
| 16 | **Soya elektromu** (Carvalho Oliveira 2025) | [OneDrive](https://1drv.ms/f/c/f5058fece847c076/EnbAR-jsjwUggPWzMQsAAAABMGG5lVbbL0I9Etnd7VKOaA?e=ymv167) | Soya | Yakma, tuz | 12 bitki | 62,5 Hz | 2 sa önce + 3 sa sonra | Bilinmiyor | Belirtilmemiş | Bağlantı açılmadı | [PMC12599359](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12599359/fullTextXML) |

Not: 4, 5, 6 ve 7 numaralı kayıtlar bizim domates verimizle **aynı laboratuvardan** (Buss/Hamann). Farklı deney ve tür, ama bağımsız laboratuvar sayılmaz ([Zenodo 6618131](https://zenodo.org/records/6618131)). 6759849 numaralı WatchPlant kaydı da bu gruptandır ve 4 numaradaki ışık verisiyle örtüşür ([Zenodo 6759849](https://zenodo.org/records/6759849)).

### Tablo 2. Zaman tuzağı testine uygunluk

| # | Veri seti | Kontrol aynı anda mı? | Uygun mu? | Neden |
|---|---|---|---|---|
| 1 | Matić domates | **Evet.** Aynı tarayıcıda, 15 gün boyunca | **Evet** | Başka laboratuvar, paralel kontrol, iki tekrar. "Yalnızca zamanı bilen model hastayı ayırabiliyor mu?" sorusu doğrudan sorulabilir. Zayıf yanı: 3'e 3 bitki ve çok yavaş örnekleme. |
| 2 | Vitis bağ | **Evet.** Tüm asmalar tek cihazda aynı anda | **Kısmen** | Uzun süre ve hava verisi var. Ölü kütükler "canlı olmayan" kontrol işi görür. Ama her asmanın durumu sabit: bitki kimliği ile sınıf aynı şey. Bitki-dışarıda-bırak (leave-one-plant-out) bölme şart. |
| 3 | Marul | Kontrol grubu var; **aynı anda mı, ayrı partilerde mi bilinmiyor** | **Kısmen** | 60 bitki iyi. Ama pencere 8 saat ve sensör klasik yüzey potansiyeli değil, elektrokimyasal biyobelirteç sensörü. |
| 4 | PhytoNodes 6618131 | Hayır. "Uyaran yok" sınıfı aynı düzenekte ama zamanda sırayla | **Kısmen** | Saat/blok tuzağını test etmek için iyi. Uyaranlar planlı saat bloklarına bağlı. Bu, dosya adlarından çıkarım; içerik kontrol edilmedi. |
| 5 | 7126105 | Bilinmiyor | **Kısmen** | 4 numarayla aynı dosyaların bir kısmını içeriyor. Bitki sayısı belirsiz. |
| 6 | 11400865 | Hayır | **Hayır** | Stres/kontrol tasarımı yok. Bir derleme bunu aynı bitkide saatlik bloklarla değerlendirilmiş olarak kodluyor ([Zenodo 21317390](https://zenodo.org/records/21317390)). |
| 7 | Sarmaşık ısı/ozon | Hayır. Negatif sınıf, uyaran öncesi dönem | **Kısmen** | Yeni bitkilerle test var. Ama ayrı kontrol bitkisi yok, aynı laboratuvar. |
| 8 | HIPB-MM | Bilinmiyor | **Sızıntı için Evet, zaman tuzağı için Hayır** | Resmî bölmede sızıntı kanıtlandı. Tür, hız ve sınıf anlamları bilinmiyor. |
| 9 | LIB | Bilinmiyor | **Hayır (şimdilik)** | Tür ve tasarım doğrulanmadı. |
| 10–11 | Madariaga, sinek kapan | Hayır | **Hayır** | Kayıtlar saniye–dakika. Paralel kontrol yok. |
| 12 | Gloor çuha çiçeği | 3 istasyon aynı anda; stres yok | **Kısmen** | "Ortak çevre kayması" negatif kontrolü olabilir. Dosyalar önce kontrol edilmeli. |
| 13–14 | Gloor jest / Kalanchoe | 13'te yalıtılmış kontrol var; 14'te tek bitki | **Hayır** | Çok kısa ya da tek bitki. |
| 15 | Ladin | Stres yok, ağaçlar aynı anda | **Hayır** | Gözlemsel; stres deneyi değil. |
| 16 | Soya | Bitkiler yan yana; uyaran oturum içinde önce/sonra | **Hayır** | Önce/sonra tasarımı, zamanla iç içe. |

---

## Notlar arasında üç çelişki var

**Matić bitki sayısı.** Bir not, Data in Brief makalesinin tam metnine dayanır. Orada 3 aşılanmış + 3 aşılanmamış kontrol bitkisi vardır. Kontroller aynı anda, aynı elektrot düzeniyle izlenmiştir ([Europe PMC PMC12557507](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12557507/fullTextXML)). Başka bir yapay zekâ ise 24 bitki dedi. Bu sayı doğrulanamadı; PDF'ler 403 hatası verdi. **Doğrulanmış olan: 3 + 3 bitki, deney iki kez yapıldı.** Toplam bitki sayısı iki tekrarla birlikte farklı olabilir; bu açık bir soru.

**Ladin verisinin durumu.** Bir notta Dryad API'si kaydı "submitted" gösteriyor, yani henüz yayımlanmamış olabilir ([Dryad API](https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.brv15dvhr)). Doğrulama notunda ise Dryad sayfası açılmış ve 5 ağaç + 5 kütük, 15 MB görülmüş. **Durum: içerik görüldü, ama indirme izni kesin değil.** Testimiz için zaten zayıf aday.

**Vitis verisinin yeri.** Makale verinin "Dryad'a konacağını" söylüyor (gelecek zaman). Bulunan veri ise Zenodo'da ([Europe PMC PMC12467882](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12467882/fullTextXML)). **Doğrulanmış olan: Zenodo kaydı.**

---

## Başka yapay zekânın listesi kontrol edildi: biri uydurma

Başka bir yapay zekânın önerdiği liste sayfalar açılarak tek tek kontrol edildi.

| Önerilen | Sonuç |
|---|---|
| Mendeley yr8zhsc6mh (Matić) | **Var.** Ama "24 bitki" doğrulanamadı (yukarıya bak). Makale, enfekte bitkilerin tüm süre boyunca daha düşük potansiyelde olduğunu söylüyor. Bu, bitki/kanal taban farkı olabilir. Ayrıca turba ile su ortamı %97,5 doğrulukla ayrılıyor. Bu ikinci bir karıştırıcı. |
| **Zenodo 22081982** | Var ama **bağımsız değil.** Bizim domates verimizin (18876513) eşlik dosyaları: sonuçlar, kalibrasyon, sera iklimi, fotoğraflar. 19,4 MB, ham sinyal yok ([Zenodo 22081982](https://zenodo.org/records/22081982)). Projede grup eşlemesi için zaten kullanıldı. |
| Picea abies (Dryad / Figshare c.7758791) | Var (Dryad açıldı, Figshare 403). Stres deneyi değil. |
| **Figshare 22763590 "PLEASED Chemical Stress"** | **404, bulunamadı. Aramada da çıkmadı. Büyük olasılıkla uydurma.** |
| Najdenovska 2021 (kırmızı örümcek) | Makale var, veri açık değil. Vivent'ten istek gerekir. |
| gmsouza@ufpel.edu.br (Souza) | Doğrulanmadı. |
| Zenodo 20466876 (sinek kapan yanma) | Var ama yalnızca 14 kB özet xlsx. Ham sinyal yok. |
| Zenodo 10017659, 13169748, PlantES | Kontrol edilmedi. |

O metindeki iki hata da kayda geçmeli. "1–5 Hz sirkadiyen ritim" yanlıştır; gün ritmi yaklaşık 24 saatliktir (1/86.400 Hz). Ayrıca metin bizim bulgularımızı (AUC 0,82, +20 puan) kendi bulgusu gibi tekrar ediyor. Bu kaynak olarak kullanılamaz.

---

## İstek üzerine alınabilecek veriler en iyi tasarımlara sahip

Hedefimize en yakın tasarımlar (çok bitki, kontrol, günlerce kayıt) ticari şirket Vivent'tedir. Erişim tamamen şirketin kararına bağlıdır ([HEIA-FR PDF](https://www.heia-fr.ch/media/mzdfunau/published-version.pdf)).

| Veri | Tasarım | Kime yazılır | Kaynak |
|---|---|---|---|
| Najdenovska 2021b (genel stres) | 36 domates (12 akar, 12 kuraklık, 12 besin eksikliği), 500 Hz | Nigel Wallbridge, research@vivent.ch | [HEIA-FR PDF](https://www.heia-fr.ch/media/mzdfunau/published-version.pdf) |
| González i Juclà 2023 (azot eksikliği) | 16 domates, kontrol bitkileri tam besinde, 500 Hz, 15 gün | research@vivent.ch | [PMC10267180](https://pmc.ncbi.nlm.nih.gov/articles/PMC10267180/) |
| Tran 2019 (sulama) | Domates, 400 Hz, ~2 hafta, sıralı program | Veri beyanı yok; Vivent | [PMC6864072](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6864072/fullTextXML) |
| Vivent domates/kayısı kuraklık (bioRxiv 2025) | Tür başına 8 kontrol + 8 kurak bitki | Vivent | [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.03.692091v1.full-text) |
| Li 2024, Plant Methods (Arabidopsis yaralama, karanlık) | 55 bitki, 100 Hz | Sorumlu yazar (Wu Q) | [PMC10964643](https://pmc.ncbi.nlm.nih.gov/articles/PMC10964643/) |
| Friedrich/Kurenda 2025 (soya, kokarca böceği) | — | Sorumlu yazar | [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41598-025-25530-2&format=json) |
| de Toledo 2024 (fasulye; su, PEG, NaCl) | 144 bitki, oturumda 3 bitki + ölü doku referansı, 62,5 Hz, 4 sa | Veri beyanı yok; yazarlar (Souza grubu, e-posta doğrulanmadı) | [PMC10984121](https://pmc.ncbi.nlm.nih.gov/articles/PMC10984121/) |
| Dış mekân ağaçları (bioRxiv 2024) | 6 tür + ölü dal, 400 Hz, 18–36 hafta | Veri beyanı yok | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.03.626606v1.full) |
| PLEASED (Chatterjee) | Domates/lahana; NaCl, H2SO4, ozon; 10 Hz | Chatterjee / Vitaletti (e-posta notlarda yok) | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0285321) |

Qi 2026 (Clivia) ve Sukhov 2021 (bezelye yakma) da "istek üzerine" diyor ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1186/s13007-024-01169-4&format=json)).

**Elenen kaynaklar ve nedenleri**

| Kaynak | Neden elendi |
|---|---|
| Zenodo 22081982 | Kendi domates verimizin eşlik dosyaları; ham sinyal yok |
| Zenodo 20577639 (TAMC-PLANTAS) | Madariaga kütüphanesinin yeniden paketlenmiş kopyası |
| Zenodo 20466876 | 14 kB özet, ham sinyal yok |
| Figshare 22763590 | 404; büyük olasılıkla uydurma |
| PLEASED MEGA klasörü | Bağlantı açık, ama API 21 klasör ve **0 dosya** döndürdü; site başka içeriğe dönmüş ([Wayback](http://web.archive.org/web/20200811160451/https://pleased-fp7.eu/dataset/)) |
| Figshare 33084887 (bakla) | Yalnızca 37 kB türetilmiş değer, ham iz yok |
| Kaggle "Plant-Health-Data" | Sentetik/tablo veri gibi görünüyor ([baselight](https://baselight.app/u/kaggle/dataset/ziya07_plant_health_data)) |
| HF raghavkataria00/plant-bio-signals | ~2 kB csv, kullanılamaz |
| Zenodo 22095454 (Mind(the)Plant) | Elektrofizyoloji içerdiği doğrulanmadı |
| Dryad, Dataverse, OSF aramaları | İlgili veri çıkmadı ([Dryad API](https://datadryad.org/api/v2/search?q=plant%20electrophysiology)) |

Mendeley API'si ve Kaggle doğrudan taranamadı. Science Data Bank hiç taranmadı. Yani liste tam değildir.

---

## Literatür sızıntının yaygın olduğunu açıkça gösteriyor

Makale için en güçlü kaynak **Wahid & Nambo 2026 sistematik derlemesidir**. Derleme 57 makine öğrenmesi çalışmasını inceledi ([Zenodo 21317390](https://zenodo.org/records/21317390)):

| Ölçü | Değer |
|---|---|
| Hiç tahmin doğrulaması yapmayan çalışma | %40,4 |
| Sızıntıya açık çalışma | 13/57 (%22,8) |
| Testi tam bağımsız çalışma | 4/57 (%7,0) |
| Sızıntıya açık değerlendirme tasarımı | 17/42 (%40,5) |
| Açık veri/kod paylaşan çalışma | 8/57 |

**HIPB-MM resmî bölmesi.** Notlarda HF dosya listesi tek tek sayıldı. Eğitim 3.218, test 805 parça. Her sınıfta **testteki kayıtların hepsi eğitimde de var** (17/17, 11/11, 21/21, 4/4). Yani bölme kayıt düzeyinde değil, parça düzeyinde rastgele. X sınıfı yalnızca 4 kayıttan geliyor ([HF API](https://huggingface.co/api/datasets/JM1122/HIPB-MM/tree/main/HIPB-MM)). Makale %69,81 doğruluk bildiriyor ([IJCAI 2026](https://www.ijcai.org/proceedings/2026/833)). Bu kontrol bizim yaptığımız bir sayımdır; yayımlanmış bir bulgu değildir.

**Doğruluk, bölme türüne göre düşüyor.** Rastgele pencere bölmeli çalışmalar %95–99 bildiriyor. Bitki ayrık testlerde sonuçlar %76–88 civarında. Sarmaşık ozon testinde %76,96 ([arXiv 2412.13312](https://arxiv.org/html/2412.13312)), azot eksikliğinde tek tahminde ~%88 ([PMC10267180](https://pmc.ncbi.nlm.nih.gov/articles/PMC10267180/)). Gerçek kullanımda ozon doğruluğu %46,7'ye iniyor ([arXiv 2509.24992](https://arxiv.org/html/2509.24992)). Sızıntıya açık diğer örnekler: Reissig 2021 domates olgunlaşması (karıştırılmış pencerelerde StratifiedKFold) ([Frontiers](https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2021.696829/full)), PLEASED (rastgele 50:50) ([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0285321)) ve Gloor 2025 (tek bitki, parça düzeyinde bölme) ([PMC12649952](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12649952/fullTextXML)). Not: birçok makalenin bölme bilgisi derlemeden alındı. Birincil metin açılamadı (MDPI, Springer 403).

**Zaman tuzağı neredeyse hiç ele alınmamış.** Çoğu çalışmada kontrol, aynı bitkinin uyaran öncesi dönemidir. Paralel kontrol bitkisi kaydeden az sayıda istisna var: Vivent azot çalışması ve Reissig 2021a (deney başına 1 kontrol + 4 işlem) ([Zenodo 21317390](https://zenodo.org/records/21317390)). Bu, makalemizin özgünlük iddiasını destekliyor.

---

## Önerilen ilk üç ve sonraki adımlar (onay bekliyor)

| Sıra | Veri | Makaledeki rolü | Ana risk |
|---|---|---|---|
| 1 | **Matić domates külleme** | Zaman tuzağı testinin bağımsız tekrarı: başka laboratuvar, paralel kontrol | 3'e 3 bitki; ~200 s'lik örnekleme EKG modellerine uymayabilir (çıkarım); kanal–bitki eşlemesi henüz bilinmiyor |
| 2 | **Vitis bağ** | Uzun süreli, aynı anda kaydedilmiş bitkiler; ölü kütükler negatif kontrol | Bitki = sınıf; yalnızca 8 bitki; 1 Hz |
| 3 | **HIPB-MM** | Başka bir grubun açık verisinde sızıntının gösterimi (kayıt düzeyinde yeniden bölme) | Tür, hız ve sınıf anlamı bilinmiyor; ham kayıt klasörü eksik olabilir |

Yedek: **PhytoNodes 6618131** saat/blok tuzağı için. Ama bizimle aynı laboratuvardan.

Aşağıdaki adımlar **karar değildir, öneridir**. Proje kuralı gereği her biri senin onayını bekler:

| Adım | Ne yapılır | Maliyet |
|---|---|---|
| A | Matić verisini indir (60 MB); Excel dosyalarından kanal → bitki → grup eşlemesini çıkar | Düşük |
| B | Matić'te "yalnızca zaman" modelini ve bitki-dışarıda-bırak testini çalıştır | Orta |
| C | Vitis zip'ini indir (170 MB); 16 CSV'yi bitkilere ve sağlık durumuna eşle | Düşük |
| D | HIPB-MM'yi kayıt düzeyinde yeniden böl; resmî bölmeyle karşılaştır | Düşük–orta; kapsam dışı bir yan iş, ayrıca onay gerekir |
| E | Vivent'e (research@vivent.ch) veri isteği e-postası yaz | Düşük; sonuç Vivent'in kararına bağlı |
| F | Marul verisinde grupların aynı anda mı kaydedildiğini README/xlsx'ten kontrol et | Düşük |

## Sonuç

Asıl sonuç şudur: alanda paralel kontrollü açık veri neredeyse yok. Bu bir eksik değil, makalenin bir bulgusu olabilir. "Sızıntı ve zaman tuzağı yaygın, ama test etmek için gereken veri tasarımı da nadir" demek mümkün. Literatür sayıları (57 çalışmadan 4'ünde tam bağımsız test) bunu destekliyor. Bizim domates verimiz, Matić ile birlikte, bu nadir tasarımın iki örneğinden biri olur.

İkinci sonuç bir uyarıdır. En uygun bağımsız veriler (Matić ~200 s, Vitis 1 Hz) çok yavaş örneklenmiş. Bu yüzden zaman tuzağı testine uygunlar. Ama insan EKG temel modellerinin aktarımını test etmeye uymayabilirler. Bu bir çıkarımdır; veriye bakılmadan kesinleşmez. Yani bağımsız tekrar büyük olasılıkla makalenin **zaman tuzağı** kısmını güçlendirir, **EKG aktarımı** kısmını değil. Yüksek hızlı ve paralel kontrollü veri (Vivent, 500 Hz) ancak istek üzerine alınabilir.

## Karşılaştırma çalışması için tasarım kontrolü (2026-10-10)

Kurallar: (1) yeterli bağımsız birim, (2) stres ile kontrol aynı zamanda, (3) grup ile cihaz/oturum çakışmıyor,
(4) etiket sinyalden bağımsız, (5) açık lisans. Yalnızca açıklama, makale yöntemi, dosya listesi ve sütun
başlıklarına bakıldı; sinyal değerlerine bakılmadı (ön kayıt temiz kalsın diye).

| Veri | Karar | Gerekçe |
|---|---|---|
| Asma (Zenodo 16270285) | ❌ | 8 asma + 1 kütük; sağlık etiketi bitkiyle birebir (agronom gözlemi). Makalede sinyalden sağlık sınıflaması yok, ML yalnızca hava tahmini. Sinyal ~242 sa (bir yıl olan hava verisi). |
| Marul (Zenodo 19386720) | ❌ | Elektrofizyoloji değil: sütunlar `Time, Plant_ID, Group, K_Ratio, pH_Value, H2O2_Conc` (elektrokimyasal derişim). Asit stresini pH ile ölçmek döngüsel. |
| Madariaga (figshare 24161100, 82,6 MB) | 🟡 | Plant SpikerBox, 10 kHz, .wav; 16 tür, 89 bitki, 398 kayıt; uyaran başı/sonu elle işaretli (.txt). Alev/dokunma görevi geçersiz (dokunma yalnızca sinekkapan ve küstüm otunda → tür ile çakışık). Kullanılabilir görev: aynı kayıtta uyaran öncesi/sonrası. Risk: elle işaretleme ve uyaran sırasında hareket izi; "tepki var" etiketi eşikten türetilmiş (kullanılmamalı). |
| Sarmaşık ısı/ozon (Zenodo 15696845, 15,7 GB) | ❌ | Isı günde 5 kez **sabit saatlerde** (08:00–20:30) → uyaran günün saatiyle çakışık; kontrol bitkisi tanımlı değil; uyaran zamanlarının dosyada işaretli olduğu belirtilmemiş; Buss/Hamann laboratuvarı (bağımsız değil). |
| HIPB-MM (HF JM1122/HIPB-MM) | 🟡 | *Arabidopsis*, tek cihaz (Keithley 2401), 16 s parçalar. Sınıflar (parça sayısından eşlendi): M = *H. armigera* (17 kayıt), T = *S. exigua* (21 kayıt), X = *P. xylostella* (4 kayıt, 18–20.05), N = böceksiz (11 kayıt, **tek gün** 25.05). N ve X tarihle çakışık → kullanılmaz. **M ile T aynı günlerde (13–17.05)** kaydedilmiş → kayıt-dışarıda-bırak ile M/T görevi kullanılabilir. Risk: parçalar "sinyalde belirgin dalgalanma" şartıyla seçilmiş (iki sınıfa eşit uygulanıyor). |
| Külleme, Matić (Mendeley yr8zhsc6mh) | 🟡 keşif | 3 deney × 12 bitki (6 aşılı + 6 sağlıklı), aynı anda, tek kaydedici; etiket aşılamadan. Daha önce incelendi (sonuclar §14) → ön kayıtlı sayılamaz, keşif olarak raporlanır. Yalnızca ham deney sayfaları kullanılmalı ("ABC averages" değil). ~226 s aralık. |

**Sonuç (2026-10-10):** Uygun ya da koşullu uygun veri 3 ayrı laboratuvardan: Buss (domates, vaka), Madariaga (uyaran öncesi/sonrası),
HIPB-MM (M/T); Matić keşif olarak eklenebilir. "En az 3 laboratuvar" koşulu sağlandı, ama hepsi koşullu. Sıradaki adım: ön kayıt.
