"""Zamioculcas uyaran verisi (Buss vd. 2023, Zenodo 7126105): cihaz karıştırıcısı testi.

Sıcak sınıfı yalnızca lrpi0, rüzgâr sınıfı yalnızca lrpi1 cihazından geliyor. Yazarların kendi rastgele
%70/%30 bölmesiyle (train_stat/test_stat), sınıf yalnızca UYARAN ÖNCESİ 340 örnekten tahmin edilir.
Uyaran öncesinde sınıf bilgisi olamaz; yüksek doğruluk cihaz/zaman kimliğinin öğrenildiğini gösterir.
Etiketler: 0 rüzgâr (lrpi1), 1 sıcak (lrpi0), 2 uyaran yok (iki cihaz), 3 mavi, 4 kırmızı ışık (2021, ayrı seri).

Kullanım: python scripts/uyaran_cihaz_kontrol.py
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
D="D:/ekg/data/raw/uyaran_siniflandirma/SupplementaryCode/SupplementaryCode/datasets/"
tr=np.genfromtxt(D+"train_stat.tsv"); te=np.genfromtxt(D+"test_stat.tsv")
print("shape",tr.shape,te.shape,"labels",np.unique(tr[:,0],return_counts=True))
B=340
def feat(x):
    t=np.arange(x.shape[1])
    return np.c_[x.mean(1),x.std(1),np.polyfit(t,x.T,1)[0],x.min(1),x.max(1),np.median(x,1)]
for name,sl in [("ONCE (uyaran oncesi)",slice(1,B+1)),("SONRA (uyaran sonrasi)",slice(B+1,None))]:
    for cls in [None,(0,1),(0,2),(1,2),(0,1,2)]:
        mtr=np.ones(len(tr),bool) if cls is None else np.isin(tr[:,0],cls)
        mte=np.ones(len(te),bool) if cls is None else np.isin(te[:,0],cls)
        m=RandomForestClassifier(500,random_state=0,n_jobs=-1).fit(feat(tr[mtr,sl]),tr[mtr,0])
        p=m.predict(feat(te[mte,sl]))
        chance=np.bincount(te[mte,0].astype(int)).max()/mte.sum()
        print(f"{name:24s} siniflar={cls or 'hepsi'}: dogruluk={accuracy_score(te[mte,0],p):.3f} (cogunluk={chance:.3f})")
