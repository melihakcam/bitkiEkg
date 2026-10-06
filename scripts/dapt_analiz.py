import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = r"D:\ekg\results"
ft = pd.read_csv(rf"{R}\colab\dapt\ozet_ince_ayar_6h.csv")
so = pd.read_csv(rf"{R}\colab\dapt\sonda_6h.csv")
k = ft.dogruluk * 23
assert np.allclose(k, k.round()), "ince ayar doğruluk*23 tam sayı değil"
lo = so[so.bolme == "lopo"].assign(bitki=lambda d: d.katman.str.split().str[-1].astype(int))
assert np.allclose(lo.dogruluk * 23, (lo.dogruluk * 23).round()), "sonda doğruluk*23 tam sayı değil"
print("satır:", len(ft), len(so), "| kollar:", ft.groupby("kol").size().to_dict())

t = pd.read_csv(rf"{R}\tables\temel_modeller_katman.csv")
t = t[(t.pencere == "6h") & (t.bolme == "lopo")].assign(bitki=lambda d: d.katman.str.split().str[-1].astype(int))

tab = {}
for kol in ["ekg", "rastgele", "ekg_dapt", "rastgele_dapt"]:
    tab[f"ince ayar: {kol}"] = ft[ft.kol == kol].set_index("bitki")[["dogruluk", "auc"]]
    tab[f"donmuş: {kol}"] = lo[lo.kol == kol].set_index("bitki")[["dogruluk", "auc"]]
for m in ["tsfresh + LightGBM", "MiniRocket"]:
    tab[m] = t[t.model == m].set_index("bitki")[["dogruluk", "auc"]]

ozet = pd.DataFrame({ad: {"dogruluk": d.dogruluk.mean() * 100, "dog_std": d.dogruluk.std() * 100,
                          "auc": d.auc.mean() * 100} for ad, d in tab.items()}).T.round(1)
print(ozet.sort_values("dogruluk", ascending=False).to_string())


def kars(a, b):
    for o in ("dogruluk", "auc"):
        x = (tab[a][o] - tab[b][o]).dropna() * 100
        rng = np.random.default_rng(42)
        bs = rng.choice(x.to_numpy(), (10000, len(x))).mean(1)
        p = wilcoxon(x).pvalue if (x != 0).any() else 1
        print(f"{a:28s} − {b:24s} {o:9s}: {x.mean():+5.1f} [{np.percentile(bs, 2.5):+.1f}; {np.percentile(bs, 97.5):+.1f}] "
              f"p={p:.3f}  ({int((x > 0).sum())}/{len(x)} bitkide önde)")


print()
kars("ince ayar: ekg_dapt", "ince ayar: rastgele_dapt")
kars("ince ayar: ekg", "ince ayar: rastgele")
kars("ince ayar: ekg_dapt", "ince ayar: ekg")
kars("donmuş: ekg_dapt", "donmuş: rastgele_dapt")
kars("donmuş: ekg", "donmuş: rastgele")
kars("donmuş: ekg_dapt", "donmuş: ekg")
kars("ince ayar: ekg_dapt", "tsfresh + LightGBM")
kars("donmuş: ekg_dapt", "tsfresh + LightGBM")
kars("donmuş: ekg", "tsfresh + LightGBM")
