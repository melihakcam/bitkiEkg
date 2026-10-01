"""Zenodo veri setlerini indirir.

Kullanım örnekleri:
    python scripts/download_data.py --list                          # dosya listesi ve boyutlar
    python scripts/download_data.py --peek domates_su_stresi        # zip içeriği (yalnızca birkaç MB indirir)
    python scripts/download_data.py sarmasik_dis_ortam              # tüm dosyaları indir
    python scripts/download_data.py uyaran_siniflandirma --files classification_results.xlsx
    python scripts/download_data.py sarmasik_dis_ortam --extract    # indir ve zip'leri aç

İndirmeler `.part` dosyasına yazılır; bağlantı koparsa aynı komut kaldığı yerden devam eder.
Bittiğinde Zenodo'nun verdiği MD5 özetiyle doğrulanır.
"""

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from bitki_ekg.config import get_path, load_config

ZENODO_API = "https://zenodo.org/api/records/{}"
CHUNK = 1024 * 1024  # 1 MB
TIMEOUT = 60


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def record_files(zenodo_id: int) -> list[dict]:
    """Kayıttaki dosyaları [{name, size, md5, url}] olarak döndürür."""
    r = requests.get(ZENODO_API.format(zenodo_id), timeout=TIMEOUT)
    r.raise_for_status()
    files = []
    for f in r.json()["files"]:
        algo, _, digest = f["checksum"].partition(":")
        files.append({
            "name": f["key"],
            "size": f["size"],
            "md5": digest if algo == "md5" else None,
            "url": f["links"]["self"],
        })
    return files


# --- Zip içeriğini indirmeden listeleme -------------------------------------

class HttpRangeFile(io.RawIOBase):
    """HTTP Range istekleriyle okunan, yalnızca okunur ve konumlanabilir dosya.

    zipfile modülü yalnızca dosyanın sonundaki içerik dizinini okuduğundan
    büyük bir zip'in içindekiler birkaç MB indirilerek listelenebilir.
    """

    def __init__(self, url: str, size: int, session: requests.Session | None = None):
        self.url, self.size, self.pos = url, size, 0
        self.session = session or requests.Session()
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        data = self._fetch(self.pos, end - 1)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def _fetch(self, start: int, end: int) -> bytes:
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=TIMEOUT)
        if r.status_code != 206:
            raise RuntimeError(f"Sunucu Range isteğini desteklemiyor (HTTP {r.status_code})")
        self.bytes_fetched += len(r.content)
        return r.content


def peek_zip(f: dict) -> None:
    remote = HttpRangeFile(f["url"], f["size"])
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=256 * 1024)) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        print(f"\n  {f['name']}: {len(infos)} öğe, açılmış boyut {fmt_size(total)}")
        for i in infos:
            print(f"    {fmt_size(i.file_size):>10}  {i.filename}")
    print(f"  (listeleme için indirilen: {fmt_size(remote.bytes_fetched)})")


# --- İndirme ----------------------------------------------------------------

def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(f: dict, out_dir: Path) -> Path:
    target = out_dir / f["name"]
    if target.exists() and target.stat().st_size == f["size"]:
        print(f"  zaten var: {target.name}")
        return target

    part = target.with_name(target.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}

    with requests.get(f["url"], headers=headers, stream=True, timeout=TIMEOUT) as r:
        if done and r.status_code != 206:  # sunucu devam etmeyi desteklemiyor, baştan başla
            done = 0
        r.raise_for_status()
        with open(part, "ab" if done else "wb") as fh, tqdm(
            total=f["size"], initial=done, unit="B", unit_scale=True, desc=f["name"]
        ) as bar:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))

    if part.stat().st_size != f["size"]:
        raise RuntimeError(f"{f['name']}: eksik indirme, komutu tekrar çalıştırın (kaldığı yerden devam eder)")
    if f["md5"] and md5sum(part) != f["md5"]:
        part.unlink()
        raise RuntimeError(f"{f['name']}: MD5 doğrulaması başarısız, dosya silindi")
    part.replace(target)
    print(f"  doğrulandı: {target.name}")
    return target


def extract(path: Path) -> None:
    # .xlsx/.docx de içten zip'tir; yalnızca .zip uzantılılar açılır
    if path.suffix.lower() != ".zip" or not zipfile.is_zipfile(path):
        return
    dest = path.with_suffix("")
    print(f"  açılıyor: {path.name} -> {dest}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)


def main() -> int:
    cfg = load_config()
    datasets = cfg["datasets"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"veri seti anahtarları: {', '.join(datasets)}")
    ap.add_argument("--list", action="store_true", help="dosyaları ve boyutları listele, indirme yapma")
    ap.add_argument("--peek", action="store_true", help="zip dosyalarının içeriğini indirmeden listele")
    ap.add_argument("--files", nargs="+", help="yalnızca bu dosyaları indir")
    ap.add_argument("--extract", action="store_true", help="indirilen zip'leri aç")
    args = ap.parse_args()

    unknown = set(args.datasets) - set(datasets)
    if unknown:
        ap.error(f"bilinmeyen veri seti: {', '.join(sorted(unknown))}")
    if not (args.datasets or args.list or args.peek):
        ap.error("indirilecek veri setini belirtin ya da --list kullanın")
    names = args.datasets or list(datasets)

    raw = get_path("raw", cfg)
    for name in names:
        ds = datasets[name]
        files = record_files(ds["zenodo_id"])
        if args.files:
            files = [f for f in files if f["name"] in args.files]
        print(f"\n=== {name} (zenodo {ds['zenodo_id']}) — {len(files)} dosya, {fmt_size(sum(f['size'] for f in files))}")

        if args.list:
            for f in files:
                print(f"  {fmt_size(f['size']):>10}  {f['name']}")
            continue
        if args.peek:
            for f in files:
                if f["name"].endswith(".zip"):
                    peek_zip(f)
            continue

        out_dir = raw / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            path = download(f, out_dir)
            if args.extract:
                extract(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
