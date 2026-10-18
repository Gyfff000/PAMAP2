"""Download the official archive; resume with validated HTTP Range when supported."""
from pathlib import Path
import time
import urllib.request
import zipfile
from .common import ROOT, save_json, sha256

URL = 'https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip'

def main():
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / 'uci_pamap2.zip'
    if not archive.exists():
        part = archive.with_suffix('.zip.part')
        for attempt in range(5):
            try:
                offset = part.stat().st_size if part.exists() else 0
                headers = {'User-Agent': 'PAMAP2-Coursework/1.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=90) as r:
                    append = offset > 0 and r.status == 206 and r.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
                    if not append:
                        offset = 0
                    count = offset
                    last = time.monotonic()
                    with part.open('ab' if append else 'wb') as f:
                        while block := r.read(1024 * 1024):
                            f.write(block)
                            count += len(block)
                            if time.monotonic() - last > 8:
                                print(f'Downloaded {count / 1024**2:.1f} MiB', flush=True)
                                last = time.monotonic()
                with zipfile.ZipFile(part) as z:
                    if z.testzip() is not None:
                        raise ValueError('Archive CRC validation failed')
                part.replace(archive)
                break
            except Exception as e:
                print(f'Download attempt {attempt+1}: {type(e).__name__}: {e}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(2)
    with zipfile.ZipFile(archive) as outer:
        names = outer.namelist()
        inner = next((n for n in names if n.endswith('PAMAP2_Dataset.zip')), None)
        if inner:
            target = raw / 'PAMAP2_Dataset.zip'
            if not target.exists():
                with outer.open(inner) as src, target.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            dataset = target
        else:
            dataset = archive
    with zipfile.ZipFile(dataset) as z:
        for info in z.infolist():
            name = info.filename.replace('\\', '/')
            if '/Protocol/' in name and name.endswith('.dat'):
                target = raw / 'Protocol' / Path(name).name
            elif name.endswith('.pdf'):
                target = raw / 'documentation' / Path(name).name
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != info.file_size:
                import shutil
                with z.open(info) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
    save_json(ROOT / 'outputs/data_source.json', {
        'url': URL, 'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'archive_sha256': sha256(archive), 'archive_bytes': archive.stat().st_size,
        'dataset_doi': '10.24432/C5NW2H', 'license': 'CC BY 4.0',
        'protocol_files': [{ 'name': f.name, 'sha256': sha256(f), 'bytes': f.stat().st_size}
                           for f in sorted((raw / 'Protocol').glob('*.dat'))]})
    print('Official Protocol recordings extracted and hashed.', flush=True)

if __name__ == '__main__':
    main()
