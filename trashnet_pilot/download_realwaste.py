"""Download the five materials from the original author's pinned release."""
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = '156e8e9154eeb163df30fcfd767f55b19633bb06'
BASE = f'https://raw.githubusercontent.com/sam-single/realwaste/{COMMIT}/'
COUNTS = {'Cardboard': 461, 'Glass': 420, 'Metal': 790, 'Paper': 500, 'Plastic': 921}
ARCHIVE = ROOT/'data/downloads/realwaste-author-156e8e9.zip'


def fetch(item):
    category, index = item
    name = f'{category}_{index}.jpg'
    target = ROOT/'data/external_realwaste/images'/category/name
    if not target.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE+f'RealWaste/{category}/{name}', timeout=40) as response:
                    content = response.read(20*1024*1024+1)
                if not 0 < len(content) <= 20*1024*1024:
                    raise RuntimeError('Unexpected image size')
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix('.jpg.part')
                partial.write_bytes(content)
                partial.replace(target)
                break
            except (OSError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(2*(attempt+1))
    return category, target


def main():
    plan = json.loads((ROOT/'outputs/external_realwaste_plan.json').read_text(encoding='utf-8'))
    if plan['author_commit'] != COMMIT or plan['source_archive_kind'] != 'author_five_material_subset':
        raise RuntimeError('Acquisition does not match frozen plan')
    started = time.perf_counter()
    tasks = [(category, index) for category, count in COUNTS.items() for index in range(1, count+1)]
    paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch, item) for item in tasks]
        for future in concurrent.futures.as_completed(futures):
            paths.append(future.result())
            if len(paths) % 100 == 0:
                print(f'Downloaded {len(paths)}/{len(tasks)} author images, {time.perf_counter()-started:.1f}s', flush=True)
    # Locally assembled, explicitly not the byte-identical UCI ZIP.
    temporary = ARCHIVE.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as zipped:
        for category, path in sorted(paths):
            zipped.write(path, f'realwaste-main/RealWaste/{category}/{path.name}')
    temporary.replace(ARCHIVE)
    receipt = dict(author_commit=COMMIT, raw_base=BASE, image_count=len(paths),
                   local_archive=str(ARCHIVE.relative_to(ROOT)), locally_assembled_subset=True,
                   elapsed_seconds=time.perf_counter()-started)
    (ROOT/'sources/realwaste_download_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
