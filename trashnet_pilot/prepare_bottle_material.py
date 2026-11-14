"""Prepare a small, source-batch-separated single-bottle evaluation from TACO."""
import hashlib
import io
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
from urllib.request import urlopen

from PIL import Image, ImageOps
from diagnose_domain_inputs import ROOT, digest, read, read_csv, save, write_csv
from prepare_data import phash


DATA = ROOT/'data/bottle_material'
DOWNLOADS = ROOT/'data/downloads'
ANNOTATION = DOWNLOADS/'taco_annotations.json'
ORIGINAL_LINKS = DOWNLOADS/'taco_flickr_annotations.json'
LINKS_REVISION = '6401db9ddd29af3d923304e3c09785707ed0174c'
CLASSES = {5:'plastic', 6:'glass'}
HOLDOUT_BATCHES = {'batch_2', 'batch_8', 'batch_11', 'batch_14'}


def select_bottles(annotations, original):
    image_map = {row['id']:row for row in annotations['images']}
    links = {row['id']:row for row in original['images']}
    old_objects = {row['id']:row for row in original['annotations']}
    by_image = defaultdict(list)
    for row in annotations['annotations']:
        if row['category_id'] in (4,5,6):
            by_image[row['image_id']].append(row)
    eligible = defaultdict(list)
    for image_id, objects in by_image.items():
        if len(objects) != 1:
            continue
        bottle = objects[0]
        if bottle['category_id'] not in CLASSES or bottle.get('iscrowd'):
            continue
        image, original_image = image_map[image_id], links[image_id]
        if any(image[key] != original_image[key] for key in ('file_name','width','height')):
            raise RuntimeError('Image metadata changed between author revisions')
        old_bottle = old_objects[bottle['id']]
        if any(old_bottle[key] != bottle[key] for key in ('image_id','category_id','bbox')):
            raise RuntimeError('Bottle annotation changed between author revisions')
        url = original_image.get('flickr_640_url') or original_image.get('coco_url')
        if not url:
            continue
        url = url.replace('http://','https://',1)
        hostname = urlsplit(url).hostname or ''
        if not hostname.endswith('.staticflickr.com') or urlsplit(url).scheme != 'https':
            continue
        batch = image['file_name'].split('/')[0]
        fold = 'holdout' if batch in HOLDOUT_BATCHES else 'development'
        label = CLASSES[bottle['category_id']]
        eligible[fold,label].append((image, bottle, url, batch, fold, label))
    selected = []
    for (fold,label), rows in sorted(eligible.items()):
        rows.sort(key=lambda row:hashlib.sha256(str(row[0]['id']).encode()).digest())
        selected.extend(rows[:75] if label=='plastic' else rows)
    return selected


def fetch(record):
    image, annotation, url, batch, fold, label = record
    image_id = image['id']
    try:
        with urlopen(url, timeout=14) as response:
            raw = response.read(3*1024*1024+1)
        if len(raw) > 3*1024*1024:
            raise ValueError('Image exceeds download bound')
        with Image.open(io.BytesIO(raw)) as source:
            picture = ImageOps.exif_transpose(source).convert('RGB')
        sx, sy = picture.width/image['width'], picture.height/image['height']
        if abs(sx/sy-1) > .02:
            raise ValueError('Thumbnail and annotation aspect ratios differ')
        x,y,w,h = annotation['bbox']
        if min(w*sx,h*sy) < 24:
            raise ValueError('Annotated bottle is smaller than 24 pixels')
        box = (max(0, int((x-.10*w)*sx)), max(0, int((y-.10*h)*sy)),
               min(picture.width, int((x+1.10*w)*sx+.9999)), min(picture.height, int((y+1.10*h)*sy+.9999)))
        cropped = picture.crop(box)
        return dict(image_id=image_id, annotation_id=annotation['id'], source_file=image['file_name'], source_batch=batch,
                    split=fold, class_name=label, source_category=annotation['category_id'], source_url=url,
                    source_sha256=hashlib.sha256(raw).hexdigest(), crop_box=list(box), image_phash=phash(picture),
                    crop_phash=phash(cropped), crop=cropped), None
    except (OSError, ValueError) as error:
        return None, dict(image_id=image_id, split=fold, class_name=label, source_url=url, reason=str(error))


def main():
    manifest = DATA/'manifest.csv'
    provenance = DATA/'provenance.json'
    if provenance.exists():
        if read(provenance)['manifest_sha256'] != digest(manifest):
            raise RuntimeError('Bottle manifest changed')
        for row in read_csv(manifest):
            if digest(ROOT/row['path']) != row['file_sha256']:
                raise RuntimeError('Bottle sample file changed')
        print(read(provenance)['counts'], flush=True)
        return
    DATA.mkdir(parents=True, exist_ok=True)
    selected = select_bottles(read(ANNOTATION), read(ORIGINAL_LINKS))
    print('Fetching',len(selected),'annotated bottle images from the original author links',flush=True)
    downloaded, rejected = [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index,(item,error) in enumerate(pool.map(fetch, selected),1):
            if item:
                downloaded.append(item)
            else:
                rejected.append(error)
            if index % 40 == 0:
                print(f'Processed {index}/{len(selected)}',flush=True)
    excluded = set()
    for pos,a in enumerate(downloaded):
        for b in downloaded[:pos]:
            if a['split'] == b['split']:
                continue
            if (a['source_sha256']==b['source_sha256'] or (a['image_phash']^b['image_phash']).bit_count()<=4
                    or (a['crop_phash']^b['crop_phash']).bit_count()<=4):
                excluded.update([a['image_id'],b['image_id']])
    records = []
    for item in downloaded:
        if item['image_id'] in excluded:
            continue
        path = DATA/f"bottle_{item['image_id']:04d}.png"
        item.pop('crop').save(path)
        records.append(dict(path=path.relative_to(ROOT).as_posix(),file_sha256=digest(path),**item))
    counts = {fold:dict(Counter(row['class_name'] for row in records if row['split']==fold)) for fold in ('development','holdout')}
    if any(set(value)!=set(CLASSES.values()) for value in counts.values()):
        raise RuntimeError('Bottle sample requires both classes in both folds')
    write_csv(manifest, records)
    save(provenance,dict(annotation_source=(DOWNLOADS/'taco_annotations_source.txt').read_text(encoding='utf-8-sig').strip(),
                         annotation_sha256=digest(ANNOTATION),original_link_revision=LINKS_REVISION,
                         original_link_metadata_sha256=digest(ORIGINAL_LINKS),manifest_sha256=digest(manifest),
                         holdout_source_batches=sorted(HOLDOUT_BATCHES),counts=counts,
                         crop='One annotated bottle per image. Box padded 10% on each side.',
                         duplicate_rule='Remove both sides of cross-fold exact source matches and pHash distance <=4 on source or crop.',
                         excluded_cross_fold_ids=sorted(excluded),unavailable_or_invalid=rejected,
                         grouping_limit='Source batches and image similarity are available. Physical item identity is unknown.'))
    print('Saved bottle sample:',counts,'rejected',len(rejected),'cross-fold removed',len(excluded),flush=True)


if __name__ == '__main__':
    main()
