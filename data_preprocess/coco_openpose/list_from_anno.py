from pycocotools.coco import COCO
import sys
from tqdm import tqdm

def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        url = img_meta['coco_url']

        yield [img_id, img_file_name, w, h, url, anns]

split = sys.argv[1]
anno_path = "/data/yyzhao/diffusion/coco_text/annotations/person_keypoints_{}.json".format(split)
save_path = "/data/yyzhao/diffusion/coco_text/person_list_{}.txt".format(split)

coco_dataset = COCO(anno_path)
data_set_length = len(list(coco_dataset.imgs.keys()))

persons_data = []
with tqdm(total=data_set_length) as pbar:
    for img_id, img_fname, w, h, url, meta in get_meta(coco_dataset):
        for m in meta:
            if m['num_keypoints'] > 0:
                persons_data.append(img_fname)

        pbar.update(1)

print("All person in COCO {} is {}".format(split, len(persons_data)))
person_images = sorted(list(set(persons_data)))
print("Images with person in COCO {} is {}".format(split, len(person_images)))

with open(save_path, "w") as f:
    for _fn in person_images:
        f.write(_fn + '\n')

