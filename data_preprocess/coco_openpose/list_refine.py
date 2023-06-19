import os
import numpy as np
import imageio
import sys
from tqdm import tqdm

split = sys.argv[1]

data_dir = '/data/yyzhao/diffusion/coco_text'
condition = 'openposefull'
imagelist_path = os.path.join(data_dir, "person_list_{}.txt".format(split))
with open(imagelist_path, 'r') as f:
    image_list = f.read().splitlines()

save_dir_condition = os.path.join(data_dir, split + "_"+condition)
os.makedirs(save_dir_condition, exist_ok=True)
# ipdb.set_trace()

drop_list = []
for fn in tqdm(image_list):
    f_path = os.path.join(save_dir_condition, fn)
    image = imageio.imread(f_path)
    if len(np.unique(image)) == 1:
        ## NO detected pose
        drop_list.append(fn)

print('Required to drop {} images'.format(len(drop_list)))
remain_list = list( set(image_list) - set(drop_list) )
print('Remaining {} images'.format(len(remain_list)))
with open(imagelist_path.replace('.txt', '_new.txt'), 'w') as f:
    for _img in remain_list:
        f.write(_img + '\n')


