from PIL import Image
import torch
import numpy as np
import cv2
import imageio
import os
import ipdb
import json
from glob import glob
from tqdm import tqdm
import sys
from controlnet_aux import MidasDetector, OpenposeDetector


device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = '/data/yyzhao/diffusion/coco_text'
# condition = 'openpose'
condition = 'openposefull'
hand_face = False
if condition == 'openposefull':
    hand_face = True

split = sys.argv[1]
imagelist_path = os.path.join(data_dir, "person_list_{}_new.txt".format(split))
with open(imagelist_path, 'r') as f:
    image_list = f.read().splitlines()

save_dir_condition = os.path.join(data_dir, split + "_"+condition)
os.makedirs(save_dir_condition, exist_ok=True)
# ipdb.set_trace()

model = OpenposeDetector.from_pretrained("lllyasviel/Annotators")#.to(device)


for fn in tqdm(image_list):
    f_path = os.path.join(data_dir, split, fn)
    image = Image.open(f_path).convert("RGB")
    processed_image = model(image, hand_and_face=hand_face)
    # ipdb.set_trace()
    processed_image = processed_image.resize(image.size)
    processed_image.save(os.path.join(save_dir_condition, fn))



