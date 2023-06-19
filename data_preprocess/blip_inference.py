from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
import numpy as np
import cv2
import imageio
import os
import ipdb
import json
from glob import glob
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16
)
model.to(device)
apply_midas = MidasDetector()

data_dir = '/home/yyzhao/diffusion/coco_text'
split = 'train2017'
# save_dir = 'test_annotator'
save_dir_depth = os.path.join(data_dir, split + "_depth")
os.makedirs(save_dir_depth, exist_ok=True)
# ipdb.set_trace()
files = sorted(glob(os.path.join(data_dir, split, "*.jpg")))
with open(os.path.join(data_dir, split + "_blip2text.json"), 'w') as outfile:
    for i, f_path in enumerate(tqdm(files)):
        # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        image = Image.open(f_path).convert("RGB")
        fname = f_path.split('/')[-1][:-4]
        # image.save(os.path.join(save_dir, "image-"+fname+'.jpg'))

        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16) ## inputs.pixel_values 1,3,224,224

        generated_ids = model.generate(**inputs) # generated_ids 1,8
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print(generated_text)
        ## NOTE write the text
        json.dump({fname: generated_text,}, outfile)
        outfile.write('\n')

        ## depth estimation
        input_image = np.asarray(image, dtype=np.uint8) # H,W,3 np.asarray get the image is H,W,3
        detected_map, _ = apply_midas(resize_image(input_image, 384))
        detected_map = HWC3(detected_map) # H',W',3
        width, height = image.size
        detected_map = cv2.resize(detected_map, (width, height), interpolation=cv2.INTER_LINEAR) # H,W,3
        imageio.imwrite(os.path.join(save_dir_depth, fname+'.jpg'), detected_map)
        # if i > 15:
        #     break


