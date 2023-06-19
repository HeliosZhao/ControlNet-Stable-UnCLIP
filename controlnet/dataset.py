import os
import random

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms

import json
import cv2
from PIL import Image


from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import ipdb


class ControlNetDepthUnCLIPDataset(Dataset):
    '''
    ControlNet dataset for coco
    data_root: root for data
    condition_dir: dir for condition image
    image_dir: dir for image
    caption_file: file name for captioning
    tokenizer: tokenizer model
    preprocessor: clip preprocessor

    '''

    def __init__(self, cfg, tokenizer, preprocessor=None):
        # self.data = []
        self.data_root = cfg.data_root
        self.image_root = os.path.join(self.data_root, cfg.image_dir)
        self.condition_root = os.path.join(self.data_root, cfg.condition_dir)
        
        with open(os.path.join(self.data_root, cfg.file_list), 'r') as f:
            self.data = f.read().splitlines() ## xxx.jpg


        prompt_path = os.path.join(self.data_root, cfg.caption_file)

        self.prompts = {}
        with open(prompt_path, 'rt') as f:
            for line in f:
                self.prompts.update(json.loads(line)) ## each line is dict{filename: prompt}
        
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.resolution = cfg.resolution

    def tokenize_captions(self, caption, is_train=True):
        captions = []
        # ipdb.set_trace()
        if random.random() < self.cfg.proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption should contain either strings or lists of strings."
            )

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        prompt = self.prompts[fname[:-4]] # no ext

        source_filename = os.path.join(self.condition_root, fname)
        target_filename = os.path.join(self.image_root, fname)

        source = Image.open(source_filename) #.convert("RGB") # condition
        target = Image.open(target_filename).convert("RGB") # image

        max_ratio = 1.25
        new_size = random.randint(self.resolution, int(self.resolution * max_ratio)-1) 
        ## resize the shortest side
        source = T.Resize(new_size, interpolation=T.InterpolationMode.BILINEAR)(source) ## T.Resize already includes the resize of shortest edge
        target = T.Resize(new_size, interpolation=T.InterpolationMode.BILINEAR)(target)

        aug_params = T.RandomCrop.get_params(source, output_size=(self.resolution, self.resolution))
        source = T.functional.crop(source, *aug_params)
        target = T.functional.crop(target, *aug_params)
        
        source_tensor = T.ToTensor()(source) ## condition 0-1, ToTensor will convert to C,H,W
        target_tensor = T.ToTensor()(target) * 2 - 1 # image -1,1

        ## CLIP image preprocess
        if self.preprocessor is not None:
            images_preprocessed = self.preprocessor(images=target, return_tensors="pt").pixel_values  # 1,3,H,W tensor
            condition_images_preprocessed = self.preprocessor(images=source, return_tensors="pt").pixel_values  # 1,3,H,W tensor
        else: ## this is for not unclip
            images_preprocessed = None
            condition_images_preprocessed = None

        examples = {}
        examples["input_ids"] = self.tokenize_captions(prompt) # 1,77

        examples["pixel_values"] = target_tensor
        examples["conditioning_pixel_values"] = source_tensor

        examples["images_preprocessed"] = images_preprocessed
        examples["condition_images_preprocessed"] = condition_images_preprocessed

        return examples



class ControlNetPoseUnCLIPDataset(Dataset):
    '''
    ControlNet dataset for coco pose
    data_root: root for data
    condition_dir: dir for condition image
    image_dir: dir for image
    caption_file: file name for captioning
    tokenizer: tokenizer model
    preprocessor: clip preprocessor

    '''

    def __init__(self, cfg, tokenizer, preprocessor=None):
        # self.data = []
        self.data_root = cfg.data_root
        self.image_root = os.path.join(self.data_root, cfg.image_dir)
        self.condition_root = os.path.join(self.data_root, cfg.condition_dir)
        
        with open(os.path.join(self.data_root, cfg.file_list), 'r') as f:
            self.data = f.read().splitlines() ## xxx.jpg


        prompt_path = os.path.join(self.data_root, cfg.caption_file)

        self.prompts = {}
        with open(prompt_path, 'rt') as f:
            for line in f:
                self.prompts.update(json.loads(line)) ## each line is dict{filename: prompt}
        
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.resolution = cfg.resolution

        from pycocotools.coco import COCO
        self.coco = COCO(os.path.join(self.data_root, cfg.anno_path))
        self.max_ratio = 1.25
        self.random_crop_prob = cfg.get('random_crop_prob', 0.0)
    
    def coco_crop_info(self, img_id, source):
        '''
        For object detection annotations, the format is "bbox" : [x,y,width,height]
        Where:
        x, y: the upper-left coordinates of the bounding box
        width, height: the dimensions of your bounding box
        '''

        img_meta = self.coco.imgs[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_file_name = img_meta['file_name']
        img_w = img_meta['width']
        img_h = img_meta['height']
        url = img_meta['coco_url']

        if random.random() < self.random_crop_prob:
            ## Random Crop from a single image
            short_side = int(min(img_w, img_h))
            crop_size = random.randint(int(short_side/self.max_ratio), short_side-1)
            aug_params = T.RandomCrop.get_params(source, output_size=(crop_size, crop_size))
            return aug_params

        while len(anns):
            ann = random.choice(anns)
            anns.remove(ann)
            if ann['num_keypoints'] > 0:
                x, y, width, height = ann['bbox']
                original_shape = (int(y), int(x), int(height), int(width))
                long_side = int(max(width, height))
                center_x = int(x + width / 2)
                center_y = int(y + height / 2)
                new_size = random.randint(long_side, int(long_side * self.max_ratio)-1)

                x0 = center_x - new_size // 2
                x1 = center_x + new_size // 2
                y0 = center_y - new_size // 2
                y1 = center_y + new_size // 2

                if (x0 < 0) or (y0 < 0) or (x1 > img_w) or (y1 > img_h):
                    return original_shape
                else:
                    return (y0, x0, new_size, new_size)


    def tokenize_captions(self, caption, is_train=True):
        captions = []
        # ipdb.set_trace()
        if random.random() < self.cfg.proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption should contain either strings or lists of strings."
            )

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        prompt = self.prompts[fname[:-4]] # no ext

        source_filename = os.path.join(self.condition_root, fname)
        target_filename = os.path.join(self.image_root, fname)

        source = Image.open(source_filename) #.convert("RGB") # condition
        target = Image.open(target_filename).convert("RGB") # image

        img_id = int(fname[:-4])
        crop_params = self.coco_crop_info(img_id, source)
        # ipdb.set_trace()
        source = T.functional.crop(source, *crop_params)
        target = T.functional.crop(target, *crop_params)

        
        if len(np.unique(np.asarray(source))) == 1: ## no pose here
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        source = source.resize((self.resolution, self.resolution))
        target = target.resize((self.resolution, self.resolution))
        # source = T.Resize(self.resolution, interpolation=T.InterpolationMode.BILINEAR)(source) ## T.Resize already includes the resize of shortest edge
        # target = T.Resize(self.resolution, interpolation=T.InterpolationMode.BILINEAR)(target)

        source_tensor = T.ToTensor()(source) ## condition 0-1, ToTensor will convert to C,H,W
        target_tensor = T.ToTensor()(target) * 2 - 1 # image -1,1

        ## CLIP image preprocess
        if self.preprocessor is not None:
            images_preprocessed = self.preprocessor(images=target, return_tensors="pt").pixel_values  # 1,3,H,W tensor
            condition_images_preprocessed = self.preprocessor(images=source, return_tensors="pt").pixel_values  # 1,3,H,W tensor
        else: ## this is for not unclip
            images_preprocessed = None
            condition_images_preprocessed = None

        examples = {}
        examples["input_ids"] = self.tokenize_captions(prompt) # 1,77

        examples["pixel_values"] = target_tensor
        examples["conditioning_pixel_values"] = source_tensor

        examples["images_preprocessed"] = images_preprocessed
        examples["condition_images_preprocessed"] = condition_images_preprocessed

        return examples





def make_train_dataset(cfg, logger, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if cfg.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config_name,
            cache_dir=cfg.cache_dir,
        )
    else:
        data_files = {}
        if cfg.train_data_dir is not None:
            data_files["train"] = os.path.join(cfg.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=cfg.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if cfg.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = cfg.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{cfg.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if cfg.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = cfg.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{cfg.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if cfg.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {caption_column}")
    else:
        conditioning_image_column = cfg.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{cfg.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < cfg.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if cfg.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }




def make_train_dataset_embed(cfg, logger, tokenizer, accelerator, preprocessor):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if cfg.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config_name,
            cache_dir=cfg.cache_dir,
        )
    else:
        data_files = {}
        if cfg.train_data_dir is not None:
            data_files["train"] = os.path.join(cfg.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=cfg.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if cfg.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = cfg.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{cfg.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if cfg.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = cfg.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{cfg.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if cfg.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {caption_column}")
    else:
        conditioning_image_column = cfg.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{cfg.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < cfg.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        ## CLIP image preprocess
        images_preprocessed = preprocessor(images=examples[image_column], return_tensors="pt").pixel_values  # B,3,H,W tensor
        condition_images_preprocessed = preprocessor(images=examples[conditioning_image_column], return_tensors="pt").pixel_values  # B,3,H,W tensor

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        examples["images_preprocessed"] = images_preprocessed
        examples["condition_images_preprocessed"] = condition_images_preprocessed

        return examples

    with accelerator.main_process_first():
        if cfg.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn_embed(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples]) ## each on 3,768,768, range -1 ~ 1
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples]) ## each on 3,768,768, range 0 ~ 1
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat([example["input_ids"] for example in examples], dim=0)

    if examples[0]["images_preprocessed"] is not None:
        images_preprocessed = torch.cat([example["images_preprocessed"] for example in examples], dim=0)
    else:
        images_preprocessed = None
    if examples[0]["condition_images_preprocessed"] is not None:
        condition_images_preprocessed = torch.cat([example["condition_images_preprocessed"] for example in examples], dim=0)
    else:
        condition_images_preprocessed = None

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "images_preprocessed": images_preprocessed, 
        "condition_images_preprocessed": condition_images_preprocessed, 
    }


