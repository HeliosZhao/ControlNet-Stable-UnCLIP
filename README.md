# ControlNet training code for Stable UnCLIP

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

This example is based on the [training example in the original ControlNet repository](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md).

## Installing the dependencies

- Python>=3.9 and Pytorch>=1.13.1
- xformers 0.0.17
- Other packages in `requirements.txt`


## COCO2017 dataset

COCO2017 dataset with depth and openpose conditions is used for training. Download from the [website](https://cocodataset.org/#home).

#### Depth Condition

Generating captions with [BLIP2-OPT-2.7b](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and generating depth with Midas:
```sh
python data_preprocess/blip_inference.py
```

#### OpenPose Condition

First, obtain the image names with person inside from COCO annotation files `person_keypoints_{split}.json`. The file name list is stored in `person_list_{split}.txt`:
```
python data_preprocess/coco_openpose/list_from_anno.py train2017
```

Second, extract the open pose image condition. The conditions are saved in `{split}_openposefull`:
```
python data_preprocess/coco_openpose/condition_extraction.py train2017
```

Finally, refine the list, removing the images that cannot detected by openpose. The new file name list is stored in `person_list_{split}_new.txt`:
```
python data_preprocess/coco_openpose/list_refine.py train2017
```


## Training

Training the ControlNet for Stable Diffusion V2 UnCLIP model, conditioning on image embedding for the stable diffusion. 
- Depth Condition:
```bash
accelerate launch train_controlnet_unclip_depth.py --config configs/controlnet-coco-unclip-small-depth.yaml
```

- OpenPose Condition:
```bash
accelerate launch train_controlnet_unclip_pose.py --config configs/controlnet-coco-unclip-small-openposefull.yaml
```

## Citation
If you make use of our work, please cite ControlNet and our paper.
```bibtex
@article{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
  author={Lvmin Zhang and Maneesh Agrawala},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}

@article{zhao2023makeaprotagonist,
    title={Make-A-Protagonist: Generic Video Editing with An Ensemble of Experts},
    author={Zhao, Yuyang and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Lee, Gim Hee},
    journal={arXiv preprint arXiv:2305.08850},
    year={2023}
}
```
