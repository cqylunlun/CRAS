from torchvision import transforms
from enum import Enum

import PIL
import torch
import os

_CLASSNAMES = [
    "carpet",
    "grid",
    "tile",
    "wood",
    "leather",
    "screw",
    "pill",
    "capsule",
    "zipper",
    "cable",
    "toothbrush",
    "transistor",
    "metal_nut",
    "bottle",
    "hazelnut",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            classname='leather',
            resize=329,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split != DatasetSplit.TRAIN and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        set_name = self.split.value if self.split != DatasetSplit.VAL else "test"
        for classname in self.classname:
            classpath = os.path.join(self.source, classname, set_name)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

                if self.split != DatasetSplit.TRAIN and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
                else:
                    maskpaths_per_class[classname]["good"] = None

        data_to_iterate = []
        for classname in self.classname:
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split != DatasetSplit.TRAIN and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
