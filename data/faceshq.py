# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
sys.path.append(os.getcwd())
from data.default import DefaultDataPath
from data.data_utils import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from data.default import DefaultDataPath

from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import numpy as np
import albumentations
import glob
from torch.utils.data import Dataset

class ImageFolder(torchvision.datasets.VisionDataset):

    def __init__(self, root, train_list_file, val_list_file, 
                 split='train', resolution=256, is_eval=False, **kwargs):

        root = Path(root)
        super().__init__(root, **kwargs)

        self.train_list_file = train_list_file
        self.val_list_file = val_list_file

        self.split = self._verify_split(split)

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        if self.split == 'trainval':
            fname_list = os.listdir(self.root)
            samples = [self.root.joinpath(fname) for fname in fname_list
                       if fname.lower().endswith(self.extensions)]
        else:
            listfile = self.train_list_file if self.split == 'train' else self.val_list_file
            with open(listfile, 'r') as f:
                samples = [self.root.joinpath(line.strip()) for line in f.readlines()]

        self.samples = samples
        
        if split == "train" and not is_eval:
            transforms_ = [
                transforms.RandomResizedCrop(resolution, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        else:
            transforms_ = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        self.transforms = transforms.Compose(transforms_)

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val', 'trainval'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, with_transform=True):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None and with_transform:
            sample = self.transforms(sample)
        return {
            "image": sample
        }

class FFHQ(ImageFolder):
    root = DefaultDataPath.FFHQ.root
    train_list_file = os.path.join(root, "assets/ffhqtrain.txt")
    val_list_file = os.path.join(root, "assets/ffhqvalidation.txt")

    def __init__(self, split='train', resolution=256, is_eval=False, **kwargs):
        super().__init__(FFHQ.root, FFHQ.train_list_file, FFHQ.val_list_file, split, resolution, is_eval, **kwargs)

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class CelebAHQTrain(FacesBase):
    def __init__(self, size):
        super().__init__()
        glob_pattern = os.path.join(DefaultDataPath.CelebAHQ.root, 'train/images', '*.jpg')
        paths = sorted(glob.glob(glob_pattern))
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = None
        
        transforms_ = [torchvision.transforms.ToTensor(),]
        self.transforms = torchvision.transforms.Compose(transforms_)
    
    def __getitem__(self, i):
        example = self.data[i]
        example["image"] = self.transforms(example["image"])
        return example

class CelebAHQValidation(FacesBase):
    def __init__(self, size):
        super().__init__()
        glob_pattern = os.path.join(DefaultDataPath.CelebAHQ.root, 'test/images', '*.jpg')
        paths = sorted(glob.glob(glob_pattern))
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = None
        
        transforms_ = [torchvision.transforms.ToTensor(),]
        self.transforms = torchvision.transforms.Compose(transforms_)
    
    def __getitem__(self, i):
        example = self.data[i]
        example["image"] = self.transforms(example["image"])
        return example


class FacesHQTrain(Dataset):
    def __init__(self, size, is_eval=False):
        d1 = CelebAHQTrain(size=size)
        d2 = FFHQ(split='train', resolution=size, is_eval=is_eval)
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i][0]["image"]
        return {"image": image}

class FacesHQValidation(Dataset):
    def __init__(self, size, is_eval=False):
        d1 = CelebAHQValidation(size=size)
        d2 = FFHQ(split="val", resolution=size, is_eval=is_eval)
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i][0]["image"]
        return {"image": image}
    
class FacesHQ(Dataset):
    def __init__(self, size, is_eval=False):
        d1 = CelebAHQTrain(size=size)
        d2 = FFHQ(split='train', resolution=size, is_eval=is_eval)
        d3 = CelebAHQValidation(size=size)
        d4 = FFHQ(split="val", resolution=size, is_eval=is_eval)
        
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i][0]["image"]
        return {"image": image}
    
if __name__ == "__main__":
    dataset = FFHQ(split='train', resolution=256, is_eval=False)
    dataset_val = FFHQ(split='val', resolution=256, is_eval=False)
    
    # celebahq = CelebAHQTrain(size=256)
    # celebahq_val = CelebAHQValidation(size=256)
    # out = celebahq.__getitem__(0)
    
    print(len(dataset))
    print(len(dataset_val))
    # print(len(celebahq))
    # print(len(celebahq_val))
    
    # facehq = FacesHQTrain(size=256)
    # facehq_val = FacesHQValidation(size=256)
    # facehq_all = FacesHQ(size=256)
    # out = facehq.__getitem__(0)
    # torchvision.utils.save_image(out["image"], "facehq.png", normalize=True)
    
    # print(len(facehq), len(facehq_val), len(facehq_all))