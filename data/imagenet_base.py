import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImagePaths(Dataset):
    def __init__(self, split, is_val, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if split == "train" and not is_val:
            transforms_ = [
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        else:
            transforms_ = [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        self.transforms = transforms.Compose(transforms_)

        # if self.size is not None and self.size > 0:
        #     self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        #     if not self.random_crop:
        #         self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
        #     else:
        #         self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
        #     self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        # else:
        #     self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # image = np.array(image).astype(np.uint8)
        # we replace the original taming version image preprocess 
        # with the one in RQVAE
        # image = self.preprocessor(image=image)["image"]
        # image = (image/127.5 - 1.0).astype(np.float32)

        image = self.transforms(image)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example