import io
import os
import pickle
import string
from pathlib import Path

import lmdb
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import os, sys
sys.path.append(os.getcwd())
from data.default import DefaultDataPath

class FFHQ_LMDB(torchvision.datasets.VisionDataset):

    def __init__(self, split="train", resolution=256, is_eval=False, **kwargs):
        
        if split == "train":
            lmdb_path = DefaultDataPath.FFHQ.train_lmdb
        elif split == "val":
            lmdb_path = DefaultDataPath.FFHQ.val_lmdb
        else:
            raise ValueError()

        root = str(Path(lmdb_path))
        super().__init__(root, **kwargs)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        cache_file = os.path.join(root, cache_file)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))
            
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

    def __getitem__(self, index: int):
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return {
            "image": img
        }

    def __len__(self):
        return self.length

        
if __name__ == "__main__":
    dataset = FFHQ_LMDB(split='train', resolution=256, is_eval=False)
    dataset_val = FFHQ_LMDB(split='val', resolution=256, is_eval=False)
    
    print(len(dataset))
    print(len(dataset_val))
    
    # sample = dataset.__getitem__(0)
    
    # torchvision.utils.save_image(sample["image"], "sample_ffhq.png", normalize=True)
