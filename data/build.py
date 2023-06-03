from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from utils.utils import instantiate_from_config

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, train_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.train_val = train_val
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

        # move setup here to avoid warning
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        if self.train_val:
            if "train" in self.datasets.keys() and "validation" in self.datasets.keys():
                self.datasets["train"] = self.datasets["train"] + self.datasets["validation"]
        for k in self.datasets.keys():
            print("dataset: ", k, len(self.datasets[k]))

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            print("instantiate from: ", data_cfg)
            instantiate_from_config(data_cfg)

    # def setup(self, stage=None):
    #     self.datasets = dict(
    #         (k, instantiate_from_config(self.dataset_configs[k]))
    #         for k in self.dataset_configs)
    #     if self.wrap:
    #         for k in self.datasets:
    #             self.datasets[k] = WrappedDataset(self.datasets[k])
    #     if self.train_val:
    #         if "train" in self.datasets.keys() and "validation" in self.datasets.keys():
    #             self.datasets["train"] = self.datasets["train"] + self.datasets["validation"]
    #     for k in self.datasets.keys():
    #         print("dataset: ", k, len(self.datasets[k]))

    def _train_dataloader(self):
        if hasattr(self.datasets["train"], "collate_fn"):
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          collate_fn=self.datasets["train"].collate_fn)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)


    def _val_dataloader(self):
        if hasattr(self.datasets['validation'], "collate_fn"):
            return DataLoader(self.datasets["validation"], batch_size=self.batch_size,
                            num_workers=self.num_workers, collate_fn=self.datasets["validation"].collate_fn)
        else:
            return DataLoader(self.datasets["validation"], batch_size=self.batch_size,
                            num_workers=self.num_workers)

    def _test_dataloader(self):
        if hasattr(self.datasets["test"], "collate_fn"):
            return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                            num_workers=self.num_workers, collate_fn=self.datasets["test"].collate_fn)
        else:
            return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                            num_workers=self.num_workers)