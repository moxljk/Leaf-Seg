import torch
from collections.abc import Iterable
import os, glob
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
from . import functional as F

class TrainDataset(BaseDataset):
    """
    A dataset loader, for datasets whose folder structure is:\n
        folder
        ├── raw
        │   ├── aaaa.png
        │   ├── bbbb.png
        │   ├── cccc.png
        │   └── ...
        └── tag
            ├── aaaa.png
            ├── bbbb.png
            ├── cccc.png
            └── ...
    """
    def __init__(self, dataset_paths, threshold=0.5, img_preprocess=None, mask_preprocess=None):
        super().__init__()
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        self.mask_paths = []
        self.img_paths = []
        for dataset_path in dataset_paths:
            img_paths = glob.glob(os.path.join(dataset_path,'raw','*.png'))
            for img_path in img_paths:
                 self.mask_paths.append(os.path.join(dataset_path,'tag',os.path.basename(img_path)))
            self.img_paths += img_paths

        self.threshold = threshold
        self.img_preprocess = img_preprocess
        self.mask_preprocess = mask_preprocess

    def __getitem__(self, index):
        img_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 480), antialias=True),
        ])
        img_path = self.img_paths[index]
        img = img_preprocess(Image.open(img_path).convert("RGB"))
        if self.img_preprocess is not None:
            img = self.img_preprocess(img)
 
        mask_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 480), antialias=True),
            F.Threshold(self.threshold)
        ])
        mask_path = self.mask_paths[index]
        mask = mask_preprocess(Image.open(mask_path).convert("L"))
        if self.mask_preprocess is not None:
            mask = self.mask_preprocess(mask)

        return img, mask

    def __len__(self):
        return len(self.img_paths)


class TestDataset(BaseDataset):
    def __init__(self, dataset_path, is_mask=False, threshold=0.5, preprocess=None) -> None:
        super().__init__()
        self.img_paths = glob.glob(os.path.join(dataset_path, '*.*'))
        self.img_paths.sort(key=os.path.basename)
        self.names = [os.path.basename(path) for path in self.img_paths]
        self.is_mask = is_mask
        self.threshold = threshold
        self.preprocess = preprocess
    
    def __getitem__(self, index):

        img_path = self.img_paths[index]
        
        if not self.is_mask:
            img = Image.open(img_path).convert("RGB")
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((320, 480), antialias=True),
            ])
        else:
            img = Image.open(img_path).convert("L")
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((320, 480), antialias=True),
                F.Threshold(self.threshold)
            ])

        ret = preprocess(img)
        if self.preprocess is not None:
            ret = self.preprocess(ret)

        return ret
        
    def __len__(self):
        return len(self.img_paths)

class MaskSet:
    def __init__(self, dir, device='cuda', threshold=0.5) -> None:
        self.device = device
        self.mask_paths = glob.glob(os.path.join(dir, '*'))
        self.mask_paths.sort(key=os.path.basename)
        self.mask_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 480), antialias=True),
            F.Threshold(threshold)
        ])
    
    def __iter__(self):
        self.__ind = 0
        return self

    def __next__(self):
        try:
            mask_path = self.mask_paths[self.__ind]
        except IndexError:
            raise StopIteration
        self.__ind += 1
        return self.mask_preprocess(Image.open(mask_path).convert("L")).to(self.device)
    
class IoUTester:
    def __init__(self, threshold) -> None:
        self.IoU = F.IoU(threshold)
    
    def test(self, prediction, groud_truth):
        return torch.stack([self.IoU(pr, gt) for pr, gt in zip(prediction, groud_truth)]).cpu().numpy()