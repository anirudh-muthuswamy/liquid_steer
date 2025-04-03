import pandas as pd
import os
import torch
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


def create_csv(split='train', base_dir='idd20kII', save_dir='code_files'):

    all_files = glob.glob(os.path.join(base_dir, '**', split, '**', '*'), recursive=True)

    # Filter out directories and only keep files
    all_files = [f for f in all_files if os.path.isfile(f) and (f.endswith(".png") or f.endswith(".jpg"))]

    gtfine_files = [f for f in all_files if 'gtFine' in f and f.endswith("_newlevel3Id.png")]
    leftImg8bit_files = [f for f in all_files if 'leftImg8bit' in f]

    gtfine_dict = {os.path.basename(f).split('_')[0]: f for f in gtfine_files}  # Extract identifier before '_'
    leftImg8bit_dict = {os.path.basename(f).split('_')[0]: f for f in leftImg8bit_files}

    # Match files based on their identifiers
    matched_data = []
    for identifier, mask_path in gtfine_dict.items():
        if identifier in leftImg8bit_dict:
            img_path = leftImg8bit_dict[identifier]
            matched_data.append({"mask": mask_path, "image": img_path})

    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    save_path = os.path.join(save_dir, f"{split}_IDD.csv")
    pd.DataFrame(matched_data).to_csv(save_path, index=False)

    print(f"CSV created at: {save_path}")

    return pd.DataFrame(matched_data)

def reverse_augmentation(image, mask):
    # Imagenet mean and std for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Undo normalization
    image = image * std[:, None, None] + mean[:, None, None]
    
    # Clip values to [0, 1] (in case normalization introduces out-of-bounds values)
    image = torch.clamp(image, 0, 1)
    
    # Convert to NumPy array
    image_numpy = image.permute(1, 2, 0).numpy()  # C x H x W -> H x W x C for Matplotlib
    image_numpy = (image_numpy * 255).astype(np.uint8)  # Scale to [0, 255] for visualization
    
    mask_numpy = mask.permute(1,2,0).cpu().numpy()

    return image_numpy, mask_numpy

class IDDDataset(Dataset):
     
    def __init__(
            self, 
            df,
            aug, 
            select_class=[0], #list of classes to select (default road which is class 0)

    ):
        self.df = df
        self.aug = aug
        self.image_paths = sorted(self.df["image"].tolist())
        self.mask_paths = sorted(self.df["mask"].tolist())
 
        self.select_class = select_class

    def _preprocess_img(self):
        return T.Compose([
            T.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_augmentation_random_flip(self):
        return T.Compose([
            T.Resize((480, 640), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.RandomHorizontalFlip(p=1),
            T.Lambda(lambda x: x.float() / 255.0)
        ])
    
    def _get_augmentation(self):
        return T.Compose([
            T.Resize((480, 640), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.Lambda(lambda x: x.float() / 255.0)
        ])

    def __getitem__(self, i):
         
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY) 
        # Convert images and masks to tensors
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # Shape: (C, H, W)

        filtered_mask = np.zeros_like(mask)
        for class_id in self.select_class:
            filtered_mask[mask == class_id] = 255
        mask = filtered_mask

        mask = np.expand_dims(mask, axis=0)  # Add channel dimension for masks
        mask = torch.tensor(mask, dtype=torch.long)# Shape: (1, H, W)

        # apply augmentations
        if self.aug:
            random_flip = random.choices([True, False], weights=[0.75, 0.25])[0]
            if random_flip:
                image = self._get_augmentation_random_flip()(image)
                mask = self._get_augmentation_random_flip()(mask)
            else:
                image = self._get_augmentation()(image)
                mask = self._get_augmentation()(mask)
        else:
            image = self._get_augmentation()(image)
            mask = self._get_augmentation()(mask)

        image = self._preprocess_img()(image)

        return image, mask.round().long()
         
    def __len__(self):
        # return length of 
        return len(self.image_paths)
    
if __name__ == '__main__':

    train_df = create_csv(split='train')
    val_df = create_csv(split='val')

    train_dataset = IDDDataset(train_df, select_class=[0], aug=True)
    val_dataset = IDDDataset(val_df, select_class=[0], aug=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, prefetch_factor=2, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, prefetch_factor=2, pin_memory=True, num_workers=4)

    for images, masks in train_loader:
        print(images.shape)
        print(masks.shape)
        break
    