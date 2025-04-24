import pandas as pd
import os
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ..utils import get_preprocessed_data_pd, df_split_train_val

class CustomDataset(Dataset):
    def __init__(self, csv_file, seq_len, imgh=224, imgw=224, step_size=1, crop=True, transform=None):

        self.df = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.imgh = imgh
        self.imgw = imgw
        self.step_size = step_size
        self.crop = crop
        self.transform = transform

        self.sequences = []  # will hold tuples (seq_df, next_angle)

        # group once by sequence_id
        for _, seq_data in self.df.groupby("sequence_id"):
            # we need at least seq_len + 1 frames to form one training example
            N = len(seq_data)
            for start in range(0, N - seq_len, step_size):
                window = seq_data.iloc[start : start + seq_len]
                next_angle = seq_data.iloc[start + seq_len]["steering_angle"]
                self.sequences.append((window.reset_index(drop=True), next_angle))

        print(f"Total examples: {len(self.sequences)} "
              f"(each is {seq_len} frames → 1 target)")

    def _crop_lower_half(self, img, keep_ratio=0.6):
        h = img.shape[0]
        return img[int(h*(1-keep_ratio)) :, :, :]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_df, next_angle = self.sequences[idx]

        imgs = []
        for fp in seq_df["filepath"]:
            img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            if self.crop:
                img = self._crop_lower_half(img)
            img = cv2.resize(img, (self.imgh, self.imgw))
            if self.transform:
                img = self.transform(img)  # C×H×W
            imgs.append(img)

        # (seq_len, C, H, W)
        x = torch.stack(imgs, dim=0)
        y = torch.tensor([next_angle], dtype=torch.float32)
        return x, y
    

def create_train_val_dataset(train_csv_file, val_csv_file, seq_len = 32, imgw = 224, imgh = 224,
                             step_size = 32, crop = True, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = CustomDataset(csv_file=train_csv_file, seq_len=seq_len, imgh = imgh, imgw=imgw,
                                  step_size=step_size,crop=crop, transform=transform)
    val_dataset = CustomDataset(csv_file=val_csv_file, seq_len=seq_len,imgh = imgh, imgw=imgw,
                                step_size=step_size,crop=crop, transform=transform)

    return train_dataset, val_dataset

def create_train_val_loader(train_dataset, val_dataset, train_sampler=None, val_sampler=None, batch_size=8,
                            num_workers=4, prefetch_factor=2, pin_memory=True, train_shuffle=False):

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, 
                              prefetch_factor=prefetch_factor,pin_memory=pin_memory, shuffle=train_shuffle)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers, 
                            prefetch_factor=prefetch_factor, pin_memory=pin_memory, shuffle=False)

    print('len of train loader:', len(train_loader))
    print('len of val loader', len(val_loader))

    for (inputs, labels) in train_loader:
        print("Batch input shape:", inputs.shape)
        print("Batch label shape:", labels.shape)
        break

    return train_loader, val_loader

def get_loaders_for_training(data_dir, steering_angles_path, step_size, seq_len, imgh, imgw, filter, turn_threshold, 
                                     buffer_before, buffer_after, crop=True, train_size=0.8, save_dir='data/csv_files', 
                                     norm=True, batch_size=16, num_workers=4, prefetch_factor=4, pin_memory=True, train_shuffle=False):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    #preprocesses data
    data_preprocessed_pd = get_preprocessed_data_pd(data_dir, steering_angles_path, filter, turn_threshold, 
                                     buffer_before, buffer_after, norm, save_dir)

    if filter:
        train_csv_filename = f"train_flt_ncp_tt_{turn_threshold}_bb_{buffer_before}_ba_{buffer_after}.csv"
        val_csv_filename = f"val_flt_ncp_tt_{turn_threshold}_bb_{buffer_before}_ba_{buffer_after}.csv"
    else:
        train_csv_filename = f"train_ncp_unfiltered.csv"
        val_csv_filename = f"val_ncp_unfiltered.csv"

    #splits data into train and val
    train_dataset_path, val_dataset_path = df_split_train_val(data_preprocessed_pd,
                                                              train_csv_filename=train_csv_filename,
                                                              val_csv_filename=val_csv_filename,
                                                              save_dir=save_dir,
                                                              train_size=train_size)
    #gets custom train and test pytorch dataset
    train_dataset , val_dataset = create_train_val_dataset(train_csv_file = train_dataset_path,
                                                             val_csv_file = val_dataset_path,
                                                             seq_len=seq_len, imgh=imgh, imgw=imgw,
                                                             step_size=step_size, crop=crop)
    
    #gets dataloaders from dataset
    train_loader, val_loader = create_train_val_loader(train_dataset, val_dataset,
                                                       batch_size=batch_size, 
                                                       num_workers=num_workers, 
                                                       prefetch_factor=prefetch_factor,
                                                       pin_memory=pin_memory, 
                                                       train_shuffle=train_shuffle)

    return train_loader, val_loader

if __name__ == '__main__':

    #preprocessing csv file args
    data_dir = 'data/sullychen/07012018/data'
    steering_angles_txt_path = 'data/sullychen/07012018/data.txt'
    save_dir = 'data/csv_files_experimental'
    filter = False
    norm = False

    turn_threshold = 0.08 
    buffer_before = 32 
    buffer_after = 32
    train_size = 0.8

    #custom pytorch dataset args
    imgh=224
    imgw=224
    step_size = 1
    seq_len = 3
    crop=True

    #dataloader args
    batch_size = 16
    prefetch_factor = 2
    num_workers=4
    pin_memory=True
    train_shuffle=False

    get_loaders_for_training(
        #preprocessing args:
        data_dir, steering_angles_path=steering_angles_txt_path, save_dir=save_dir, filter=filter, norm=norm,
        turn_threshold=turn_threshold, buffer_before=buffer_before, buffer_after=buffer_after, train_size=train_size,
        #dataset args:
        imgh=imgh, imgw=imgw, step_size=step_size, seq_len=seq_len, crop=crop, 
        #dataloader args:
        batch_size=batch_size, prefetch_factor=prefetch_factor, num_workers=num_workers, pin_memory=pin_memory,
        train_shuffle=train_shuffle)