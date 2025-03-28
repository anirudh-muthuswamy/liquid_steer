import pandas as pd
import os
import torch
import numpy as np
import cv2
from collections import OrderedDict
from itertools import islice
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from check_data import get_preprocessed_data_pd

def df_split_train_val(df_filtered, train_dataset_path = 'train_data_filtered.csv', 
                        val_dataset_path = 'val_data_filtered.csv',
                        train_size = 0.8):
    train_dataset = df_filtered[:int(train_size * len(df_filtered))]
    val_dataset = df_filtered[int(train_size * len(df_filtered)):]
    print('Train dataset length:', len(train_dataset))
    print('Val dataset length:', len(val_dataset))

    train_dataset.to_csv(train_dataset_path, index=False)
    val_dataset.to_csv(val_dataset_path, index=False)

    return train_dataset_path, val_dataset_path

def calculate_mean_and_std(dataset_path):
    num_pixels = 0
    channel_sum = np.zeros(3)  # Assuming RGB images, change to (1,) for grayscale
    channel_sum_squared = np.zeros(3)  # Assuming RGB images, change to (1,) for grayscale

    for root, _, files in os.walk(dataset_path):
        for file in files:
            image_path = os.path.join(root, file)
            image = Image.open(image_path).convert('RGB')  # Convert to RGB if needed

            pixels = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
            num_pixels += pixels.size // 3  # Assuming RGB images, change to 1 for grayscale

            channel_sum += np.sum(pixels, axis=(0, 1))
            channel_sum_squared += np.sum(pixels ** 2, axis=(0, 1))

    mean = channel_sum / num_pixels
    std = np.sqrt((channel_sum_squared / num_pixels) - mean ** 2)
    return mean, std

class CustomDataset(Dataset):
    def __init__(self, csv_file, seq_len, imgh, imgw, transform=None):
        """
        Dataset that extracts continuous sequences from the given DataFrame.

        :param df: Filtered DataFrame with columns ['filepath', 'steering_angle', 'sequence_id']
        :param seq_len: Length of each sequence
        :param transform: Transformations for image preprocessing
        """
        self.df = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.transform = transform
        self.imgh = 224
        self.imgw = 224

        # Group by sequence_id and collect valid sequences
        self.sequences = OrderedDict({})
        num_sequences_total = 0
        # for each sequence id
        for seq_id in self.df["sequence_id"].unique():
            seq_data = self.df[self.df["sequence_id"] == seq_id]
            num_sequences = max(len(seq_data) - seq_len + 1, 0)
            num_sequences_total += num_sequences
            # for each sequence of len=seq_len for that sequence_id
            for i in range(len(seq_data) - seq_len + 1):  # Only full sequences
                self.sequences[(seq_id,i)] = (seq_data.iloc[i : i + seq_len])

        print(f"Total sequences extracted: {len(self.sequences)}")

    def get_ith_element(self, od, i):
        return next(islice(od.items(), i, None))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_batch = self.get_ith_element(self.sequences, idx)
        sequence_id = seq_batch[0][0]
        seq_num = seq_batch[0][1]
        seq_df = seq_batch[1]
        # Extract filepaths and steering angles
        img_names = seq_df['filepath'].tolist()
        angles = torch.tensor(seq_df['steering_angle'].tolist(), dtype=torch.float32)

        # Read and process images in one go with OpenCV
        images = []
        for img_name in img_names:
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.imgw, self.imgh ))  # Resize directly with OpenCV
            images.append(img)

        # Convert to tensor and normalize in batch
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])

        return sequence_id, seq_num, images, angles
    
def create_train_val_dataset(train_csv_file, 
                              val_csv_file,
                              seq_len = 64, 
                              imgw = 224,
                              imgh = 224,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = CustomDataset(csv_file=train_csv_file, seq_len=seq_len,imgh = imgh, imgw=imgw,
                                  transform=transform)
    val_dataset = CustomDataset(csv_file=val_csv_file, seq_len=seq_len,imgh = imgh, imgw=imgw,
                                transform=transform)

    return train_dataset, val_dataset

def create_train_val_loader(train_dataset, val_dataset, batch_size=8, shuffle=False):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, prefetch_factor=4, pin_memory=True, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, prefetch_factor=4, pin_memory=True, shuffle=shuffle)

    print('len of train loader:', len(train_loader))
    print('len of val loader', len(val_loader))

    for (_, _, inputs, labels) in train_loader:
        print("Batch input shape:", inputs.shape)
        print("Batch label shape:", labels.shape)
        break

    return train_loader, val_loader

def get_default_loaders_for_training(data_dir, steering_angles_path):
    data_preprocessed_pd = get_preprocessed_data_pd(data_dir, steering_angles_path)

    train_dataset_path, val_dataset_path = df_split_train_val(data_preprocessed_pd)
    train_dataset , val_dataset = create_train_val_dataset(train_csv_file = train_dataset_path,
                                                             val_csv_file = val_dataset_path)
    train_loader, val_loader = create_train_val_loader(train_dataset, val_dataset)

    return train_loader, val_loader

if __name__ == '__main__':

    data_dir = 'sullychen/07012018/data'
    steering_angles_txt_path = 'sullychen/07012018/data.txt'

    get_default_loaders_for_training(data_dir, steering_angles_txt_path)