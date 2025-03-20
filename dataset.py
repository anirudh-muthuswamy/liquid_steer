import pandas as pd
import os
import torch
import torch.backends.mps as mps
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from check_data import get_preprocessed_data_pd

mps.benchmark = True
device = torch.device('mps')

def df_split_train_test(df_filtered, train_dataset_path = 'train_data_filtered.csv', 
                        test_dataset_path = 'test_data_filtered.csv',
                        train_size = 0.8):
    train_dataset = df_filtered[:int(train_size * len(df_filtered))]
    test_dataset = df_filtered[int(train_size * len(df_filtered)):]
    print('Train dataset length:', len(train_dataset))
    print('Test dataset length:', len(test_dataset))

    train_dataset.to_csv(train_dataset_path, index=False)
    test_dataset.to_csv(test_dataset_path, index=False)

    return train_dataset_path, test_dataset_path

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
    def __init__(self, csv_file, seq_len, transform=None):
        """
        Dataset that extracts continuous sequences from the given DataFrame.

        :param df: Filtered DataFrame with columns ['filepath', 'steering_angle', 'sequence_id']
        :param seq_len: Length of each sequence
        :param transform: Transformations for image preprocessing
        """
        self.df = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.transform = transform

        # Group by sequence_id and collect valid sequences
        self.sequences = []
        num_sequences_total = 0
        for seq_id in self.df["sequence_id"].unique():
            seq_data = self.df[self.df["sequence_id"] == seq_id]
            num_sequences = max(len(seq_data) - seq_len + 1, 0)
            num_sequences_total += num_sequences
            for i in range(len(seq_data) - seq_len + 1):  # Only full sequences
                self.sequences.append(seq_data.iloc[i : i + seq_len])

        print(f"Total sequences extracted: {num_sequences_total}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_batch = self.sequences[idx]  # Get one full sequence
        # Extract filepaths and steering angles
        img_names = seq_batch['filepath'].tolist()
        angles = seq_batch['steering_angle'].tolist()

        # Read and process images
        images = [cv2.imread(img_name) for img_name in img_names]
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

        if self.transform:
            images = torch.stack([self.transform(img) for img in images])  # Apply transforms
        angles = torch.tensor(angles, dtype = torch.float32)
        
        images = images.to(device = device)
        angles = angles.to(device = device)

        return images, angles
    
def create_train_test_dataset(train_csv_file, 
                              test_csv_file,
                              seq_len = 64, 
                              imgw = 224,
                              imgh = 224,
                              mean = [0.543146, 0.53002986, 0.50673143],
                              std = [0.23295668, 0.22123158, 0.22100357]):
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((imgh, imgw)),  # Resize the image to the desired size
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = CustomDataset(csv_file=train_csv_file, seq_len=seq_len, transform=transform)
    test_dataset = CustomDataset(csv_file=test_csv_file, seq_len=seq_len, transform=transform)

    return train_dataset, test_dataset

def create_train_test_loader(train_dataset, test_dataset, batch_size=8, shuffle=False):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    for inputs, labels in train_loader:
        print("Batch input shape:", inputs.shape)
        print("Batch label shape:", labels.shape)
        break 

    return train_loader, test_loader

def get_default_loaders_for_training(data_dir, steering_angles_path):
    data_preprocessed_pd = get_preprocessed_data_pd(data_dir, steering_angles_path)

    train_dataset_path, test_dataset_path = df_split_train_test(data_preprocessed_pd)
    train_dataset , test_dataset = create_train_test_dataset(train_csv_file = train_dataset_path,
                                                             test_csv_file = test_dataset_path)
    train_loader, test_loader = create_train_test_loader(train_dataset, test_dataset)

    return train_loader, test_loader

if __name__ == '__main__':

    data_dir = 'sullychen/07012018/data'
    steering_angles_txt_path = 'sullychen/07012018/data.txt'

    get_default_loaders_for_training(data_dir, steering_angles_txt_path)