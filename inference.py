import torch
import numpy as np
import torch.backends.mps as mps
from matplotlib import animation
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import NCPModel
from dataset import (df_split_train_val, create_train_val_dataset, create_train_val_loader,
                     calculate_mean_and_std)
from check_data import (get_full_image_filepaths, get_steering_angles,
                        convert_to_df, filter_df_on_turns, group_data_by_sequences)
from torch.utils.data import Sampler, DataLoader
from collections import OrderedDict


class UniqueSequenceSampler(Sampler):
    """
    Samples unique seq_id while ensuring only seq_nums in range [0, 64] are considered.
    """
    def __init__(self, dataset, seq_len=64):
        self.dataset = dataset

        # Extract unique seq_ids from OrderedDict keys
        unique_seq_ids = set(seq_id for seq_id, _ in dataset.sequences.keys())

        # Collect valid indices for the first sequence (seq_num = 0) for each seq_id if available
        self.indices = []
        for seq_id in unique_seq_ids:
            key = (seq_id, 0)
            if key in dataset.sequences:
                self.indices.append(dataset.index_map[key])

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def reverse_transform(tensor):
    """
    Reverse the preprocessing transformations: Normalize -> Convert to PIL Image
    """
    # Imagenet mean and std for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Undo normalization
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    
    # Clip values to [0, 1] (in case normalization introduces out-of-bounds values)
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to NumPy array
    img = tensor.permute(1, 2, 0).numpy()  # C x H x W -> H x W x C for Matplotlib
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255] for visualization
    
    return img

def plot_dataloader_samples(dataloader, num_rows = 3, num_cols=3, figsize=(10, 10)):
    # Get one batch of data
    batch = next(iter(dataloader))
    imgs, labels = batch

    # Select the first sequence from the first few batch samples
    batch_size, seq_len, C, H, W = imgs.shape
    imgs = imgs[:, 0]  # Pick the first frame from each sequence
    labels = labels[:, 0]  # Pick the first label from each sequence
    
    # Define a grid for displaying images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle("Sanity Check: Grid of Images", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= batch_size:
            break  # Avoid errors if batch is smaller than grid size
        
        img_np = reverse_transform(imgs[i])  # Unnormalize and convert
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(f"Label: {labels[i].item():.2f}")

    plt.tight_layout()
    plt.show(block=True)

def display_sequence_as_video(sequence, labels, figsize=(4, 4)):
    """
    Displays a sequence of images as a video animation, with a different label per frame.

    Args:
        sequence (Tensor): Shape [seq_length, C, H, W]
        labels (Tensor): Shape [seq_length] containing a label for each frame.
    """
    seq_length, C, H, W = sequence.shape

    fig, ax = plt.subplots(figsize=figsize)
    # Set the title for the first frame
    ax.set_title(f"Label: {labels[0].item():.2f}")
    ax.axis('off')

    # Initialize the image plot
    img_display = ax.imshow(reverse_transform(sequence[0]), animated=True)

    def update(frame):
        # Update image data
        img_display.set_array(reverse_transform(sequence[frame]))
        # Update the title to show the label for the current frame
        ax.set_title(f"Label: {labels[frame].item():.2f}")
        return [img_display]

    ani = animation.FuncAnimation(fig, update, frames=seq_length, interval=100, blit=True)
    plt.show(block=True)

def plot_video_from_dataloader(dataloader, num_videos=3):
    """
    Displays multiple sequences as video animations from the DataLoader.

    Args:
        dataloader: PyTorch DataLoader
        num_videos (int): Number of videos to display
    """
    batch = next(iter(dataloader))  # Get one batch
    sequence_id, seq_num, imgs, labels = batch  # Assuming dataset returns (sequence_id, seq_num, images, labels)

    batch_size, seq_len, C, H, W = imgs.shape

    for i in range(min(num_videos, batch_size)):  # Display only num_videos sequences
        print(sequence_id[i], seq_num[i])
        # Here, labels[i] is assumed to be a tensor of shape [seq_len]
        display_sequence_as_video(imgs[i], labels[i])  # Show video for sequence i

if __name__ == '__main__':

    data_dir = 'sullychen/07012018/data'
    steering_angles_txt_path = 'sullychen/07012018/data.txt'
    train_dataset_path = 'train_data_filtered.csv'
    val_dataset_path = 'val_data_filtered.csv'
    train_size = 0.8
    seq_len = 64
    imgw = 224
    imgh = 224
    mean = [0.543146, 0.53002986, 0.50673143]
    std = [0.23295668, 0.22123158, 0.22100357]
    batch_size = 8
    shuffle = False

    img_paths = get_full_image_filepaths(data_dir)
    steering_angles = get_steering_angles(steering_angles_txt_path)

    data_pd = convert_to_df(img_paths, steering_angles)
    data_pd_filtered = filter_df_on_turns(data_pd)
    data_pd_filtered = group_data_by_sequences(data_pd_filtered)

    train_dataset_path, val_dataset_path = df_split_train_val(data_pd_filtered, train_dataset_path, 
                                                                    val_dataset_path, train_size)
        
    train_dataset , val_dataset = create_train_val_dataset(train_csv_file = train_dataset_path,
                                                                val_csv_file = val_dataset_path,
                                                                seq_len=seq_len, imgw=imgw, imgh=imgh, mean=mean)
    
    train_sampler = UniqueSequenceSampler(train_dataset, seq_len)
    val_sampler = UniqueSequenceSampler(val_dataset, seq_len)
        
    train_loader, val_loader = create_train_val_loader(train_dataset, val_dataset, train_sampler, val_sampler, 
                                                            batch_size=batch_size, shuffle=shuffle)

    plot_video_from_dataloader(train_loader, num_videos=3)
