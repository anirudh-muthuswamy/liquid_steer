import torch
import os
import argparse
import cv2
import collections
import numpy as np
import pandas as pd
from matplotlib import animation
from torchvision import transforms
import matplotlib.pyplot as plt
from .model import ConvNCPModel
from .dataset import create_train_val_dataset, create_train_val_loader
from ..utils import (get_full_image_filepaths, get_torch_device)
from torch.utils.data import Sampler

class UniqueSequenceSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        # extract unique seq_ids from OrderedDict keys
        unique_seq_ids = set(seq_id for seq_id, _ in dataset.sequences.keys())

        # indices for the first sequence (seq_num = 0) for each seq_id if available
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

    #imagenet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    tensor = torch.clamp(tensor, 0, 1)
    
    img = tensor.permute(1, 2, 0).numpy()  # C, H, W to H, W, C for Matplotlib
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255] for visualization
    return img

def plot_dataloader_samples(dataloader, num_rows = 3, num_cols=3, figsize=(10, 10)):

    batch = next(iter(dataloader))
    imgs, labels = batch
    # select the first sequence 
    batch_size = imgs.shape[0]
    imgs = imgs[:, 0]  # pick first frame
    labels = labels[:, 0]  # pick first label
    
    #for plotting
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle("Sanity Check: Grid of Images", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= batch_size:
            break  #if batch is smaller than grid size
        
        img_np = reverse_transform(imgs[i])
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
    seq_length = sequence.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Label: {labels[0].item():.2f}")
    ax.axis('off')

    img_display = ax.imshow(reverse_transform(sequence[0]), animated=True)

    def update(frame):
        img_display.set_array(reverse_transform(sequence[frame]))
        ax.set_title(f"Label: {labels[frame].item():.2f}")
        return [img_display]

    animation.FuncAnimation(fig, update, frames=seq_length, interval=100, blit=True)
    plt.show(block=True)

def plot_video_from_dataloader(dataloader, num_videos=3):
    """
    Displays multiple sequences as video animations from the DataLoader.

    Args:
        dataloader: PyTorch DataLoader
        num_videos (int): Number of videos to display
    """
    batch = next(iter(dataloader))
    sequence_id, seq_num, imgs, labels = batch 

    batch_size = imgs.shape[0]

    for i in range(min(num_videos, batch_size)):  #num_videos sequences
        print(sequence_id[i], seq_num[i])
        display_sequence_as_video(imgs[i], labels[i])  #video for sequence i


def get_predicted_steering_angles_from_images(model, images_dir='sullychen/07012018/data', 
                                              seq_len = 32,
                                              num_frames = 1000,
                                              transform_params={
                                                  'mean':[0.485, 0.456, 0.406],
                                                  'std':[0.229, 0.224, 0.225],
                                                  'imgh':224,
                                                  'imgw':224}):

    seq_len = 32
    frame_buffer = collections.deque(maxlen=seq_len)  # stores last seq_len frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_params['mean'], std=transform_params['std'])
    ])

    filepaths = get_full_image_filepaths(images_dir)
    predictions = {'filepath':[],
                   'predicted_angles':[]}

    for (i,frame) in enumerate(filepaths):
        print('processing frame:', i, end='\t')

        if i == num_frames - 1:
            break

        img = cv2.imread(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (transform_params['imgw'], transform_params['imgh'])) 
        img = transform(img)

        frame_buffer.append(img) 
        #until we have at least seq_len frames
        if len(frame_buffer) < seq_len:
            continue
        
        input_sequence = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(input_sequence)  # Shape: [1, seq_len]
    
        last_pred = prediction[:, -1].item()  # Return the last prediction
        print('pred angle:', last_pred)

        predictions['filepath'].append(frame)
        predictions['predicted_angles'].append(last_pred)

    predictions_df = pd.DataFrame(predictions)
    pd.DataFrame.to_csv(predictions_df, 'code_files/predictons_from_img.csv')
    return

def parse_args():
    parser = argparse.ArgumentParser(description="ConvNCP inference")
    parser.add_argument("--data_dir", type=str, default="data/sullychen/07012018/data")
    parser.add_argument("--train_dataset_path", type=str, default="data/csv_files_experimental/train_flt_ncp_tt_0.08_bb_32_ba_32.csv")
    parser.add_argument("--val_dataset_path", type=str, default="data/csv_files_experimental/val_flt_ncp_tt_0.08_bb_32_ba_32.csv")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/conv_ncp/checkpoints_bs_16_sql_32_ss_16_wl_heinit/model_epoch20.pth")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--imgw", type=int, default=224)
    parser.add_argument("--imgh", type=int, default=224)
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406],
                        help="Normalization mean ")
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225],
                        help="Normalization std ")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_shuffle", action="store_true",
                        help="if set, shuffle training data")
    parser.add_argument("--plot_sequences", action="store_true",
                        help="if set, visualize sequences during loading")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    device = get_torch_device()

    data_dir = args.data_dir
    train_dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path
    checkpoint_path = args.checkpoint_path
    seq_len = args.seq_len
    imgw = args.imgw
    imgh = args.imgh
    mean = args.mean
    std = args.std
    batch_size = args.batch_size
    train_shuffle = args.train_shuffle
    plot_sequences = args.plot_sequences

    #Plot dataloader sequences
    #-------------------------
    if plot_sequences:
        train_dataset , val_dataset = create_train_val_dataset(train_csv_file = train_dataset_path,
                                                                    val_csv_file = val_dataset_path,
                                                                    seq_len=seq_len, imgw=imgw, imgh=imgh, mean=mean, std=std,
                                                                    crop=False)
        
        train_sampler = UniqueSequenceSampler(train_dataset)
        val_sampler = UniqueSequenceSampler(val_dataset)
            
        train_loader_sampled, val_loader_sampled = create_train_val_loader(train_dataset, val_dataset, train_sampler, val_sampler, 
                                                                batch_size=batch_size, train_shuffle=train_shuffle)

        plot_video_from_dataloader(train_loader_sampled, num_videos=3)

    # load model from checkpoint:
    # Assuming extracted features are 32-dimensional
    model_ckpt = torch.load(checkpoint_path, map_location=device)
    print(model_ckpt['train_params'])
    if 'feat_per_filter' in model_ckpt['train_params']:
        model = ConvNCPModel(num_filters=8, features_per_filter=model_ckpt['train_params']['feat_per_filt'], inter_neurons = 12, command_neurons = 6,
                     motor_neurons = 1, sensory_fanout = 6, inter_fanout = 4, 
                     recurrent_command_synapses = 6, motor_fanin = 6, seed = 20190120) 
    else: #use default features_per_filter=4
        model = ConvNCPModel(num_filters=8, features_per_filter=4, inter_neurons = 12, command_neurons = 6,
                     motor_neurons = 1, sensory_fanout = 6, inter_fanout = 4, 
                     recurrent_command_synapses = 6, motor_fanin = 6, seed = 20190120) 

    model = model.to(device)
    model.load_state_dict(model_ckpt['model_state_dict'])
    current_epoch = model_ckpt['epoch']
    training_losses = model_ckpt['training_losses']
    validation_losses = model_ckpt['validation_losses']

    print('checkpoint loaded successfully!')
    get_predicted_steering_angles_from_images(model, data_dir)
