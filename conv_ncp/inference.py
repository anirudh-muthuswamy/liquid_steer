import torch
import argparse
import cv2
import os
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

# This is used for plotting different sequences. Since the step size between sequences are small, 
# consecutive sequences would be very similar if theres is just a change of 1-5 frames. Hence, 
# we use a sequence sampler, that just samples the first sequence number for a particular sequence id. 
# A sequence id might be split into different sequences of a smaller length. 
# This allows for easier visualization and sanity checks

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

#reverse transform the transformations applied during dataset and dataloader creation
def reverse_transform(tensor):

    #imagenet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    tensor = torch.clamp(tensor, 0, 1)
    
    img = tensor.permute(1, 2, 0).numpy()  # C, H, W to H, W, C for Matplotlib
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255] for visualization
    return img

#Method to plot out dataloder samples
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

# This method animates the sequences as videos with a different label (i,e the steering angle) / frame
def display_sequence_as_video(sequence, labels, figsize=(4, 4)):

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

# calls the display_sequence_as_vide for the number of videos passed as input
# this should be less than the batch size, else batch size number of images are plotted/visualized
def plot_video_from_dataloader(dataloader, num_videos=3):

    batch = next(iter(dataloader))
    sequence_id, seq_num, imgs, labels = batch 

    batch_size = imgs.shape[0]

    for i in range(min(num_videos, batch_size)):  #num_videos sequences
        print(sequence_id[i], seq_num[i])
        display_sequence_as_video(imgs[i], labels[i])  #video for sequence i

# Uses a collections.dequeue object for a frame buffer -> equal to the sequence length the model is trained 
# on . The frame buffer is used for inference to get the following steering angle given the previous
# "seq length" number of images. Saves the predictions in a csv file

def get_predicted_steering_angles_from_images(model, images_dir='sullychen/07012018/data',
                                              save_dir='predictions',
                                              seq_len = 32,
                                              num_frames = 1000,
                                              transform_params={
                                                  'mean':[0.485, 0.456, 0.406],
                                                  'std':[0.229, 0.224, 0.225],
                                                  'imgh':224,
                                                  'imgw':224}):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
    pd.DataFrame.to_csv(predictions_df, os.path.join(save_dir,'conv_ncp_predictions.csv'))
    return

# Argparser to take in inputs for running inference
def parse_args():
    parser = argparse.ArgumentParser(description="ConvNCP inference")
    parser.add_argument("--data_dir", type=str, default="data/sullychen/07012018/data")
    parser.add_argument("--train_dataset_path", type=str, default="data/csv_files_experimental/train_flt_ncp_tt_0.08_bb_32_ba_32.csv")
    parser.add_argument("--val_dataset_path", type=str, default="data/csv_files_experimental/val_flt_ncp_tt_0.08_bb_32_ba_32.csv")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/conv_ncp_exp/sl_32_ss_16_bs16_mse_crop_lr1e-3/model_epoch6.pth")
    parser.add_argument("--save_dir", type=str,default="predictions/",
                        help="path to load/save model checkpoint")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--imgw", type=int, default=224)
    parser.add_argument("--imgh", type=int, default=224)
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406],
                        help="Normalization mean ")
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225],
                        help="Normalization std ")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=1000, help="number of frames to infer")
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
    save_dir = args.save_dir
    num_frames = args.num_frames

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
    model_ckpt = torch.load(checkpoint_path, map_location=device)
    print('Train Params:\n', model_ckpt['train_params'])
    if 'feat_per_filter' in model_ckpt['train_params']:
        model = ConvNCPModel(num_filters=8, features_per_filter=model_ckpt['train_params']['feat_per_filt'], inter_neurons = 12, 
                             command_neurons = 6, motor_neurons = 1, sensory_fanout = 6, inter_fanout = 4, 
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
    get_predicted_steering_angles_from_images(model, data_dir, save_dir, seq_len, num_frames)