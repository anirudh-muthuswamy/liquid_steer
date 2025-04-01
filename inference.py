import torch
import os
import cv2
import collections
import numpy as np
import pandas as pd
import torch.backends.mps as mps
from matplotlib import animation
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from model import NCPModel
from check_data import get_full_image_filepaths
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


#wont work if the dataloader is sampled, need to have consecutive sequence numbers to have 
#temporal dependency
def get_predicted_steering_angles_from_dataloader(model, val_loader, val_dataset, output_dir):
    model.eval()
    output_df = []
    for idx, (sequence_id, seq_num, batch_x, batch_y) in enumerate(val_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with torch.no_grad():
                predictions = model(batch_x)

            keys = list(zip(sequence_id.tolist(), seq_num.tolist()))
            indices = [val_dataset.index_map[key] for key in keys]
            values_df = [val_dataset.sequences[list(val_dataset.sequences.keys())[idx]] for idx in indices]

            for i, df in enumerate(values_df):
                df['predicted_angles'] = predictions[i].detach().cpu().numpy()

            batch_df = pd.concat(values_df, ignore_index=True)  # Concatenate all DataFrames
            output_df.append(batch_df)

    os.makedirs(output_dir, exist_ok=True)

    output_df = pd.concat(output_df, ignore_index=True )
    output_path = os.path.join(output_dir, 'val_loader_predictions.csv')
    pd.DataFrame.to_csv(output_df, output_path, index= False)

    print('Predictions saved to:', output_path)


def get_predicted_steering_angles_from_images(model, images_dir='sullychen/07012018/data', 
                                              num_frames = 1000,
                                              transform_params={
                                                  'mean':[0.543146, 0.53002986, 0.50673143],
                                                  'std':[0.23295668, 0.22123158, 0.22100357],
                                                  'imgh':224,
                                                  'imgw':224}, 
                                              display=False):

    sequence_length = 64
    frame_buffer = collections.deque(maxlen=sequence_length)  # Stores last 64 frames
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=transform_params['mean'], std=transform_params['std'])
    ])

    filepaths = get_full_image_filepaths(images_dir)
    predictions = {'filepath':[],
                   'predicted_angles':[]}

    for (i,frame) in enumerate(filepaths):
        print('processing frame:', i)

        if i == num_frames - 1:
            break

        img = cv2.imread(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (transform_params['imgw'], transform_params['imgh'])) 
        img = transform(img)

        frame_buffer.append(img)  # Add new frame
        # Wait until we have at least 64 frames
        if len(frame_buffer) < sequence_length:
            continue
        
        input_sequence = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(input_sequence)  # Shape: [1, 64]
    
        last_pred = prediction[:, -1].item()  # Return the last prediction

        predictions['filepath'].append(frame)
        predictions['predicted_angles'].append(last_pred)

    predictions_df = pd.DataFrame(predictions)
    pd.DataFrame.to_csv(predictions_df, 'code_files/predictons_from_img.csv')

    return

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        mps.benchmark = True
        print("Using MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    data_dir = 'sullychen/07012018/data'
    steering_angles_txt_path = 'sullychen/07012018/data.txt'
    train_dataset_path = 'train_data_filtered.csv'
    val_dataset_path = 'val_data_filtered.csv'
    checkpoint_path = 'checkpoints/model_epoch4.pth'
    train_size = 0.8
    seq_len = 64
    imgw = 224
    imgh = 224
    mean = [0.543146, 0.53002986, 0.50673143]
    std = [0.23295668, 0.22123158, 0.22100357]
    batch_size = 8
    shuffle = False

    # img_paths = get_full_image_filepaths(data_dir)
    # steering_angles = get_steering_angles(steering_angles_txt_path)

    # data_pd = convert_to_df(img_paths, steering_angles)
    # data_pd_filtered = filter_df_on_turns(data_pd)
    # data_pd_filtered = group_data_by_sequences(data_pd_filtered)

    # train_dataset_path, val_dataset_path = df_split_train_val(data_pd_filtered, train_dataset_path, 
    #                                                                 val_dataset_path, train_size)
        
    # train_dataset , val_dataset = create_train_val_dataset(train_csv_file = train_dataset_path,
    #                                                             val_csv_file = val_dataset_path,
    #                                                             seq_len=seq_len, imgw=imgw, imgh=imgh, mean=mean)
    
    # train_sampler = UniqueSequenceSampler(train_dataset, seq_len)
    # val_sampler = UniqueSequenceSampler(val_dataset, seq_len)
        
    # train_loader_sampled, val_loader_sampled = create_train_val_loader(train_dataset, val_dataset, train_sampler, val_sampler, 
    #                                                         batch_size=batch_size, shuffle=shuffle)

    # plot_video_from_dataloader(train_loader_sampled, num_videos=3)

    #load model from checkpoint:
    # Assuming extracted features are 32-dimensional
    model = NCPModel(num_filters=8, features_per_filter=4, inter_neurons = 12, command_neurons = 6,
                     motor_neurons = 1, sensory_fanout = 6, inter_fanout = 4, 
                     recurrent_command_synapses = 6, motor_fanin = 6, seed = 20190120, device=device) 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model = model.to(device)
    model_ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(model_ckpt['model_state_dict'])
    optimizer.load_state_dict(model_ckpt['optimizer_state_dict'])
    current_epoch = model_ckpt['epoch']
    training_losses = model_ckpt['training_losses']
    validation_losses = model_ckpt['validation_losses']

    print('checkpoint loaded successfully!')

    # get_predicted_steering_angles_from_dataloader(model, val_loader_sampled, val_dataset, output_dir='code_files/')

    get_predicted_steering_angles_from_images(model, data_dir)
