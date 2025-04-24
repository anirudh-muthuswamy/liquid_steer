import os
import re
import cv2
import json
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.backends.mps as mps
import torch.backends.cudnn as cudnn


pd.set_option("display.max_rows", 200)
plt.ion() 

# get the image filepath given the data directory
def get_full_image_filepaths(data_dir):
    filepaths = os.listdir(data_dir)
    sorted_filepaths = sorted(filepaths, key=lambda x: int(re.search(r'\d+', x).group()))
    full_filepaths = [os.path.join(data_dir, file) for file in sorted_filepaths]

    return full_filepaths

#get the steering angle and timestamps as list from data.txt file
def get_steering_angles(path):
    file = open(path, 'r')
    lines = file.readlines()
    #change this to line[:-1] if using kaggle dataset else keep the same for sullychen dataset
    steering_angles = [(line[:-1]).split(' ')[1].split(',')[0] for line in lines]
    timestamps = [line.strip().split(' ', 1)[1].split(',')[1] for line in lines]

    return [steering_angles, timestamps]

# convert the filepaths, steering angles and timestamps as columns in a dataframe. Additionally normalize the
# data if required in a range of (-1, 1) where wheel max and min angles are 360 degrees.
def convert_to_df(full_filepaths, steering_angles, timestamps, norm=True):
    data = pd.DataFrame({'filepath':full_filepaths,'steering_angle':steering_angles, 'timestamps':timestamps})
    data['parsed_timestamp'] = data['timestamps'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f'))
    data = data.drop('timestamps', axis=1)
    data = data.reset_index(drop=True)
    data['steering_angle'] = data['steering_angle'].astype('float')
    #normalize steering angles to -1,1 based on actual steering range (-360, 360) of wheel
    if norm == True:
        data['steering_angle'] = data['steering_angle'] / 360
    return data

# display each image with steering angles (using a stock steering wheel image)
def display_images_with_angles(data_pd, steering_wheel_img):

    # steering wheel image
    img = cv2.imread(steering_wheel_img,0)
    rows,cols = img.shape
    i = 0
    xs = data_pd['filepath'].values
    ys = data_pd['steering_angle'].values

    #Press q to stop 
    while(cv2.waitKey(100) != ord('q')) and i < len(xs):
        try:
            # Driving image displayed as video
            full_image = cv2.imread(xs[i])

            degrees = ys[i] * 360
            print("Steering angle: " + str(degrees) + " (actual)")
            cv2.imshow("frame", full_image)

            # angle at which the steering wheel image should be rotated
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-degrees,1)
            dst = cv2.warpAffine(img,M,(cols,rows))

            cv2.imshow("steering wheel", dst)
        except:
            print('ERROR at', i)
        i += 1
    cv2.destroyAllWindows()

# display the frequency of steering angles for the dataframe before or after filtering to see distribution
def disp_freq_steering_angles(data_pd):
    # Define bin edges from -1 to 1 with step 0.1
    bin_edges = list(range(-10, 11))  # steering angle is between -1 and 1, multiply by 10
    bin_edges = [x / 10 for x in bin_edges]  # Convert back to decimal values

    data_pd['binned'] = pd.cut(data_pd['steering_angle'].astype('float'), bins=bin_edges, right=False)
    # count occurrences in each bin
    bin_counts = data_pd['binned'].value_counts().sort_index()

    plt.figure(figsize=(12, 5))
    bin_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.xlabel("Steering Angle Range")
    plt.ylabel("Frequency")
    plt.title("Frequency of Steering Angles in 0.1 Intervals")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

# Display the start and end points of sequences and see which parts of the data are cut off when filtering takes place
def disp_start_and_end_in_filtered_data(data_pd_filtered):
    turn_starts = data_pd_filtered[data_pd_filtered['turn_shift'] == 1]
    turn_ends = data_pd_filtered[data_pd_filtered['turn_shift'] == -1]

    plt.figure(figsize=(12, 5))
    plt.plot(data_pd_filtered.index, data_pd_filtered['steering_angle'], label="Steering Angle", alpha=0.5)
    plt.scatter(turn_starts.index, turn_starts['steering_angle'], color='red', label="Turn Start", marker="o")
    plt.scatter(turn_ends.index, turn_ends['steering_angle'], color='blue', label="Turn End", marker="x")

    plt.xlabel("Frame Index")
    plt.ylabel("Steering Angle")
    plt.title("Turn Start and End Points in Steering Angle Data")
    plt.legend()
    plt.grid(True)
    plt.show()

# this method is used to reduce the dataset size, since most roads are straight, in an essence balancing such that our 
# smaller dataset contains more curves than the original. This is done using a turn threshold, and a buffer of frames 
# before and after turns
def filter_df_on_turns(data_pd, turn_threshold = 0.08, buffer_before = 32, buffer_after = 32):

    turn_threshold = turn_threshold 
    buffer_before = buffer_before  
    buffer_after = buffer_after 

    data_pd['index'] = data_pd.index 
    data_pd['turning'] = (data_pd['steering_angle'].abs() > turn_threshold).astype(int)
    data_pd['turn_shift'] = data_pd['turning'].diff()  # 1 indicates start, -1 indicates end

    # this gets turn start and end indices
    turn_starts = data_pd[data_pd['turn_shift'] == 1].index
    turn_ends = data_pd[data_pd['turn_shift'] == -1].index
    if len(turn_ends) > 0 and turn_starts[0] > turn_ends[0]:  
        turn_ends = turn_ends[1:]  # drop first turn_end if it comes before a start

    selected_indices = set()
    for start, end in zip(turn_starts, turn_ends):
        start_idx = max(0, start - buffer_before)
        end_idx = min(len(data_pd) - 1, end + buffer_after)
        selected_indices.update(range(start_idx, end_idx + 1))

    data_pd_filtered = data_pd.loc[sorted(selected_indices)].reset_index(drop=True)
    data_pd_filtered = data_pd_filtered.drop(columns=['turning', 'turn_shift'])
    data_pd_filtered["sequence_id"] = (data_pd_filtered["index"].diff() != 1).cumsum()

    return data_pd_filtered

# This groupd data by sequences (i.e the sequence id column added when filter_df_on_turns is called)
def group_data_by_sequences(data_pd_filtered):
    sequence_lengths = data_pd_filtered.groupby("sequence_id").size()

    print(f"Minimum sequence length: {min(sequence_lengths)}")
    print(f"Maximum sequence length: {max(sequence_lengths)}")

    valid_sequences = sequence_lengths[sequence_lengths >= 40].index
    data_pd_filtered = data_pd_filtered[data_pd_filtered["sequence_id"].isin(valid_sequences)]

    print(f"Total valid sequences: {len(valid_sequences)}")

    return data_pd_filtered

# gets preprocessed/ filtered dataframe of steering angles using filter_df_on_turns and group_data_by_sequences
def get_preprocessed_data_pd(data_dir, steering_angles_txt_path, filter = True,
                             turn_threshold = 0.06, buffer_before = 60, buffer_after = 60,
                             norm=True, save_dir = 'data/csv_files'):
    img_paths = get_full_image_filepaths(data_dir)
    steering_angles, timestamps = get_steering_angles(steering_angles_txt_path)

    data_pd = convert_to_df(img_paths, steering_angles, timestamps, norm)
    if filter and norm:
        data_pd_filtered = filter_df_on_turns(data_pd, turn_threshold = turn_threshold, 
                                            buffer_before = buffer_before, buffer_after = buffer_after)
        data_pd_filtered = group_data_by_sequences(data_pd_filtered)

        # save
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_pd_filtered.to_csv(os.path.join(save_dir,f"flt_ncp_tt_{turn_threshold}_bb_{buffer_before}_ba_{buffer_after}.csv"), index=False)

        return data_pd_filtered
    
    elif filter and not norm:
        print('Error: If filtering, then steering values should be normalized (-1, 1), set norm to True')
        exit(1)

    else:
        sequence_id = 0
        sequence_ids = [sequence_id]

        # iterate through rows to calculate time differences (if greater than 3 seconds 
        # or not) and assign sequence IDs
        for i in range(1, len(data_pd)):
            time_diff = (data_pd['parsed_timestamp'][i] - data_pd['parsed_timestamp'][i-1]).total_seconds()
            if time_diff > 3:
                sequence_id += 1
            sequence_ids.append(sequence_id)

        data_pd['sequence_id'] = sequence_ids

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_pd.to_csv(os.path.join(save_dir,f"ncp_unfiltered.csv"), index=False)
        
        return data_pd
    
# calculate mean and std of greyscale and rgb images in dataset
def calculate_mean_and_std(dataset_path, rgb=True):
    num_pixels = 0
    if rgb:
        channel_sum = np.zeros(3) 
        channel_sum_squared = np.zeros(3)
    else:
        channel_sum = np.zeros(1,)
        channel_sum_squared = np.zeros(1,) 

    for root, _, files in os.walk(dataset_path):
        for file in files:
            image_path = os.path.join(root, file)
            if rgb:
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.open(image_path)

            pixels = np.array(image) / 255.0  # normalize pixel values
            if rgb:
                num_pixels += pixels.size // 3 
            else:
                num_pixels += pixels.size // 1

            channel_sum += np.sum(pixels, axis=(0, 1))
            channel_sum_squared += np.sum(pixels ** 2, axis=(0, 1))

    mean = channel_sum / num_pixels
    std = np.sqrt((channel_sum_squared / num_pixels) - mean ** 2)
    return mean, std

#plot loss and accuracy given list of train loss and validation loss
def plot_loss_accuracy(train_loss, val_loss, save_dir=None):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linestyle='-')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# split the complete dataframe into train and validation dataframes
def df_split_train_val(df_filtered, train_csv_filename, val_csv_filename,
                       save_dir='data/csv_files',train_size = 0.8):
    train_dataset = df_filtered[:int(train_size * len(df_filtered))]
    val_dataset = df_filtered[int(train_size * len(df_filtered)):]
    print('Train dataset length:', len(train_dataset))
    print('Val dataset length:', len(val_dataset))
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_dataset.to_csv(os.path.join(save_dir,train_csv_filename), index=False)
    val_dataset.to_csv(os.path.join(save_dir,val_csv_filename), index=False)

    return os.path.join(save_dir,train_csv_filename), os.path.join(save_dir,val_csv_filename)

#method to load config.json files for each training session
def load_config(config_path='./project_src/conv_ncp_config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# get torch device based on what torch detects for the system
def get_torch_device(dont_use_mps=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available() and dont_use_mps==False:
        device = torch.device('mps')
        mps.benchmark = True
        print("Using MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device