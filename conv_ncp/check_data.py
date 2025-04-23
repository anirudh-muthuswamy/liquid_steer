import pandas as pd
import re
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import os

pd.set_option("display.max_rows", 200)
plt.ion() 

def get_full_image_filepaths(data_dir):
    filepaths = os.listdir(data_dir)
    sorted_filepaths = sorted(filepaths, key=lambda x: int(re.search(r'\d+', x).group()))
    full_filepaths = [os.path.join(data_dir, file) for file in sorted_filepaths]

    return full_filepaths

def get_steering_angles(path):
    file = open(path, 'r')
    lines = file.readlines()
    #change this to line[:-1] if using kaggle dataset else keep the same for sullychen dataset
    steering_angles = [(line[:-1]).split(' ')[1].split(',')[0] for line in lines]
    timestamps = [line.strip().split(' ', 1)[1].split(',')[1] for line in lines]

    return [steering_angles, timestamps]

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
            # Driving Test Image displayed as Video
            full_image = cv2.imread(xs[i])

            degrees = ys[i] * 360
            print("Steering angle: " + str(degrees) + " (actual)")
            cv2.imshow("frame", full_image)

            # Angle at which the steering wheel image should be rotated
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-degrees,1)
            dst = cv2.warpAffine(img,M,(cols,rows))

            cv2.imshow("steering wheel", dst)
        except:
            print('ERROR at', i)
        i += 1
    cv2.destroyAllWindows()

def disp_freq_steering_angles(data_pd):
    # Define bin edges from -1 to 1 with step 0.1
    bin_edges = list(range(-10, 11))  # Since steering angle is between -1 and 1, multiply by 10
    bin_edges = [x / 10 for x in bin_edges]  # Convert back to decimal values

    # Assign steering angles to bins
    data_pd['binned'] = pd.cut(data_pd['steering_angle'].astype('float'), bins=bin_edges, right=False)

    # Count occurrences in each bin
    bin_counts = data_pd['binned'].value_counts().sort_index()

    # Plot bar chart
    plt.figure(figsize=(12, 5))
    bin_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    # Formatting
    plt.xlabel("Steering Angle Range")
    plt.ylabel("Frequency")
    plt.title("Frequency of Steering Angles in 0.1 Intervals")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

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

def filter_df_on_turns(data_pd, turn_threshold = 0.06, buffer_before = 60, buffer_after = 60):
    # Parameters
    turn_threshold = turn_threshold  # Define turn threshold (absolute value)
    buffer_before = buffer_before    # Frames to include before a turn
    buffer_after = buffer_after     # Frames to include after a turn

    # Load your dataset (assuming it's a DataFrame named df)
    data_pd['index'] = data_pd.index  # Preserve original ordering if needed

    # Identify where turning happens
    data_pd['turning'] = (data_pd['steering_angle'].abs() > turn_threshold).astype(int)

    # Find where turns start and end
    data_pd['turn_shift'] = data_pd['turning'].diff()  # 1 indicates start, -1 indicates end

    # Get turn start and end indices
    turn_starts = data_pd[data_pd['turn_shift'] == 1].index
    turn_ends = data_pd[data_pd['turn_shift'] == -1].index

    # Ensure equal number of start and end points
    if len(turn_ends) > 0 and turn_starts[0] > turn_ends[0]:  
        turn_ends = turn_ends[1:]  # Drop the first turn_end if it comes before a start

    # Selected indices for keeping
    selected_indices = set()

    for start, end in zip(turn_starts, turn_ends):
        # Include a buffer of frames before and after
        start_idx = max(0, start - buffer_before)
        end_idx = min(len(data_pd) - 1, end + buffer_after)
        
        # Add indices to selection
        selected_indices.update(range(start_idx, end_idx + 1))

    # Create filtered dataset
    data_pd_filtered = data_pd.loc[sorted(selected_indices)].reset_index(drop=True)

    # Drop temporary columns
    data_pd_filtered = data_pd_filtered.drop(columns=['turning', 'turn_shift'])

    # Detect sequence breaks (where the original index is not continuous)
    data_pd_filtered["sequence_id"] = (data_pd_filtered["index"].diff() != 1).cumsum()

    return data_pd_filtered

def group_data_by_sequences(data_pd_filtered):
    sequence_lengths = data_pd_filtered.groupby("sequence_id").size()

    print(f"Minimum sequence length: {min(sequence_lengths)}")
    print(f"Maximum sequence length: {max(sequence_lengths)}")

    # plt.plot(sequence_lengths)

    # Keep only sequences with at least 10 frames (adjust as needed)
    valid_sequences = sequence_lengths[sequence_lengths >= 40].index
    data_pd_filtered = data_pd_filtered[data_pd_filtered["sequence_id"].isin(valid_sequences)]

    print(f"Total valid sequences: {len(valid_sequences)}")

    return data_pd_filtered

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

        # Save
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

        # Iterate through rows to calculate time differences (if greater than 3 seconds 
        # or not) and assign sequence IDs
        for i in range(1, len(data_pd)):
            time_diff = (data_pd['parsed_timestamp'][i] - data_pd['parsed_timestamp'][i-1]).total_seconds()
            if time_diff > 3:
                sequence_id += 1
            sequence_ids.append(sequence_id)

        # Add sequence_id column to DataFrame
        data_pd['sequence_id'] = sequence_ids

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_pd.to_csv(os.path.join(save_dir,f"ncp_unfiltered.csv"), index=False)
        
        return data_pd