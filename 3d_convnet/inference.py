import torch
import argparse
import cv2
import os
import collections
import pandas as pd
import numpy as np
from torchvision import transforms
from .model import TemporalResNet
from ..utils import get_full_image_filepaths, get_steering_angles, get_torch_device

def get_predicted_steering_angles_from_images(model, device, images_dir='data/sullychen/07012018/edge_maps', 
                                              steering_angles_path='data/sullychen/07012018/data.txt', save_dir = './',
                                              num_frames = 1000,
                                              transform_params={
                                                  'mean':[0.5],
                                                  'std':[0.5],
                                                  'imgh':455,
                                                  'imgw':123}):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    sequence_length = 3
    frame_buffer = collections.deque(maxlen=sequence_length)  # Stores last "seq_len" frames
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=transform_params['mean'], std=transform_params['std'])])

    filepaths = get_full_image_filepaths(images_dir)
    steering_angles, _ = get_steering_angles(steering_angles_path)
    steering_angles = [float(angle) for angle in steering_angles]
    predictions = {'filepath':[],
                   'predicted_angles':[]}

    for (i,frame) in enumerate(filepaths):
        print('processing frame:', i, end='\t')

        if i == num_frames - 1:
            break

        img = np.expand_dims(cv2.imread(frame, cv2.IMREAD_GRAYSCALE),2)
        img = cv2.resize(img, (transform_params['imgh'], transform_params['imgw'])) 
        img = transform(img)

        frame_buffer.append(img)
        if len(frame_buffer) < sequence_length:
            continue
        
        input_sequence = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(input_sequence)  # [1, 64]
    
        last_pred = prediction[:, -1].item()  # last prediction
        print('pred angle:', last_pred)

        predictions['filepath'].append(frame)
        predictions['predicted_angles'].append(last_pred)

    predictions_df = pd.DataFrame(predictions)
    pd.DataFrame.to_csv(predictions_df, os.path.join(save_dir,'conv_lstm_preds.csv'))

def parse_args():
    parser = argparse.ArgumentParser(description="Setup data paths and training parameters")
    # file paths
    parser.add_argument("--data_dir",type=str, default="data/sullychen/07012018/edge_maps", 
                        help="root directory for raw images")
    parser.add_argument("--steering_angles_path", type=str, default="data/sullychen/07012018/data.txt",
                        help="path to steering angles text file")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/checkpoints_lstm/checkpoints_rad_1e-5/model_epoch7.pth",
                        help="path to load/save model checkpoint")
    parser.add_argument("--save_dir", type=str,
                        default="predictions/",
                        help="path to load/save model checkpoint")
    parser.add_argument("--seq_len", type=int, default=8, help="number of frames per sequence")
    parser.add_argument("--num_frames", type=int, default=1000, help="number of frames to infer")
    parser.add_argument("--imgw", type=int, default=455, help="resized image width")
    parser.add_argument("--imgh", type=int, default=123, help="resized image height")
    
    return parser.parse_args()

if __name__ == '__main__':

    device = get_torch_device(dont_use_mps=True)
    args = parse_args()

    data_dir = args.data_dir
    steering_angles_path = args.steering_angles_path
    checkpoint_path = args.checkpoint_path
    seq_len = args.seq_len
    imgw = args.imgw
    imgh = args.imgh
    num_frames = args.num_frames
    save_dir = args.save_dir

    model = TemporalResNet(in_channels=1,seq_len=seq_len, height=imgh, width=imgw).to(device=device)
    model_ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(model_ckpt['model_state_dict'])
    current_epoch = model_ckpt['epoch']
    training_losses = model_ckpt['training_losses']
    validation_losses = model_ckpt['val_losses']
    print('checkpoint loaded successfully!')

    get_predicted_steering_angles_from_images(model, device, data_dir, steering_angles_path, save_dir, 
                                              num_frames)