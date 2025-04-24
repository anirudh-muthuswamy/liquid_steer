import torch
import argparse
import cv2
import os
import collections
import pandas as pd
from torchvision import transforms
from .model import STConvLSTM
from ..utils import get_full_image_filepaths, get_steering_angles, get_torch_device


# Uses a collections.dequeue object for a frame buffer -> equal to the sequence length the model is trained 
# on . The frame buffer is used for inference to get the following steering angle given the previous
# "seq length" number of images. Saves the predictions in a csv file

def get_predicted_steering_angles_from_images(model, device, images_dir='sullychen/07012018/data', 
                                              steering_angles_path='data/sullychen/07012018/data.txt', save_dir = './',
                                              seq_len = 3,
                                              num_frames = 1000,
                                              transform_params={
                                                  'mean':[0.485, 0.456, 0.406],
                                                  'std':[0.229, 0.224, 0.225],
                                                  'imgh':224,
                                                  'imgw':224}):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    frame_buffer = collections.deque(maxlen=seq_len)  # Stores last "seq_len" frames
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

        img = cv2.imread(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (transform_params['imgw'], transform_params['imgh'])) 
        img = transform(img)

        frame_buffer.append(img)
        if len(frame_buffer) < seq_len:
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
    parser.add_argument("--data_dir",type=str, default="data/sullychen/07012018/data", 
                        help="root directory for raw images")
    parser.add_argument("--steering_angles_path", type=str, default="data/sullychen/07012018/data.txt",
                        help="path to steering angles text file")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/checkpoints_lstm/checkpoints_rad_1e-5/model_epoch7.pth",
                        help="path to load/save model checkpoint")
    parser.add_argument("--save_dir", type=str,
                        default="predictions/",
                        help="path to load/save model checkpoint")
    parser.add_argument("--seq_len", type=int, default=3, help="number of frames per sequence")
    parser.add_argument("--num_frames", type=int, default=1000, help="number of frames to infer")
    parser.add_argument("--imgw", type=int, default=224, help="resized image width")
    parser.add_argument("--imgh", type=int, default=224, help="resized image height")
    
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

    model = STConvLSTM(seq_len=seq_len, height=imgh, width=imgw).to(device=device)
    model_ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(model_ckpt['model_state_dict'])
    current_epoch = model_ckpt['epoch']
    training_losses = model_ckpt['training_losses']
    validation_losses = model_ckpt['val_losses']
    print('checkpoint loaded successfully!')

    get_predicted_steering_angles_from_images(model, device, data_dir, steering_angles_path, save_dir, seq_len,
                                              num_frames)