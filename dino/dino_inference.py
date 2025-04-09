import cv2
import pandas as pd
import numpy as np
import torch
import torch.backends.mps as mps
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torchvision import transforms as T
from dino_unet import UNetResNet50

def create_video_from_df(df, output_path='output.mp4', fps=24, size=None):
    """
    Create a video from image file paths in a DataFrame, applying shadow removal.
    
    Args:
        df (pd.DataFrame): DataFrame with 'filepath' column containing image paths.
        output_path (str): Output video path.
        fps (int): Frames per second.
        size (tuple): (width, height). If None, inferred from first image.
    """

    filepaths = df['filepath'].tolist()
    if not filepaths:
        raise ValueError("No filepaths found in DataFrame.")

    # Read first frame to get dimensions
    first_frame = cv2.imread(filepaths[0])
    if first_frame is None:
        raise ValueError(f"Could not read image: {filepaths[0]}")

    if size is None:
        height, width, _ = first_frame.shape
        size = (width, height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for i,path in enumerate(filepaths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"[Warning] Skipping unreadable image: {path}")
            continue
        out.write(frame)

    out.release()
    print(f"[Done] Video saved to: {output_path}")

def ema_smoothing(current_mask, ema_mask, alpha=0.2):
    """
    Apply exponential moving average (EMA) smoothing to binary masks over time.
    Args:
        current_mask (np.ndarray): Current frame's mask (binary 0/1).
        ema_mask (np.ndarray): Running EMA mask.
        alpha (float): Smoothing factor. Lower = smoother.
    Returns:
        Tuple of (smoothed binary mask, updated ema_mask)
    """
    if ema_mask is None:
        ema_mask = current_mask.astype(np.float32)
    else:
        ema_mask = alpha * current_mask + (1 - alpha) * ema_mask
    
    smoothed_mask = (ema_mask > 0.5).astype(np.uint8)
    return smoothed_mask, ema_mask

def process_inference_mask(mask,kernel, min_area=500):
    """
    Smooth the predicted mask by removing small objects and applying morphological operations.
    Args:
        mask (np.ndarray): Binary mask from model output.
        min_area (int): Minimum area to keep a connected component.
    Returns:
        np.ndarray: Cleaned mask.
    """
    mask = np.uint8(mask == 1) * 255  # Convert to 0-255 binary mask

    # Morphological open and close to smooth edges

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    # Convert back to binary class map (0 or 1)
    return (cleaned_mask > 0).astype(np.uint8)

def getTransform():
        return T.Compose([
            T.Resize((480, 640), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.Lambda(lambda x: x.float() / 255.0),
            T.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
def infer_single_image(model, transform_fn, device, image):
    """
    Run inference on a single image using the same augmentations and preprocessing 
    as defined in the dataset class.
    
    Args:
        model: Trained PyTorch segmentation model.
        image: Input image in BGR format (as returned by cv2.imread).
    
    Returns:
        output: Model output after sigmoid and argmax (1-channel prediction).
        image: Normalized and resized image used for inference (torch.Tensor).
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    image_tensor = transform_fn(image_tensor).contiguous(memory_format=torch.channels_last).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output).cpu().numpy()  # (1, C, H, W)
        output = np.argmax(output, axis=1)  # Convert to 1-channel output if needed

    return output.squeeze(0), image_tensor.squeeze(0)  # Return (H, W) and (C, H, W)

def draw_segmentation_map(labels, palette):
    """
    :param labels: Label array from the model. Should be of shape 
        <height x width>. No channel information required.
    :param palette: List containing color information.
        e.g. [[0, 255, 0], [255, 255, 0]] 
    """
    # create Numpy arrays containing zeros
    # to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(palette)):
        index = labels == label_num
        red_map[index] = palette[label_num][0]
        green_map[index] = palette[label_num][1]
        blue_map[index] = palette[label_num][2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    """
    :param image: Image in RGB format.
    :param segmented_image: Segmentation map in RGB format. 
    """
    alpha = 0.8  # transparency for the original image
    beta = 1.0   # transparency for the segmentation map
    gamma = 0    # scalar added to each sum
    segmented_image = np.uint8(segmented_image)
    image = np.array(image.cpu()).transpose(1,2,0)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image * std) + mean  # Revert normalization

    # Convert to [0, 255] and uint8
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    # Overlay the segmentation map on the original image
    overlay = cv2.addWeighted(image, alpha, segmented_image, beta, gamma)
    transform_fn = getTransform
    
    return overlay

if __name__ == "__main__":

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

    data_df = pd.read_csv('data/csv_files/ncp_steering_data.csv')[0:400]


    create_video_from_df(data_df, output_path='sullychen.mp4', fps=20)

    # Ensure the VideoCapture is opened correctly
    cap = cv2.VideoCapture("sullychen.mp4")

    # Prepare to save the output video using the 'H264' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'H264' codec if available
    out = cv2.VideoWriter('sullychen_overlayed_10x10_200.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    model =  UNetResNet50(num_classes=2).to(device=device)
    ckpt = torch.load('checkpoints/dino/model_epoch20.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    frame_cnt = 0
    transform_fn = getTransform()
    ema_mask = None  # For temporal smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            
            # Ensure frame has the correct size
            original_height, original_width = frame.shape[:2]

            # Run inference on the frame
            result, img = infer_single_image(model, transform_fn, device, frame)

            LABEL_COLORS_LIST = [
                [0, 0, 0],      # background (black)
                [255, 0, 0]     # road (red)
            ]

            # Ensure the processed mask matches the frame size
            processed_mask = process_inference_mask(result, kernel, min_area=200)
            processed_mask, ema_mask = ema_smoothing(processed_mask, ema_mask, alpha=0.4)
            segmentation_map = draw_segmentation_map(processed_mask, LABEL_COLORS_LIST)
            
            # Ensure the output image size matches the input frame size
            overlay_image = image_overlay(img, segmentation_map)

            # Resize to match the original frame size, if needed
            if overlay_image.shape[:2] != (original_height, original_width):
                overlay_image = cv2.resize(overlay_image, (original_width, original_height))
            
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            # Write the frame to the output video
            out.write(overlay_image)
        
            frame_cnt += 1
            print(f"Processed frame {frame_cnt}")

    finally:
        # Release resources
        cap.release()
        out.release()
        print("Video processing completed.")