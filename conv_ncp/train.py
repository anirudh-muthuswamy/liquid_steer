import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.mps as mps
import torch.backends.cudnn as cudnn
import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import ConvNCPModel, WeightedMSE
from dataset import get_loaders_for_training

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

def overlay_visual_backprop(input_tensor, mask, save_path=None, alpha=0.1):
    """
    Overlays the visual backprop mask on the original input image.

    Args:
        input_tensor: [3, H, W] torch.Tensor (before batch dimension), normalized
        mask: [H, W] numpy array, already normalized [0, 1]
        save_path: optional path to save overlay image
        alpha: blending factor (heatmap vs original)
    """
    #denormalize image (undo mean/std normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = input_tensor.clone().detach().cpu().numpy() 
    img = img * std[:, None, None] + mean[:, None, None]
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))  # -> [H, W, 3]

    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    #colormap to mask and overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlayed = (1 - alpha) * img + alpha * heatmap
    overlayed = np.clip(overlayed, 0, 1)

    plt.imshow(overlayed)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def train_validate(train_loader, val_loader, optimizer, model, criterion, train_params, current_epoch=0, epochs=10, 
                   save_dir = 'checkpoints/', training_losses = [], validation_losses = [], save_every=10):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    backprop_save_dir = os.path.join(save_dir,'backprops')
    if not os.path.exists(backprop_save_dir):
        os.makedirs(backprop_save_dir)

    this_run_epoch = 0
    for epoch in range(current_epoch, epochs): 
        model.train()
        total_train_loss = 0.0
        for _, (_, _, batch_x, batch_y) in tqdm(enumerate(train_loader), 
                                          desc=f'Training {epoch+1}/{epochs}:', total=len(train_loader), ncols=100):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        training_losses.append(total_train_loss/len(train_loader))

        print(f"Train Loss: {total_train_loss/len(train_loader)}")

        #validation loop
        model.eval()
        total_val_loss = 0.0
        for _, (_, _, batch_x, batch_y) in tqdm(enumerate(val_loader), 
                                          desc=f'Validation {epoch+1}/{epochs}:', total=len(val_loader), ncols=100):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with torch.no_grad():
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

            total_val_loss += loss.item()

        validation_losses.append(total_val_loss/ len(val_loader))

        print(f"Validation Loss: {total_val_loss / len(val_loader)}")
        
        # visualbackprop dump
        with torch.no_grad():
            model.eval()
            batch = next(iter(train_loader))
            _, _, batch_x, _ = batch
            batch_x = batch_x.to(device)

            B, T, C, H, W = batch_x.shape
            x_flat = batch_x.view(B*T, C, H, W)

            _ = model.conv_head(x_flat)  # intermediate activations
            vis_mask = model.conv_head.visual_backprop(idx=0)
            input_image = x_flat[0]  # one image: shape [3, H, W]
            overlay_visual_backprop(input_image, vis_mask, save_path=f'{backprop_save_dir}/epoch_{epoch+1}.png', alpha=0.5)
            plt.close()

        this_run_epoch += 1
        if this_run_epoch % save_every  == 0:

            checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'train_params': train_params,
            }

            model_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pth')
            torch.save(checkpoint, model_path)
            print(f"Checkpoint saved to {save_dir}\n")

    return model_path

def init_weights_he(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def load_config(config_path='./project_src/conv_ncp_config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

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

    config = load_config()

    train_loader, val_loader = get_loaders_for_training(
    # Preprocessing args:
    data_dir=config["data_dir"], steering_angles_path=config["steering_angles_txt_path"], save_dir=config["csv_save_dir"],
    filter=config["filter"], norm=config["norm"], turn_threshold=config["turn_threshold"], buffer_before=config["buffer_before"],
    buffer_after=config["buffer_after"], train_size=config["train_size"],

    # Dataset args:
    imgh=config["imgh"], imgw=config["imgw"], step_size=config["step_size"], seq_len=config["seq_len"], crop=config["crop"],

    # Dataloader args:
    batch_size=config["batch_size"], prefetch_factor=config["prefetch_factor"], num_workers=config["num_workers"], 
    pin_memory=config["pin_memory"], train_shuffle=config["train_shuffle"])

    
    # Assuming extracted features from conv head (8*4) are 32-dimensional
    model = ConvNCPModel(num_filters=8, features_per_filter=config['feat_per_filt'], inter_neurons = 12,
                        command_neurons = 6, motor_neurons = 1, sensory_fanout = 6, inter_fanout = 4, 
                        recurrent_command_synapses = 6, motor_fanin = 6, seed = 20190120) 
    
    if config['he_init']:
        model.apply(init_weights_he)
        
    model = model.to(device)

    # Define loss function and optimizer
    criterion = WeightedMSE(config['alpha'])
    optimizer = optim.Adam([
    # Convolutional head
    {'params': model.conv_head.parameters(), 'lr': config['conv_head_lr']},
    # NCP/LTC
    {'params': model.ltc.parameters(), 'lr': config['ncp_lr']},
    # Output layer
    {'params': model.fc_out.parameters(), 'lr': config['ncp_lr']}], 
        betas=config['optim_betas'])

    if config['load_from_ckpt']:
        model_ckpt = torch.load(config['ckpt_path'], map_location=device)
        model.load_state_dict(model_ckpt['model_state_dict'])
        optimizer.load_state_dict(model_ckpt['optimizer_state_dict'])
        current_epoch = model_ckpt['epoch']
        training_losses = model_ckpt['training_losses']
        validation_losses = model_ckpt['validation_losses']
        loaded_train_params = model_ckpt['train_params']
        print('checkpoint loaded successfully!')

    else:
            current_epoch = 0
            training_losses = []
            validation_losses = []
    
    if len(training_losses) > 0:
        print("last training and validation losses:", training_losses[-1], validation_losses[-1])
    else:
        print('Training losses:', training_losses)
        print('Validation losses:', validation_losses)
    print('Current Epoch Number:', current_epoch)

    final_model_path = train_validate(train_loader=train_loader,
          val_loader=val_loader,
          optimizer=optimizer,
          model=model,
          train_params=config,
          criterion=criterion,
          current_epoch=current_epoch,
          epochs=config['epochs'], 
          save_dir=config['ckpt_save_dir'],
          training_losses=training_losses,
          validation_losses=validation_losses,
          save_every=config['save_every'])
    
    final_checkpoint = torch.load(final_model_path)

    training_losses = final_checkpoint['training_losses']
    validation_losses = final_checkpoint['validation_losses']

    plot_loss_accuracy(training_losses, validation_losses, save_dir=config['ckpt_save_dir'])