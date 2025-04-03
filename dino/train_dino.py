
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.mps as mps
import torch.backends.cudnn as cudnn
import torchmetrics
from segmentation_models_pytorch.losses import DiceLoss
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from project_src.dino.IDD_Dataset import IDDDataset
from project_src.dino.dino_unet import UNetResNet50
from torch.utils.data import DataLoader

def visualize_segmentation(images, masks, outputs):
    # Convert tensors to CPU and detach
    images = images.cpu().detach()
    masks = masks.cpu().detach()
    outputs = outputs.cpu().detach()

    # Get predicted class indices
    preds = torch.argmax(outputs, dim=1)

    # Loop through the batch and visualize
    for i in range(min(images.shape[0], 1)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 10))

        # Plot the input image
        axs[0].imshow(images[i].permute(1, 2, 0))
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        # Plot the ground truth mask
        axs[1].imshow(masks[i].reshape(480, 640), cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis('off')

        # Plot the predicted mask
        axs[2].imshow(preds[i], cmap='gray')
        axs[2].set_title("Predicted Mask")
        axs[2].axis('off')

        plt.show()

def train_validate(model, train_loader, val_loader, optimizer, loss_fn, iou_metric, device, 
                   epochs=20, current_epoch=0, save_dir = 'dino_checkpoints/', train_losses = [], val_losses = [],
                   train_ious = [], val_ious = [], visualize=False, num_visualizations=2):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(current_epoch, epochs): 
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        for _, (images, masks) in tqdm(enumerate(train_loader), 
                                          desc=f'Training {epoch+1}/{epochs}:', total=len(train_loader), ncols=100):
            
            images, masks = images.to(device), masks.to(device)            
            outputs = model(images)
            
            loss = loss_fn(outputs, masks.squeeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and IoU
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            running_iou += iou_metric(preds, masks.squeeze(1)).item()

        train_loss = running_loss / len(train_loader)
        train_iou = running_iou / len(train_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train IoU = {train_iou:.4f}")

        model.eval()
        running_val_loss = 0.0
        running_val_iou = 0.0
        visualizations_done = 0

        with torch.no_grad():
            for i, (images, masks) in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)       
                loss = loss_fn(outputs, masks.squeeze(1))
    
                # Accumulate loss and IoU
                running_val_loss += loss.item()

                # Compute IoU
                preds = torch.argmax(outputs, dim=1)
                running_val_iou += iou_metric(preds, masks.squeeze(1)).item()
                
                # Visualization logic
                if visualize and visualizations_done < num_visualizations:

                    visualize_segmentation(images, 
                                            masks, 
                                            outputs)
                    visualizations_done += 1
        
        val_loss = running_val_loss / len(val_loader)
        val_iou = running_val_iou / len(val_loader)

        print(f"          Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}")

        # Loss storing
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # IOU storing
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1, 
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious
        }

        model_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pth')
        torch.save(checkpoint, model_path)
        print(f"Checkpoint saved to {save_dir}\n")

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

    data_dir = 'code_files/IDD_Data'
    load_from_ckpt = False
    checkpoint_path = ''

    train_df = pd.read_csv(os.path.join(data_dir, 'train_IDD.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_IDD.csv'))

    train_dataset = IDDDataset(train_df, aug=True, select_class=[0])
    val_dataset = IDDDataset(val_df, aug=False, select_class=[0])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

    dice_loss = DiceLoss(mode='multiclass')
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss_fn = lambda outputs, targets: dice_loss(outputs, targets) + cross_entropy_loss(outputs, targets)

    # define metrics
    iou_metric = torchmetrics.JaccardIndex(num_classes=2, task="multiclass").to(device=device)
    
    model = UNetResNet50(num_classes=2).to(device=device)
    
    # define optimizer
    optimizer = optim.Adam([dict(params=model.parameters(), lr=1e-4,  weight_decay=1e-5)])

    if load_from_ckpt:
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(model_ckpt['model_state_dict'])
        optimizer.load_state_dict(model_ckpt['optimizer_state_dict'])
        current_epoch = model_ckpt['epoch']
        train_losses = model_ckpt['training_losses']
        val_losses = model_ckpt['validation_losses']
        train_ious = model_ckpt['train_ious']
        val_ious = model_ckpt['val_ious']

        print('checkpoint loaded successfully!')
    else:
            current_epoch = 0
            train_losses = []
            val_losses = []
            train_ious = []
            val_ious = []
    
    if len(train_losses) > 0:
        print("last training and validation losses:", train_losses[-1], val_losses[-1])
    else:
        print('Training losses:', train_losses)
        print('Validation losses:', val_losses)
    print('Current Epoch Number:', current_epoch)

    final_model_path = train_validate(model=model,
        train_loader=train_loader,
          val_loader=val_loader,
          optimizer=optimizer,
          loss_fn=loss_fn,
          iou_metric=iou_metric,
          device=device,
          epochs=20,
          current_epoch=current_epoch,
          save_dir='code_files/dino_checkpoints',
          train_losses=train_losses,
          val_losses=val_losses,
          train_ious=train_ious,
          val_ious=val_ious
    )




