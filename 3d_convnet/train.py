from tqdm import tqdm
import torch
from .dataset import get_loaders_for_training
import torch.nn as nn
from .model import TemporalResNet, WeightedMSE
import os
import matplotlib.pyplot as plt
from ..utils import get_torch_device, load_config

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

def train_validate(train_loader, val_loader, optimizer, model, device, criterion,train_params, epochs=10, 
                   training_losses = None, val_losses = None, save_every=2, save_dir='checkpoints/'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if training_losses is None: training_losses = []
    if val_losses is None: val_losses = []
    
    for epoch in range(epochs): 
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train   = 0

        for (batch_x, batch_y) in tqdm(train_loader, desc=f'Training {epoch+1}/{epochs}:', ncols=100):

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, pred_labels = predictions.max(dim=1)
            correct_train += (pred_labels == batch_y).sum().item()
            total_train += batch_y.size(0)

        avg_train_loss = running_train_loss / len(train_loader)

        training_losses.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.4f}")

        #validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val   = 0
        for (batch_x, batch_y) in tqdm(val_loader, desc=f'Val {epoch+1}/{epochs}:', ncols=100):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with torch.no_grad():
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

            running_val_loss += loss.item()
            _, pred_labels = predictions.max(dim=1)
            correct_val += (pred_labels == batch_y).sum().item()
            total_val += batch_y.size(0)

        avg_val_loss = running_val_loss / len(val_loader)

        val_losses.append(avg_val_loss)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if (epoch+1) % save_every == 0:

            checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'training_losses': training_losses,
            'val_losses': val_losses,
            'train_params': train_params,
            }

            model_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pth')
            torch.save(checkpoint, model_path)
            print(f"Checkpoint saved to {save_dir}\n")

    return training_losses, val_losses

if __name__ == '__main__':

    device = get_torch_device(dont_use_mps=True)
    config = load_config(config_path='project_src/3d_convnet_config.json')

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

    model = TemporalResNet(in_channels=1, height=123, width=455).to(device=device)
    if config['criterion']=='mse':
        criterion = nn.MSELoss()
    elif config['criterion']=='weighted_mse':
        criterion = WeightedMSE(alpha=config['alpha'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_validate(train_loader, val_loader, optimizer, model, device, criterion, train_params=config,
                   epochs=config['epochs'])