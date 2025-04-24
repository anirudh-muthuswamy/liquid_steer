from tqdm import tqdm
import torch
from .dataset import get_loaders_for_training
import torch.nn as nn
from .model import STConvLSTM, WeightedMSE
import os
from ..utils import get_torch_device, load_config, plot_loss_accuracy

# train and validation per epoch method, with checkpoint saving
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

# the train method uses a conv_lstm_config.json file with all the parameters defined for model creation
# dataloading, training, and saving the checkpoints.

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

    model = STConvLSTM(seq_len=config['seq_len'], height=config['imgh'], width=config['imgw'], input_channels=3, hidden_channels=8, 
                       fc_units=50, dropout=0.5).to(device=device)
    
    if config['criterion']=='mse':
        criterion = nn.MSELoss()
    elif config['criterion']=='weighted_mse':
        criterion = WeightedMSE(alpha=config['alpha'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

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

    training_losses, val_losses = train_validate(train_loader, val_loader, optimizer, model, device, criterion, train_params=config,
                   save_every=config['save_every'], epochs=config['epochs'])
    
    plot_loss_accuracy(training_losses, validation_losses, save_dir=config['ckpt_save_dir'])