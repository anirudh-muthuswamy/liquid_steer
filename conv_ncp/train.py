import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.mps as mps
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import ConvNCPModel, WeightedSteeringLoss
from dataset import (df_split_train_val, create_train_val_dataset, create_train_val_loader)
from check_data import get_preprocessed_data_pd
from argparse import ArgumentParser

def plot_loss_accuracy(train_loss, val_loss):
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
    plt.show()

def train_validate(train_loader, val_loader, optimizer, model, criterion, train_params, current_epoch=0, epochs=10, 
                   save_dir = 'checkpoints/', training_losses = [], validation_losses = [], save_every=10):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

        this_run_epoch += 1

        print('save_every:', save_every)
        print('this_run_epoch:', this_run_epoch)
        print('save_every % this_run_epoch', this_run_epoch %  save_every)

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

def create_parser():
    parser = ArgumentParser(description='Training parameters for the model')
    
    # Data related parameters
    parser.add_argument('--load_from_ckpt', action='store_true', default=False,
                        help='Whether to load from checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints/conv_ncp/sl_32_ss_32_bs16_weighted',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='checkpoints/conv_ncp/sl_32_ss_32_bs16_weighted/model_epoch10.pth',
                        help='Path to the checkpoint to load from')
    parser.add_argument('--train_dataset_path', type=str, default='data/csv_files/train_ncp_data_filtered.csv',
                        help='Path to the training dataset CSV')
    parser.add_argument('--val_dataset_path', type=str, default='data/csv_files/val_ncp_data_filtered.csv',
                        help='Path to the validation dataset CSV')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='Proportion of data to use for training')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='Sequence length')
    parser.add_argument('--step_size', type=int, default=32,
                        help='Step size')
    parser.add_argument('--imgw', type=int, default=224,
                        help='Image width')
    parser.add_argument('--imgh', type=int, default=224,
                        help='Image height')
    parser.add_argument('--mean', type=float, nargs=3, 
                        default=[0.543146, 0.53002986, 0.50673143],
                        help='Mean values for normalization')
    parser.add_argument('--std', type=float, nargs=3, 
                        default=[0.23295668, 0.22123158, 0.22100357],
                        help='Standard deviation values for normalization')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='Whether to shuffle the data')
    
    # Optimizer parameters
    parser.add_argument('--conv_head_lr', type=float, default=2.5e-5,
                        help='Learning rate for convolutional head')
    parser.add_argument('--ncp_lr', type=float, default=1e-3,
                        help='Learning rate for NCP')
    parser.add_argument('--optim_betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Beta parameters for Adam optimizer')
    return parser

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

    parser = create_parser()
    args = parser.parse_args()

    #data related parameters
    load_from_ckpt = args.load_from_ckpt
    save_dir = args.save_dir
    checkpoint_path = args.checkpoint_path
    train_dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path

    train_params = {
    'train_size': args.train_size,
    'seq_len': args.seq_len,
    'step_size': args.step_size,
    'imgw': args.imgw,
    'imgh': args.imgh,
    'mean': args.mean,
    'std': args.std,
    'batch_size': args.batch_size,
    'shuffle': args.shuffle,
    'conv_head_lr': args.conv_head_lr,
    'ncp_lr': args.ncp_lr,
    'optim_betas': args.optim_betas}

    train_dataset , val_dataset = create_train_val_dataset(train_csv_file = train_dataset_path,
                                                             val_csv_file = val_dataset_path,
                                                             seq_len=train_params['seq_len'], 
                                                             step_size=train_params['step_size'],
                                                             imgw=train_params['imgw'], imgh=train_params['imgh'], 
                                                             mean=train_params['mean'])
    
    train_loader, val_loader = create_train_val_loader(train_dataset, val_dataset,
                                                         batch_size=train_params['batch_size'], 
                                                         shuffle=train_params['shuffle'])
    
    # Assuming extracted features from conv head (8*4) are 32-dimensional
    model = ConvNCPModel(num_filters=8, features_per_filter=4, inter_neurons = 12, command_neurons = 6,
                     motor_neurons = 1, sensory_fanout = 6, inter_fanout = 4, 
                     recurrent_command_synapses = 6, motor_fanin = 6, seed = 20190120) 
    model = model.to(device)

    # Define loss function and optimizer
    criterion = WeightedSteeringLoss(alpha=0.1)
    optimizer = optim.Adam([
    # Convolutional head
    {'params': model.conv_head.parameters(), 'lr': train_params['conv_head_lr']},
    # NCP/LTC
    {'params': model.ltc.parameters(), 'lr': train_params['ncp_lr']},
    # Output layer
    {'params': model.fc_out.parameters(), 'lr': train_params['ncp_lr']}], 
        betas=train_params['optim_betas'])

    if load_from_ckpt:
        model_ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(model_ckpt['model_state_dict'])
        optimizer.load_state_dict(model_ckpt['optimizer_state_dict'])
        current_epoch = model_ckpt['epoch']
        training_losses = model_ckpt['training_losses']
        validation_losses = model_ckpt['validation_losses']
        loaded_train_params = model_ckpt['train_params']

        assert loaded_train_params == train_params

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
          train_params=train_params,
          criterion=criterion,
          current_epoch=current_epoch,
          epochs=10, 
          save_dir=save_dir,
          training_losses=training_losses,
          validation_losses=validation_losses)
    
    final_checkpoint = torch.load(final_model_path)

    training_losses = final_checkpoint['training_losses']
    validation_losses = final_checkpoint['validation_losses']

    plot_loss_accuracy(training_losses, validation_losses)