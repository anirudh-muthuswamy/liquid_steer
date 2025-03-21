
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.mps as mps
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import NCPModel
from dataset import (df_split_train_val, create_train_val_dataset, create_train_val_loader,
                     calculate_mean_and_std)
from check_data import (get_full_image_filepaths, get_steering_angles,
                        convert_to_df, filter_df_on_turns, group_data_by_sequences)

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

def train(train_loader, val_loader, optimizer, model, criterion, epochs=10, save_dir = 'checkpoints/'):

    training_losses = []
    validation_losses = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs): 
        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(train_loader), desc='Training', total=len(train_loader), ncols=50):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item()}", end='')

        #validation loop
        model.eval()
        total_loss = 0
        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(val_loader), desc='Validation', total=len(val_loader), ncols=50):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            total_loss += loss.item()

    print(f" Validation Loss: {total_loss / len(val_loader)}")

    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch + 1, 
    'training_losses': training_losses,
    'validation_losses': validation_losses,
    }

    model_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pth')
    torch.save(checkpoint, model_path)
    print(f"Checkpoint saved to {save_dir}\n")

    model.train()  # Set model back to training mode

    return model_path

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

    #data related parameters
    data_dir = 'sullychen/07012018/data'
    steering_angles_txt_path = 'sullychen/07012018/data.txt'
    train_dataset_path = 'train_data_filtered.csv'
    val_dataset_path = 'val_data_filtered.csv'
    train_size = 0.8
    seq_len = 64
    imgw = 224
    imgh = 224
    # mean, std = calculate_mean_and_std(data_dir)
    mean = [0.543146, 0.53002986, 0.50673143]
    std = [0.23295668, 0.22123158, 0.22100357]
    batch_size = 8
    shuffle = False

    img_paths = get_full_image_filepaths(data_dir)
    steering_angles = get_steering_angles(steering_angles_txt_path)

    data_pd = convert_to_df(img_paths, steering_angles)
    data_pd_filtered = filter_df_on_turns(data_pd)
    data_pd_filtered = group_data_by_sequences(data_pd_filtered)



    train_dataset_path, val_dataset_path = df_split_train_val(data_pd_filtered, train_dataset_path, 
                                                                val_dataset_path, train_size)
    
    train_dataset , val_dataset = create_train_val_dataset(train_csv_file = train_dataset_path,
                                                             val_csv_file = val_dataset_path,
                                                             seq_len=seq_len, imgw=imgw, imgh=imgh, mean=mean)
    
    train_loader, val_loader = create_train_val_loader(train_dataset, val_dataset,
                                                         batch_size=batch_size, shuffle=shuffle)
    
    # Assuming extracted features are 32-dimensional
    model = NCPModel(num_filters=8, features_per_filter=4, inter_neurons = 12, command_neurons = 6,
                     motor_neurons = 1, sensory_fanout = 6, inter_fanout = 4, 
                     recurrent_command_synapses = 6, motor_fanin = 6, seed = 20190120, device=device) 
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    final_model_path = train(train_loader=train_loader,
          val_loader=val_loader,
          optimizer=optimizer,
          model=model,
          criterion=criterion,
          epochs=100)
    
    checkpoint = torch.load(final_model_path)

    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']

    plot_loss_accuracy(training_losses, validation_losses)