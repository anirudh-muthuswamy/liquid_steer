import torch
import torch.nn as nn
from ncps.wirings import NCP
from ncps.torch import LTC
import torch.nn.functional as F

class WeightedMSE(nn.Module):
    def __init__(self, alpha=0.1):
        super(WeightedMSE, self).__init__()
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        # Calculate the squared error
        squared_error = (predictions - targets)**2
        
        # Calculate the weighting factor: w(y) = exp(|y|*alpha)
        weights = torch.exp(torch.abs(targets) * self.alpha)
        
        # Apply the weights to the squared error
        weighted_loss = squared_error * weights
        
        # Return the mean of the weighted loss
        return weighted_loss.mean()

class convolutional_head(nn.Module):
    def __init__(self, num_filters = 8, features_per_filter = 4):
        super(convolutional_head, self).__init__()

        self.num_filters = num_filters
        self.features_per_filter = features_per_filter

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully Connected Layers to extract features per filter
        self.fc_layers = nn.ModuleList([
            nn.Linear(28 * 28, features_per_filter) for _ in range(num_filters)
        ])

    def forward(self, x):
        """
        Forward pass of the convolutional head.

        :param x: Input tensor of shape [batch, channels, height, width]
        :return: Feature vector of shape [batch, num_filters * features_per_filter]
        """
        batch_size = x.shape[0]

        # Apply convolutional layers
        x = self.conv_layers(x)  # Shape: [batch, num_filters, height, width] -> [512, 8, 28, 28] for seq len of 64 and batch size of 8

        # Extract individual filter outputs
        filter_outputs = torch.split(x, 1, dim=1)  # Splitting along channel dimension -> len of 8 for num_filters = 8
        #shape of a single filter_output -> [512, 1, 28, 28]

        feature_vectors = []
        for i in range(self.num_filters):
            filter_out = filter_outputs[i].view(batch_size, -1)  # Flatten each filter output -> shape of: [512, 784]
            feature_vec = F.relu(self.fc_layers[i](filter_out))  # Apply FC layer -> shape of: [512, 4]
            feature_vectors.append(feature_vec)

        # Concatenate feature vectors
        feature_layer = torch.cat(feature_vectors, dim=1)  # Shape: [batch, num_filters * features_per_filter]
        # shape of:[512, 32]

        return feature_layer
    
class ConvNCPModel(nn.Module):
    def __init__(self, num_filters=8, features_per_filter=4, 
                 inter_neurons = 12, command_neurons = 6, motor_neurons = 1, 
                 sensory_fanout = 6, inter_fanout = 4, recurrent_command_synapses = 6,
                 motor_fanin = 6, seed = 20190120):
        super(ConvNCPModel, self).__init__()

        # Define NCP wiring based on CommandLayerWormnetArchitectureMK2 parameters
        wiring = NCP(
            inter_neurons=inter_neurons,   # Number of interneurons
            command_neurons=command_neurons,  # Number of command neurons
            motor_neurons=motor_neurons,    # Output neurons (1 for steering)
            sensory_fanout=sensory_fanout,   # Number of interneurons each sensory neuron connects to
            inter_fanout=inter_fanout,     # Number of command neurons each interneuron connects to
            recurrent_command_synapses=recurrent_command_synapses,  # Recurrent connections in the command layer
            motor_fanin=motor_fanin,      # Number of command neurons each motor neuron connects to
            seed=seed       # Random seed for reproducibility
        )

        self.conv_head = convolutional_head(num_filters, features_per_filter)

        # Define the LTC with the correct input size
        self.ltc = LTC(
            input_size=num_filters * features_per_filter,  # This should match the Convolutional Head output
            units=wiring,
            return_sequences=True
        )

        # Fully connected layer to map motor neuron output to a steering angle
        self.fc_out = nn.Linear(wiring.output_dim, 1)

    def forward(self, x):
        """
        Forward pass: Conv Head → LTC-NCP → Fully Connected.
        :param x: Input shape [batch, seq_len, channels, height, width]
        :return: Steering angles [batch, seq_len]
        """
        batch_size, seq_len, c, h, w = x.size()

        # Flatten batch and sequence for convolutional processing
        x = x.view(batch_size * seq_len, c, h, w)

        # Extract features using Convolutional Head
        features = self.conv_head(x)  # Shape: [batch * seq_len, feature_dim]

        # Reshape back to [batch, seq_len, feature_dim] for LTC
        features = features.view(batch_size, seq_len, -1)

        # Pass through LTC
        outputs, _ = self.ltc(features) # shape of -> [batch, seq_len, 1]

        # Map NCP output to steering angle
        predictions = self.fc_out(outputs) # shape of -> [batch, seq_len, 1]
        return predictions.squeeze(-1)  # Shape: [batch, seq_len]