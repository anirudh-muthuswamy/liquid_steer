import torch
import torch.nn as nn
from ncps.wirings import NCP
from ncps.torch import LTC
import torch.nn.functional as F
from torchinfo import summary

class WeightedMSE(nn.Module):
    def __init__(self, alpha=0.1):
        super(WeightedMSE, self).__init__()
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        # squared error
        squared_error = (predictions - targets)**2
        
        # weighting factor: w(y) = exp(|y|*alpha)
        weights = torch.exp(torch.abs(targets) * self.alpha)
        weighted_loss = squared_error * weights

        return weighted_loss.mean()

class convolutional_head(nn.Module):
    def __init__(self, num_filters = 8, features_per_filter = 16):
        super(convolutional_head, self).__init__()

        self.num_filters = num_filters
        self.features_per_filter = features_per_filter

        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, num_filters, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        # FC to extract features per filter
        self.fc_layers = nn.ModuleList([
            nn.Linear(28 * 28, features_per_filter) for _ in range(num_filters)
        ])

        self.activations = []
        self.feature_layer = None

    def forward(self, x):
        """
        Forward pass of the convolutional head.

        :param x: Input tensor of shape [batch, channels, height, width]
        :return: Feature vector of shape [batch, num_filters * features_per_filter]
        """

        self.activations = []
        batch_size = x.shape[0]

        # apply conv layers
        # [batch, num_filters, height, width] -> [512, 8, 28, 28] for seq len of 64 and batch size of 8 (64*8 = 512)

        x = self.relu(self.conv1(x)); self.activations.append(x)
        x = self.relu(self.conv2(x)); self.activations.append(x)
        x = self.relu(self.conv3(x)); self.activations.append(x)
        x = self.relu(self.conv4(x)); self.activations.append(x)
        x = self.relu(self.conv5(x)); self.activations.append(x)

        # individual filter outputs
        filter_outputs = torch.split(x, 1, dim=1)  # splitting along channel dimension -> len of 8 for num_filters = 8
        #shape of a single filter_output -> [512, 1, 28, 28]

        feature_vectors = []
        for i in range(self.num_filters):
            filter_out = filter_outputs[i].view(batch_size, -1)  # flatten each filter output -> shape of: [512, 784]
            feature_vec = F.relu(self.fc_layers[i](filter_out))  # apply FC layer -> shape of: [512, 4]
            feature_vectors.append(feature_vec)

        # concat feature vectors
        feature_layer = torch.cat(feature_vectors, dim=1)  # [batch, num_filters * features_per_filter]
        # [512, 32]
        self.feature_layer = feature_layer

        return feature_layer
    
    def visual_backprop(self, idx=0):
        """
        VisualBackprop-like mask computation using torch (GPU compatible).
        Returns: [H, W] attention mask (still returned as a CPU numpy array).
        """
        # mean maps for each layer
        means = []

        for layer_act in self.activations:
            # [B, C, H, W]
            a = layer_act[idx]  # [C, H, W]
            a = a.float()
            per_channel_max = torch.amax(torch.amax(a, dim=1), dim=1) + 1e-6  # [C]
            norm = a / per_channel_max[:, None, None]  # [C, H, W]
            mean_map = norm.mean(dim=0)  # [H, W]
            means.append(mean_map)

        # feature-level activation mask
        feat_layer = self.feature_layer[idx]  # [num_filters * features_per_filter]
        feat_layer = torch.abs(feat_layer).view(self.num_filters, self.features_per_filter)  # [F, P]
        feat_mask = feat_layer.mean(dim=1)  # [F]
        feat_mask = feat_mask / (feat_mask.max() + 1e-6)

        # applies a weighting on last activation map
        mask = means[-1] * feat_mask.mean()  # [H, W]

        # backward pass through mean activations
        for i in range(len(means) - 2, -1, -1):
            larger = means[i]  # [H, W]
            smaller = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=larger.shape, mode='bilinear', align_corners=False)
            smaller = smaller.squeeze()
            mask = larger * smaller

        mask = mask - mask.min()
        mask = mask / (mask.max() + 1e-6)
        return mask.detach().cpu().numpy()

    
class ConvNCPModel(nn.Module):
    def __init__(self, num_filters=8, features_per_filter=4, 
                 inter_neurons = 12, command_neurons = 6, motor_neurons = 1, 
                 sensory_fanout = 6, inter_fanout = 4, recurrent_command_synapses = 6,
                 motor_fanin = 6, seed = 20190120):
        super(ConvNCPModel, self).__init__()

        # Define NCP wiring based on CommandLayerWormnetArchitecture parameters (from NCP Paper)
        wiring = NCP(
            inter_neurons=inter_neurons,   # no. of interneurons
            command_neurons=command_neurons,  # no. of command neurons
            motor_neurons=motor_neurons,    # out neurons (1 for steering)
            sensory_fanout=sensory_fanout,   # no. of interneurons each sensory neuron connects to
            inter_fanout=inter_fanout,     # no. of command neurons each interneuron connects to
            recurrent_command_synapses=recurrent_command_synapses,  # reccurent connections in the command layer
            motor_fanin=motor_fanin,      # no. of command neurons each motor neuron connects to
            seed=seed       # rand seed for reproducibility
        )

        self.conv_head = convolutional_head(num_filters, features_per_filter)

        self.ltc = LTC(
            input_size=num_filters * features_per_filter,  # should match the conv head output
            units=wiring,
            return_sequences=True
        )

        # FC layer to map motor neuron output to a steering angle
        self.fc_out = nn.Linear(wiring.output_dim, 1)

    def forward(self, x):
        """
        Forward pass: Conv Head -> LTC-NCP -> Fully Connected.
        :param x: Input shape [batch, seq_len, channels, height, width]
        :return: Steering angles [batch, seq_len]
        """
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.conv_head(x)  # [batch * seq_len, feature_dim]

        # [batch, seq_len, feature_dim]
        features = features.view(batch_size, seq_len, -1)
        outputs, _ = self.ltc(features) # [batch, seq_len, 1]
        predictions = self.fc_out(outputs) # [batch, seq_len, 1]
        return predictions.squeeze(-1)  # [batch, seq_len]
    
if __name__ == "__main__":
    # random input
    B, S, C, H, W = 2, 16, 3, 224, 224
    dummy = torch.randn(B, S, C, H, W)

    #default model params
    model = ConvNCPModel()
    
    summary(model=model, input_size=dummy.shape)