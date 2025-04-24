import torch
import torch.nn as nn
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

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        # output 4*hidden for the gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x, hc):
        h_cur, c_cur = hc
        # concatenate on channel axis
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )

class STConvLSTM(nn.Module):
    def __init__(
        self, seq_len=3, height=224, width=224, input_channels=3, hidden_channels=8, fc_units=50, dropout=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.height = height
        self.width  = width

        # stack of 4 ConvLSTM layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(input_channels, hidden_channels, (3, 3)),
            ConvLSTMCell(hidden_channels, hidden_channels, (3, 3)),
            ConvLSTMCell(hidden_channels, hidden_channels, (3, 3)),
            ConvLSTMCell(hidden_channels, hidden_channels, (3, 3)),
        ])
        # one BatchNorm3d per layer (treating time as depth)
        self.bns = nn.ModuleList([
            nn.BatchNorm3d(hidden_channels) for _ in range(4)
        ])

        self.conv3d = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=2,
            kernel_size=(3, 3, 3),
            padding=1
        )
        self.pool3d = nn.MaxPool3d((2, 2, 2))

        #flattened size: out_channels * (seq_len//2) * (height//2) * (width//2)
        flat_size = 2 * (seq_len // 2) * (height // 2) * (width // 2)
        self.fc1     = nn.Linear(flat_size, fc_units)
        self.leaky   = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(fc_units, 1)

    def _run_convlstm(self, cell: ConvLSTMCell, bn: nn.BatchNorm3d, x: torch.Tensor) -> torch.Tensor:

        #x: (batch, seq, channels, H, W)
        b, seq, _, h, w = x.size()
        h_t, c_t = cell.init_hidden(b, (h, w))
        outputs = []
        for t in range(seq):
            h_t, c_t = cell(x[:, t], (h_t, c_t))
            outputs.append(h_t)
        x_seq = torch.stack(outputs, dim=1) #(b, seq, hidden, H, W)
        x_bn  = x_seq.permute(0, 2, 1, 3, 4) #(b, hidden, seq, H, W)
        x_bn  = bn(x_bn)
        return x_bn.permute(0, 2, 1, 3, 4) #(b, seq, hidden, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, C, H, W)
        for cell, bn in zip(self.cells, self.bns):
            x = self._run_convlstm(cell, bn, x)

        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = self.pool3d(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.leaky(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    # random input
    B, C, T, H, W = 2, 3, 3, 224, 224
    dummy = torch.randn(B, C, T, H, W)
    model = STConvLSTM(seq_len=3, height=224, width=224, input_channels=3, hidden_channels=8, fc_units=50, dropout=0.5)
    
    summary(model=model, input_size=dummy.shape)

