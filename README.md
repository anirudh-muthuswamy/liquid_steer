# Comparative Analysis of Temporal Deep Learning Models for Steering Angle Prediction

This project provides a comparative evaluation of three temporal deep learning models for vehicle steering angle prediction using video data. The models implemented include:

1. **Neural Circuit Policies (NCP)** using Liquid Time-Constant (LTC) networks
2. **Spatio-Temporal LSTM Network** (ConvLSTM-based)
3. **Temporal Residual Network** with 3D Convolutions

Each model is implemented in its own directory with associated training and inference scripts, along with model-specific configuration files and best-performing weights.

(All models can be run on CUDA devices, whereas when using an MPS device, Conv3D for the LSTM and Temporal Residual Network is not implemented, hence it should be run on CPU)

The dataset used can be found here: https://github.com/SullyChen/driving-datasets
---

## Project Structure

```
project_src/
├── 3d_convnet/               # Temporal residual network with 3D convolutions
│   ├── train.py
│   └── inference.py
├── conv_lstm/                # Spatio-temporal LSTM network
│   ├── train.py
│   └── inference.py
├── conv_ncp/                 # Neural Circuit Policy (NCP) with LTC
│   ├── train.py
│   └── inference.py
├── best_model_weights/      # Stores best weights for all models
├── kaggle_notebooks/        # Jupyter notebooks used for experimentation
├── full_environment.yml     # Conda environment specification
├── 3d_convnet_config.json
├── conv_lstm_config.json
├── conv_ncp_config.json
└── README.md

Outside project_src/
├── checkpoints/             # [USER CREATED] Directory to save checkpoints
├── predictions/             # [USER CREATED] Directory to save prediction outputs
└── data/                    # [USER CREATED] Input data directory
```

---

## Environment Setup

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

To create and activate the conda environment:

```bash
conda env create -f full_environment.yml
conda activate steering_env
```

---

## Running the Models

### Setup

Before running any model, make sure these directories exist in the root of the project (outside the project_src directory):

```bash
mkdir -p checkpoints predictions data
```

### Training

Train each model by executing the following commands from outside the project_src directory :

```bash
# Train Temporal Residual Network (3D ConvNet)
python -m project_src.3d_convnet.train

# Train Spatio-Temporal LSTM
python -m project_src.conv_lstm.train

# Train NCP (Liquid Time-Constant)
python -m project_src.conv_ncp.train
```

Each model uses its own `*_config.json` file in the root directory to manage hyperparameters.

### Inference

Perform inference with the best model weights:

```bash
# Inference with Temporal Residual Network
python -m project_src.3d_convnet.inference

# Inference with Spatio-Temporal LSTM
python -m project_src.conv_lstm.inference

# Inference with NCP Model
python -m project_src.conv_ncp.inference
```

The `dataset.py` and `model.py` files can also be called similarly


## Checkpoints and Results

- Trained model weights are stored in `best_model_weights/`
- Output predictions are saved to `predictions/`
- Training checkpoints are saved to `checkpoints/`

---

## Config Files

Hyperparameters and training settings can be edited in the following files:

- `3d_convnet_config.json`
- `conv_lstm_config.json`
- `conv_ncp_config.json`


