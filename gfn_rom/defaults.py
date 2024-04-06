import torch

# --- PLOTTING ---

# default params for matplotlib
plt_params = {'legend.fontsize': 'x-large',
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}

# --- PREPROCESSING ---

# train/test split rate
rate = 30

# seeds for reproducibility
# seed to split dataset into train/test
split_seed = 10
# seed for GFN-ROM initialisation
seed = 10

# select precision
# (explanation for when this can matter at https://blog.demofox.org/2017/11/21/floating-point-precision/)
precision = torch.float64

# --- GFN-ROM ---

# Latent dimension
latent_size = 200

# Mapper sizes
# Mapper maps from parameters to latent dimension
# We optionally allow the addition of further layers
mapper_sizes = [50, 50, 50, 50]

# Autoencoder sizes
# Autoencoder maps from full data to latent representation
# We optionally allow the addition of further layers
ae_sizes = []

# Activation
act = torch.nn.Tanh

# Use either fixed ("fixed"), adaptive ("adapt") or precomputed adaptive method ("preadapt")
mode = 'fixed'

# Weight to give to the mapper loss compared to the autoencoder loss
mapper_weight = 10.0

# Number of epochs
epochs = 5000

# Learning rate
lr = 0.001

# L2 regularisation hyperparamter
lambda_ = 10**-5

# --- RBNICS ---

# N_basis = N_basis_factor * n_params
N_basis_factor = 1.5
# Tolerance for POD-Galerkin
tol = 0 # stopping criteria