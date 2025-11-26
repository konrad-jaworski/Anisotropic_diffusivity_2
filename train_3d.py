import torch
import torch.optim as optim
import numpy as np
from networks import FCN
from helper_function import DomainDataset,DataGeneration
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader

# Setting device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# /--------------------------------------------------------------------/
# Training parameters
N_epoch=100
lr=1e-3 

# Parameters for the grad norm moving averages
lr2=1e-2
alpha=0.26
gradnorm_mode=True
alpha_ema=0.01
ema_losses=None

z=0.5

N_interior=60000
N_ic=20000
N_bc=5000 # Sampled from each of the boundary produce 4*N_bc samples
N_coll=100000

sampling_mode=0 # Depending whether 0 or 1 we sampled data once or every iteration 0-sampled once
decay_mode=0 # Activated exponetial decay of the lr parameter during training 0-no decay
decay_every=500 # Frequency of exponential decay

batch_size=1024
# /--------------------------------------------------------------------/
# Data preparation

data_path=r'/Volumes/KINGSTON/Synthetic_data_no_defect/2025_10_24_sample_100x100x5mm_no_defect_isotropic_gaussian_heat.npz'
data=np.load(data_path,allow_pickle=True)
data_cube = data['data'][34:, :, :]  # shape [T, Y, X]

# Sampling randomly data from the data domain
d_operator=DataGeneration(data_cube,N_interior,N_ic,N_bc)

# Producing collocation points for the PDE loss
coll_data=DomainDataset(n_samples=N_coll,n_dim=4,method='lhs')

if sampling_mode==0:
    X_coll=coll_data.resample()
    X_data,Y_data=d_operator.generate() # shape [N,4] -> [T,Y,X,Z]
    
    dataset = TensorDataset(X_coll, X_data, Y_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
# Definition of the network
layers=[4,100,100,100,100,1]
PINN=FCN(layers)
PINN=PINN.to(device) 

# Layer sele
shared_layer=list(PINN.linears[-2].parameters())[0]
run_iter=0

# Logging
log_weights_1=[]
log_weights_2=[]
log_loss_total=[]
log_loss_data=[]
log_loss_phys=[]
log_a=[]


# Setting up the optimizer
optimizer_1=optim.Adam(PINN.parameters(),lr=lr) # Optimizer for the network

if decay_mode==1:
    scheduler = ExponentialLR(optimizer_1, gamma=0.95)

PINN.train()
for epoch in tqdm(range(N_epoch)):
    if sampling_mode==1:
        X_coll=coll_data.resample()
        X_data,Y_data=d_operator.generate() # shape [N,4] -> [T,Y,X,Z]
        dataset = TensorDataset(X_coll, X_data, Y_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
       
    for x_coll_batch,x_data_batch,y_data_batch in loader:
        