import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from networks import fcn_isotropic
from helper_function import NormalizeData, DomainDataset, DataPointsDataset
from torch.utils.data import DataLoader

# Training parameters 
# ----------------------------------------------------------------/
N_epoch=50000
lr=1e-3
batch_size=5000
N_samples = 100000 # Number of the data points for the data loss
#-----------------------------------------------------------------/

# Physical parameters
#-----------------------------------------------------------------/
sampling_time=30.0 # FPS camera recording
height=0.1 # [m] heigh of the sample
width=0.1 # [m] width of the sample 
thickness=0.006 # [m] thickness of sample
#-----------------------------------------------------------------/

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Data preprocessing
#-------------------------------------------------------------------------------------------------------------/
data = np.load(r'F:\Synthetic_data_pulse\param_fbh_size10mm_depth30pct_thickness6mm.npz', allow_pickle=True)
data = np.array(data['data'], dtype=np.float32)
data=torch.from_numpy(data)

p_data=data+273.15 # conversion to Kelvin temperature
p_data=p_data[4:,:,:] # Trimming sequence to 21 [s] of recording

T = p_data          # temperature data, shape (630, 512, 512)
time,vertical,horizontal=T.shape

t = torch.linspace(0, time/sampling_time, time) # [0,T_max]
y = torch.linspace(0, height, vertical) # [0, H]
x = torch.linspace(0, width, horizontal) # [0, W]

# Normalizing the data into region of -1 and 1
t_norm,ub_t,lb_t=NormalizeData(t)
y_norm,ub_y,lb_y=NormalizeData(y)
x_norm,ub_x,lb_x=NormalizeData(x)
T_norm,ub_T,lb_T=NormalizeData(T)

# Maintaining limits for reconstruction of them in PDE loss
# Temperature scaling
T_scale = (ub_T - lb_T) / 2.0
scale_t = (ub_t - lb_t) / 2.0
scale_x = (ub_x - lb_x) / 2.0
scale_y = (ub_y - lb_y) / 2.0
scale_z = (thickness) / 2.0 

scales = (T_scale, scale_y, scale_x, scale_t, scale_z)

# Meshgrid
t_grid, y_grid, x_grid = torch.meshgrid(t_norm, y_norm, x_norm, indexing='ij')

# Create mask of valid points (True = valid, False = corrupted)
mask = torch.ones_like(T, dtype=torch.bool)
mask[:, 150:250, 150:250] = False  # exclude this region

# Apply mask to coordinates and data
t_valid = t_grid[mask]
y_valid = y_grid[mask]
x_valid = x_grid[mask]
T_valid = T_norm[mask]

# Concatenate into coordinate tensor
coordis_valid = torch.stack([t_valid, y_valid, x_valid], dim=1)

# Subsampling data points for the data loss
idx = torch.randperm(coordis_valid.shape[0])[:N_samples]

# We have 100_000 randomly selected data points for the data loss which are normalized all between -1 and 1 
X_train = coordis_valid[idx].float()
y_train = T_valid[idx].float().unsqueeze(1)

data_dataset = DataPointsDataset(X_train, y_train)
data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)

#-------------------------------------------------------------------------------------------------------------/
# Data processing for the PDE loss

phys_dataset = DomainDataset(n_samples=N_samples, n_dim=3, method='lhs')
phys_loader = DataLoader(phys_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)

#-------------------------------------------------------------------------------------------------------------/

# Parameters determining network architecture
layers_temp = np.array([3,30,30,30,30,30,30,30,30,1]) # 8 hidden layers
layer_lap=np.array([3,20,20,1]) # 2 hidden layers

# Network definition
PINN=fcn_isotropic(layers_temp,layer_lap)
PINN=PINN.to(device)

# Optimizer
optimizer=optim.Adam(PINN.parameters(),lr=lr)

#-------------------------------------------------------------------------------------------------------------/
# Training loop
epoch_avg_losses=[]
data_avg_losses=[]
physics_avg_losses=[]

n_batches = min(len(data_loader), len(phys_loader))

for epoch in range(N_epoch):
    PINN.train()
    running_total = 0.0
    running_data = 0.0
    running_phys = 0.0

    for (X_data_batch, y_data_batch), X_phys_batch in zip(data_loader, phys_loader):
        X_data_batch = X_data_batch.to(device)
        y_data_batch = y_data_batch.to(device)
        X_phys_batch = X_phys_batch.to(device)


        optimizer.zero_grad()

        loss,loss_u,loss_f=PINN.loss(X_data_batch,y_data_batch,X_phys_batch,scales)
        loss.backward()
        optimizer.step()

        running_total += loss.item()
        running_data += loss_u.item()
        running_phys += loss_f.item()

    # average over batches in this epoch
    avg_total = running_total / n_batches
    avg_data = running_data / n_batches
    avg_phys = running_phys / n_batches

    epoch_avg_losses.append(avg_total)
    data_avg_losses.append(avg_data)
    physics_avg_losses.append(avg_phys)

    if (epoch % 10 == 0) or (epoch == N_epoch-1):
            print(f"Epoch {epoch:04d}  Total={avg_total:.6e}  Data={avg_data:.6e}  Phys={avg_phys:.6e}  a={PINN.a.item():.3e}")

if len(epoch_avg_losses) > 0:
    print(f"Total_loss at start: {epoch_avg_losses[0]:.6e} | Total_loss at end: {epoch_avg_losses[-1]:.6e}")
    print(f"Data_loss  at start: {data_avg_losses[0]:.6e} | Data_loss  at end: {data_avg_losses[-1]:.6e}")
    print(f"Phys_loss  at start: {physics_avg_losses[0]:.6e} | Phys_loss  at end: {physics_avg_losses[-1]:.6e}")
else:
    print("No logged losses.")


torch.save(PINN.state_dict(), "model_weights.pth")
torch.save(epoch_avg_losses,"Total_loss.pth")
torch.save(data_avg_losses,"Data_loss.pth")
torch.save(physics_avg_losses,"Physic_loss.pth")


print(f"Learned diffusivity a = {PINN.a.item():.6e} [m^2/s] ")
