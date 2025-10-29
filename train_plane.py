import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from networks import fcn_plane
from helper_function import NormalizeData, DomainDataset, DataPointsDataset
from torch.utils.data import DataLoader
from scipy.stats import qmc

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Training parameters
#--------------------------------------------------------------------/
N_epoch=10000
lr=1e-4
batch_size=100
N_samples = 100000 
#--------------------------------------------------------------------/

# Physical parameters of the sample
#--------------------------------------------------------------------/
sampling_time=30.0 # FPS camera recording
height=0.1 # [m] heigh of the sample
width=0.1 # [m] width of the sample 
thickness=0.005 # [m] thickness of sample

frame_of_max_temp=34 # Frame index corresponding to the maximum temperature, so we only analyze the cooling phase

target_conductivity=2.0 # [W/(m*K)] Target conductivity of isotropic case
Cp=700.0 # [J/(kg*K)] Specific heat of the material
q=1600.0 # [kg/m^3] Density of the material
#--------------------------------------------------------------------/

# Data preprocessing
#--------------------------------------------------------------------/
data = np.load(r'F:\Synthetic_data_no_defect\2025_10_24_sample_100x100x5mm_no_defect_isotropic_gaussian_heat.npz', allow_pickle=True)

# Trimming the data and converting them to Kelvins
data=data['data'][frame_of_max_temp:,:,:]+275.15 # Converting to kelvins
data=torch.from_numpy(data).float()

T_samples,Y_samples,X_samples=data.shape

# Defining the domain
t = torch.linspace(0, T_samples/sampling_time, T_samples) # [0,T_max]
y = torch.linspace(0, height, Y_samples) # [0, H]
x = torch.linspace(0, width, X_samples) # [0, W]

# Max scalling the domain to [0 - 1] regions
t_norm,t_scale=NormalizeData(t)
y_norm,y_scale=NormalizeData(y)
x_norm,x_scale=NormalizeData(x)

t_grid, y_grid, x_grid = torch.meshgrid(t_norm, y_norm, x_norm, indexing='ij')

data_coordis = torch.stack([t_grid.reshape(-1,1),
                              y_grid.reshape(-1,1),
                                x_grid.reshape(-1,1)], dim=1).squeeze(-1)

y=data.flatten().unsqueeze(-1)


# Subsampling the dataset for data loss
idx = torch.randperm(data_coordis.shape[0])[:N_samples]
X_train = data_coordis[idx]
y_train = y[idx]
#--------------------------------------------------------------------/

# Collocation points for the PDE loss 
#--------------------------------------------------------------------/
phys_dataset = DomainDataset(n_samples=N_samples, n_dim=3, method='lhs')
#--------------------------------------------------------------------/


# Initial condition points 
#--------------------------------------------------------------------/
I_temp=data[0,:,:]
I_temp=I_temp.flatten().unsqueeze(-1)

y_grid,x_grid=torch.meshgrid(y_norm,x_norm,indexing='ij')

I_coordis=torch.stack([y_grid.reshape(-1,1),
                       x_grid.reshape(-1,1)],dim=1).squeeze(-1)

initial_time=torch.zeros(I_coordis.size(0),1)
Initial_position=torch.cat([initial_time,I_coordis],dim=1)

idx = torch.randperm(Initial_position.shape[0])[:N_samples]

X_ic=Initial_position[idx]
Y_ic=I_temp[idx]
#--------------------------------------------------------------------/

# Boundary condition points 
#--------------------------------------------------------------------/
# --- Masks for boundaries skipping initial conditions ---
mask_y = ((data_coordis[:, 1] == 0) | (data_coordis[:, 1] == 1)) & (data_coordis[:,0] != 0)  # y = 0 or 1 and t not 0
mask_x = ((data_coordis[:, 2] == 0) | (data_coordis[:, 2] == 1)) & (data_coordis[:,0] != 0)    # x = 0 or 1 and t not 0
mask_t = (data_coordis[:, 0] == 1) 

mask_bc = mask_y | mask_x | mask_t

coords_bc = data_coordis[mask_bc]
values_bc = y[mask_bc]

idx = torch.randperm(coords_bc.shape[0])[:N_samples]

X_bc=coords_bc[idx]
Y_bc=values_bc[idx]

#--------------------------------------------------------------------/

# Dataloaders for batching
#--------------------------------------------------------------------/

data_dataset=DataPointsDataset(X_train,y_train)
ic_dataset=DataPointsDataset(X_ic,Y_ic)
bc_dataset=DataPointsDataset(X_bc,Y_bc)


data_loader = DataLoader(data_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
phys_loader = DataLoader(phys_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
ic_loader = DataLoader(ic_dataset,batch_size=batch_size, shuffle=True, num_workers=2)
bc_loader = DataLoader(bc_dataset,batch_size=batch_size, shuffle=True, num_workers=2)

#--------------------------------------------------------------------/


# Network and optimizer initialization
#--------------------------------------------------------------------/
layers= np.array([3,30,30,30,30,30,30,30,30,1]) # 8 hidden layers

# Network definition
PINN=fcn_plane(layers,q,Cp,t_scale,x_scale,y_scale)
PINN=PINN.to(device)

optimizer = torch.optim.Adam(list(PINN.parameters()),lr=lr)
#--------------------------------------------------------------------/
epoch_avg_losses=[]
data_avg_losses=[]
physics_avg_losses=[]
ic_avg_losses=[]
bc_avg_losses=[]

n_batches = min(len(data_loader), len(phys_loader), len(ic_loader), len(bc_loader))
if __name__ == "__main__":
    for epoch in range(N_epoch):
        PINN.train()
        running_total = 0.0
        running_data = 0.0
        running_phys = 0.0
        running_ic=0.0
        running_bc=0.0

        for (X_data_batch, y_data_batch), X_phys_batch,(X_ic_batch, y_ic_batch),(X_bc_batch,y_bc_batch) in zip(data_loader, phys_loader,ic_loader,bc_loader):
            X_data_batch = X_data_batch.to(device)
            y_data_batch = y_data_batch.to(device)
            X_phys_batch = X_phys_batch.to(device)
            X_ic_batch=X_ic_batch.to(device)
            y_ic_batch=y_ic_batch.to(device)
            X_bc_batch=X_bc_batch.to(device)
            y_bc_batch=y_bc_batch.to(device)

            optimizer.zero_grad()
            loss,loss_data,loss_phys,loss_bc,loss_ic=PINN.loss(X_data_batch,y_data_batch,
                                        X_bc_batch,
                                        y_bc_batch,
                                        X_ic_batch,
                                        y_ic_batch,
                                        X_phys_batch
                                        )
            
            loss.backward()
            optimizer.step()

            running_total += loss.item()
            running_data += loss_data.item()
            running_phys += loss_phys.item()
            running_bc+=loss_bc.item()
            running_ic+=loss_ic.item()

        # average over batches in this epoch
        avg_total = running_total / n_batches
        avg_data = running_data / n_batches
        avg_phys = running_phys / n_batches
        avg_bc=running_bc/n_batches
        avg_ic=running_ic/n_batches

        epoch_avg_losses.append(avg_total)
        data_avg_losses.append(avg_data)
        physics_avg_losses.append(avg_phys)
        ic_avg_losses.append(avg_ic)
        bc_avg_losses.append(avg_bc)

        if (epoch % 10 == 0) or (epoch == N_epoch-1):
                print(f"Epoch {epoch:04d}  Total={avg_total:.6e}  Data={avg_data:.6e}  Phys={avg_phys:.6e} IC={avg_ic:.6e} BC={avg_bc:.6e}  k={PINN.k.item():.3e}")
