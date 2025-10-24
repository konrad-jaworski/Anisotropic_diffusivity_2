import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from networks import fcn


# Training parameters 
# ----------------------------------------------------------------/
N_epoch=100
lr=1e-1 # Investigating larger steps
patiance=5000
plot_diffusion_per=1
batch_size=5000
dataset_size=100000
#-----------------------------------------------------------------/

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

data = np.load(r'E:\Heat_diffusion_laser_metadata\30_Sep_2025_06_30_29_FBH13mm_step_size_sim_step_0_002m_p1.npz', allow_pickle=True)
data = np.array(data['data'], dtype=np.float32)

# Data preprocessing
t,y,x=data.shape
y_center=y//2
x_center=x//2

# We take into consideration only samller square of 100 x 100 pixels and cooling region
spread=75
data=data[10:,y_center-spread:y_center+spread,x_center-spread:x_center+spread]

# Organiziing data in network freidly format
Nt,Ny,Nx=data.shape

t = torch.linspace(0, 1, Nt)
x = torch.linspace(0, 1, Nx)
y = torch.linspace(0, 1, Ny)

tt, yy, xx = torch.meshgrid(t, y, x, indexing='ij')

coords = torch.stack([tt, yy,xx], dim=-1)  # shape (399, 100, 100, 3)
coords = coords.reshape(-1, 3)              # shape (399*100*100, 3)

values = data.reshape(-1, 1)                   # shape (399*100*100, 1)
values=torch.from_numpy(values)

# Randomly selected samples
idx = torch.randperm(coords.shape[0])[:dataset_size]

X_train=coords[idx]
y_train=values[idx]

X_train=X_train.to(device)
y_train=y_train.to(device)

# Normalization of the temperature values
y_train=(y_train-y_train.mean())/y_train.std()

dataset_size = X_train.shape[0]


# Parameters determining network architecture
layers_temp = np.array([3,30,30,30,30,30,30,30,30,1]) #8 hidden layers
layer_lap=np.array([3,20,20,1])

# Network definition
PINN=fcn(layers_temp,layer_lap)
PINN=PINN.to(device)

# Optimizer
optimizer=optim.Adam(PINN.parameters(),lr=lr)


# Logging purposes
epoch_avg_losses=[]
data_avg_losses=[]
physics_avg_losses=[]

for epoch in range(N_epoch):
    PINN.train()
    perm = torch.randperm(dataset_size)
    epoch_loss = 0.0
    data_loss = 0.0
    physics_loss = 0.0

    with tqdm(range(0, dataset_size, batch_size), desc=f"Epoch {epoch+1}/{N_epoch}", leave=False) as batch_bar:
        for j in batch_bar:
            idx = perm[j:j + batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            loss, loss_data, loss_physics = PINN.loss(X_batch, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            data_loss += loss_data.item()
            physics_loss += loss_physics.item()

            batch_bar.set_postfix({
                "Loss": f"{loss.item():.3e}",
                "Data": f"{loss_data.item():.3e}",
                "PDE": f"{loss_physics.item():.3e}"
            })

    num_batches = int(np.ceil(dataset_size / batch_size))
    epoch_avg_losses.append(epoch_loss / num_batches)
    data_avg_losses.append(data_loss / num_batches)
    physics_avg_losses.append(physics_loss / num_batches)

print(f'Total_loss at start:{epoch_avg_losses[0]} | Total_loss at the end:{epoch_avg_losses[-1]}')
print(f'Data_loss at start:{data_avg_losses[0]} | Data_loss at the end:{data_avg_losses[-1]}')
print(f'Physics_loss at start:{physics_avg_losses[0]} | Physics_loss at the end:{physics_avg_losses[-1]}')

print(f'Diffusivity at x:{PINN.a_x} | Diffusivity at y:{PINN.a_y} | Diffusivity at z:{PINN.a_z} | ')
