import torch
import torch.nn as nn
import torch.optim as optim
from helper_function import DataGenerator
from networks import ThermalDiffusionPINN
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Physical scales of the problem 14 [s] 0.1 [m] width and height of the space domain and 0.005 [m] of the depth domain
phys_scales=torch.tensor([14.0,0.1,0.1,0.005]).to(torch.float32)

data_uniform=np.load(r"C:\Users\stone\Desktop\Synthetic_data_no_defect\2025_10_24_sample_100x100x5mm_no_defect_isotropic.npz",allow_pickle=True)
# data_uniform=np.load(r"/Volumes/KINGSTON/Synthetic_data_no_defect/2025_10_24_sample_100x100x5mm_no_defect_isotropic.npz",allow_pickle=True)
ccase=torch.from_numpy(data_uniform['data']).to(torch.float32)

# Normalization of the temperature value
ccase=(ccase-ccase.min())/(ccase.max()-ccase.min())

operator=DataGenerator(ccase)

# Generate data
X,Y=operator.get_indexes_latin(5000)
X_data,Y_data=operator.network_format(X,Y,2)

# Generate boundary data
X_bd,Y_bd=operator.boundary_data(4)
X_bd_data,Y_bd_data=operator.network_format(X_bd,Y_bd,2)

# Initial conditions of the points
X_init,Y_init=operator.set_initial_condition_points(20000)

# Collocation points
time_axis,space_points=operator.set_collocation_points(100,5000)

# Normalization of the coordinate space

def normalize_coord(X,phys_scales):
    for i in range(4):
        X[:,i]=X[:,i]/phys_scales[i]
    return X

X_data_norm=normalize_coord(X_data,phys_scales)
X_bd_data_norm=normalize_coord(X_bd_data,phys_scales)
X_init_norm=normalize_coord(X_init,phys_scales)

X_data=torch.concatenate([X_data_norm,X_bd_data_norm,X_init_norm],dim=0)
Y_data=torch.concatenate([Y_data,Y_bd_data,Y_init],dim=0)

time_axis=time_axis/phys_scales[0]
space_points[:,0]=space_points[:,0]/phys_scales[1]
space_points[:,1]=space_points[:,1]/phys_scales[2]
space_points[:,2]=space_points[:,2]/phys_scales[3]

# Moving all of the data to the GPU
X_data=X_data.to(device)
Y_data=Y_data.to(device)

time_axis=time_axis.to(device)
space_points=space_points.to(device)

# Definition of the network
layers=[4,50,50,50,50,1]
PINN=ThermalDiffusionPINN(layers)
PINN.to(device)


log_loss_total=[]
log_loss_data=[]
log_loss_phys=[]

optimizer=optim.Adam(PINN.parameters(),lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10000,        # Initial restart period (10% of total epochs)
    T_mult=2,         # Double the period after each restart
    eta_min=1e-6      # Minimum learning rate
)

N_epoch=100000


early_stop_patience = 5000
early_stop_delta = 1e-6
best_loss = float("inf")
epochs_no_improve = 0


for i in tqdm(range(N_epoch)):
    
    # Zero the gradients
    optimizer.zero_grad()

    # Data loss
    data_loss=PINN.Data_loss(X_data,Y_data)

    pde_loss = torch.tensor(0.0, device=device)

    for i in range(len(time_axis) - 1):
        step = operator.time_stepping(time_axis, space_points, i)
        pde_loss += PINN.PDE_loss(step)

    pde_loss = pde_loss / (len(time_axis) - 1)

    loss = 0.5*data_loss + 0.5*pde_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    log_loss_data.append(data_loss.item())
    log_loss_phys.append(pde_loss.item())
    log_loss_total.append(loss.item())

    current_loss = loss.item()

    # ---- Early stopping logic ----
    if current_loss < best_loss - early_stop_delta:
        best_loss = current_loss
        epochs_no_improve = 0

        # Save best model
        torch.save(PINN.state_dict(), "pinn_best.pth")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
        print(f"\nEarly stopping at epoch {i}")
        print(f"Best loss: {best_loss:.6e}")
        break

    if i % 100 == 0:
        print(f"Epoch {i}, Total Loss: {loss.item():.6f}, Data Loss: {data_loss.item():.6f}, PDE Loss: {pde_loss.item():.6f}")
    
# Save the model
torch.save(PINN.state_dict(), 'pinn_final.pth')
log_loss_data=torch.from_numpy(np.array(log_loss_data))
log_loss_phys=torch.from_numpy(np.array(log_loss_phys))
log_loss_total=torch.from_numpy(np.array(log_loss_total))
torch.save(log_loss_data, 'log_loss_data.pt')
torch.save(log_loss_phys, 'log_loss_phys.pt')
torch.save(log_loss_total, 'log_loss_total.pt') 




