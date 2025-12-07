import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=np.load(r'C:\Users\stone\Desktop\Synthetic_data_no_defect\2025_11_18_sample_100x100x5mm_no_defect_isotropic_gaussian_heat_no_conv_cond_5.npz',allow_pickle=True)

# Normalize temperature data with the envireoment
temp=data['data']
temp=(temp-temp.min())/(temp.max()-temp.min())
print(f"Temp min: {temp.min()} | Temp max: {temp.max()}")

class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    

T,Y,X=temp.shape
y_center=Y//2
x_center=X//2
u_np=temp[34:,y_center,x_center]

# define a neural network to train
pinn = FCN(2,1,100,4).to(device)

# Measured data
u=torch.from_numpy(u_np).to(torch.float32).view(-1,1)

t=torch.linspace(0,10,300).view(-1,1)
x_boundary=torch.zeros_like(u)

x_data=torch.hstack([t,x_boundary])
x_boundary_back=torch.tensor([0.0,0.1,10.0,0.1]).view(-1,2)
X=torch.concatenate([x_data,x_boundary_back],dim=0)

u_back=torch.tensor([0.0,0.0]).view(-1,1)
u_back
Y=torch.vstack([u,u_back])

# Collocation points
res=100 # Resolution parameter to our solution
time_res=3000
z_coll=torch.linspace(0,0.005,res)
t_coll=torch.linspace(0,10,time_res)

# Torch meshgrid only accept 1d tensors as input
T_mesh, X_mesh = torch.meshgrid(t_coll, z_coll, indexing='ij')
X_coll = torch.stack([T_mesh.reshape(-1), X_mesh.reshape(-1)], dim=1)

X=X.requires_grad_(True).to(device)
Y=Y.requires_grad_(True).to(device)
X_coll=X_coll.requires_grad_(True).to(device)
flux_at_boundary=torch.zeros_like(Y).requires_grad_(True).to(device)

n_epoch=100001

log_losses=[]
log_data_losses=[]
log_bnd_losses=[]
log_pde_losses=[]
log_a=[]

# We assume our starting point to be somewhere here
a_phys=5.0/(1600*700)
a = torch.log(torch.tensor([a_phys], dtype=torch.float32))
a=a.to(device)

optimiser=optim.Adam(pinn.parameters(),lr=1e-3)

for i in tqdm(range(n_epoch)):
  optimiser.zero_grad()
    
  # compute each term of the PINN loss function above
  # using the following hyperparameters
  lambda1, lambda2, lambda3 = 1, 1e-1, 1e-3

  # Calculate boundary loss data
  u_p=pinn(X)
  loss1=torch.mean((u_p-Y)**2)

  # Calculate Neuman boundary loss
  u_tx=torch.autograd.grad(u_p,X,torch.ones_like(u_p),create_graph=True)[0]
  u_t=u_tx[:,0].view(-1,1) # Time gradient
  u_x=u_tx[:,1].view(-1,1) # Space gradient
  loss2=torch.mean((u_x-flux_at_boundary)**2)

  # Calculate PDE loss
  u_coll=pinn(X_coll)
  u_coll_tx=torch.autograd.grad(u_coll,X_coll,torch.ones_like(u_coll),create_graph=True)[0]
  u_coll_t=u_coll_tx[:,0].view(-1,1) # Gradient in time
  u_coll_x=u_coll_tx[:,1].view(-1,1) # Gradient in space

  u_coll_xx=torch.autograd.grad(u_coll_x,X_coll,torch.ones_like(u_coll_x),create_graph=True)[0][:,1].view(-1,1)
    
  a_actual=torch.exp(a)
  f=u_coll_t-a_actual*u_coll_xx
  loss3=torch.mean(f**2)

  losses=lambda1*loss1+lambda2*loss2+lambda3*loss3    

  losses.backward()
  optimiser.step()

  log_losses.append(losses.item())
  log_data_losses.append(loss1.item())
  log_bnd_losses.append(loss2.item())
  log_pde_losses.append(loss3.item())
  log_a.append(a_actual.item())

  if i % 100 == 0:
    print(f"[{i}] "
          f"Total: {losses.item():.4e} | "
          f"Data: {loss1.item():.4e} | "
          f"Boundary: {loss2.item():.4e} | "
          f"PDE: {loss3.item():.4e}"
          f" | a: {a_actual.item():.4e}"
          )

torch.save(pinn.state_dict(), 'pinn_1D_diffusivity_model_more_collocation.pth')
