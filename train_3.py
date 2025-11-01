import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from helper_function import PointsDataset
from networks import FCN
from tqdm import tqdm
from itertools import zip_longest


# Characterisitc of the sample
sampling_time=30.0 # FPS camera recording
height=0.1 # [m] heigh of the sample
width=0.1 # [m] width of the sample 
thickness=0.005 # [m] thickness of sample

Cp=700.0 # [J/(kg*K)] Specific heat of the material
q=1600.0 # [kg/m^3] Density of the material

# Scales of the domain
t_scale=10.0
y_scale=0.1
x_scale=0.1

N_samples_data=50000 # Number of sample points
N_samples_collocation=50000 # Number of collocation points
N_samples_bc=50000
N_samples_ic=50000
alpha=0.48 # Asymmetry parameter for GradNorm


gradnorm_mode=True # Flag to activate GradNorm

num_epoch=1000  # Number of epochs
lr1=1e-3 # Learning rate for Adam optimizer of the network
lr2=1e-2 # Learning rate for Adam optimizer of the weights in GradNorm

# Data points
X_data=torch.load(r"E:\Workspaces\Diffusicity_estimation_2\Anisotropic_diffusivity_2\data\X_total.pt")
Y_data=torch.load(r"E:\Workspaces\Diffusicity_estimation_2\Anisotropic_diffusivity_2\data\Y_total.pt")

idx_data = torch.randperm(X_data.shape[0])[:N_samples_data]

X_train_data=X_data[idx_data]
Y_train_data=Y_data[idx_data]

# 100_000 Samples of collocation points in the domain
# X_phys=torch.load(r"E:\Workspaces\Diffusicity_estimation_2\Anisotropic_diffusivity_2\data\X_phys_sobol.pt")

engine = torch.quasirandom.SobolEngine(dimension=3)
X_phys_sobol = engine.draw(N_samples_collocation)


# Boundary condition points Dirichlet BCs
X_bc=torch.load(r"E:\Workspaces\Diffusicity_estimation_2\Anisotropic_diffusivity_2\data\X_bc.pt")
Y_bc=torch.load(r"E:\Workspaces\Diffusicity_estimation_2\Anisotropic_diffusivity_2\data\Y_bc.pt")

idx_bc= torch.randperm(X_bc.shape[0])[:N_samples_bc]

X_train_bc=X_bc[idx_bc]
Y_train_bc=Y_bc[idx_bc]

# Initial condition points

X_ic=torch.load(r"E:\Workspaces\Diffusicity_estimation_2\Anisotropic_diffusivity_2\data\X_ic.pt")
Y_ic=torch.load(r"E:\Workspaces\Diffusicity_estimation_2\Anisotropic_diffusivity_2\data\Y_ic.pt")

idx_ic=torch.randperm(X_ic.shape[0])[:N_samples_ic]

X_train_ic=X_ic[idx_ic]
Y_train_bc=Y_ic[idx_ic]


# Creating datasets and dataloaders
data_dataset=PointsDataset(X_train_data,Y_train_data)
data_loader=DataLoader(data_dataset,batch_size=128,shuffle=True,num_workers=0)
n_batches_data=len(data_loader)

phys_dataset=PointsDataset(X_phys_sobol)
phys_loader=DataLoader(phys_dataset,batch_size=128,shuffle=True,num_workers=0)
n_batches_phys=len(phys_loader)

ic_dataset=PointsDataset(X_train_ic,Y_train_bc)
ic_loader=DataLoader(ic_dataset,batch_size=128,shuffle=True,num_workers=0)
n_batches_ic=len(ic_loader)

bc_dataset=PointsDataset(X_train_bc,Y_train_bc)
bc_loader=DataLoader(bc_dataset,batch_size=128,shuffle=True,num_workers=0)
n_batches_bc=len(bc_loader)

# Definition of the network
layers=[3,50,50,50,50,50,1]
PINN=FCN(layers,q,Cp,t_scale,x_scale,y_scale)
PINN=PINN.cuda() # Moving model to GPU


# Layers to apply GradNorm
shared_layer=list(PINN.linears[-1].parameters())

log_weights=[]
log_loss_total=[]
log_loss_data=[]
log_loss_phys=[]
log_loss_bc=[]
log_loss_ic=[]

# Setting up the optimizaers
optimizer_1=optim.Adam(PINN.parameters(),lr=lr1) # Optimizer for the network
iters=0

PINN.train()

for epoch in tqdm(range(num_epoch)):
    running_total = 0.0
    running_data = 0.0
    running_phys = 0.0
    running_ic = 0.0
    running_bc = 0.0

    for batches in zip_longest(data_loader, phys_loader, ic_loader, bc_loader):
        data_batch, phys_batch, ic_batch, bc_batch = batches

        # Default all to None-safe values
        X_data_batch = Y_data_batch = X_phys_batch = X_ic_batch = Y_ic_batch = X_bc_batch = Y_bc_batch = None

        # Unpack non-empty batches and move them to GPU
        if data_batch is not None:
            X_data_batch, Y_data_batch = [b.cuda() for b in data_batch]
        if phys_batch is not None:
            X_phys_batch =  phys_batch.cuda()
        if ic_batch is not None:
            X_ic_batch, Y_ic_batch = [b.cuda() for b in ic_batch]
        if bc_batch is not None:
            X_bc_batch, Y_bc_batch = [b.cuda() for b in bc_batch]

        optimizer_1.zero_grad()
        
        # Compute losses conditionally
        loss_data = PINN.Data_loss(X_data_batch, Y_data_batch) if data_batch is not None else 0.0
        loss_phys = PINN.PDE_loss(X_phys_batch) if phys_batch is not None else 0.0
        loss_ic   = PINN.IC_loss(X_ic_batch, Y_ic_batch) if ic_batch is not None else 0.0
        loss_bc   = PINN.BC_loss(X_bc_batch, Y_bc_batch) if bc_batch is not None else 0.0

        losses=torch.stack([loss_data,loss_phys,loss_ic,loss_bc])
        if gradnorm_mode:
            
            # Initialization for GradNorm
            if iters == 0:
                print("GradNorm activated")
                # init weights
                weights=torch.ones_like(losses)
                weights=torch.nn.Parameter(weights)
                T=weights.sum().detach() # sum of weights
                # set optimizer for weights
                optimizer_2=torch.optim.Adam([weights],lr=lr2)
                # set L(0)
                l0=losses.detach()

            weighted_loss = weights @ losses
            optimizer_1.zero_grad()
            weighted_loss.backward(retain_graph=True)

            gw = []
            for i in range(len(losses)):
                dl = torch.autograd.grad(weights[i]*losses[i], shared_layer, retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)

            loss_ratio = losses.detach() / l0
            rt = loss_ratio / loss_ratio.mean()
            gw_avg = gw.mean().detach()

            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            optimizer_2.zero_grad()
            gradnorm_loss.backward()

            optimizer_1.step()
            optimizer_2.step()

            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            optimizer_2 = torch.optim.Adam([weights], lr=lr2)
            iters += 1

        else:
            loss = loss_data + loss_phys + loss_ic + loss_bc
            if loss != 0.0:
                loss.backward()
                optimizer_1.step()

        # Track
        # running_total += float(loss.item())
        running_data  += float(loss_data.item())
        running_phys  += float(loss_phys.item())
        running_ic    += float(loss_ic.item())
        running_bc    += float(loss_bc.item())
    # Mean losses per epoch
    log_weights.append(shared_layer[0].detach().cpu().numpy())
    log_loss_data.append(running_data / max(1, n_batches_data))
    log_loss_phys.append(running_phys / max(1, n_batches_phys))
    log_loss_ic.append(running_ic / max(1, n_batches_ic))
    log_loss_bc.append(running_bc / max(1, n_batches_bc))

    if epoch % 10 == 0 or epoch == num_epoch - 1:
        print(f"Epoch {epoch:04d} Data={running_data / max(1, n_batches_data):.6e}  Phys={running_phys / max(1, n_batches_phys):.6e}  IC={running_ic / max(1, n_batches_ic):.6e}  BC={running_bc / max(1, n_batches_bc):.6e}  k={PINN.k.item():.3e}")












