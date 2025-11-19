import torch
import torch.optim as optim
import numpy as np
from networks import FCN
from helper_function import DomainDataset,gradNorm
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
#--------------------------------------------------------------------/
N_epoch=10000
lr=1e-4
N_run=1

# Number of samples
n_coll=100000 # collocation points
n_bc=100000 # boundary condition points
n_ic=100000 # Initial condition points

# Grad norm parameters
lr2=1e-2 # Learning rate for grad norm
alpha=0.26 # assymetry parameter
gradnorm_mode=True # Flag to activate GradNorm

# Weight value of the physic loss when grad norm is deactivated
z=0.1

target_diffusivity=5.0/(700.0*1600.0) # Target thermal diffusivity

# Data preprocessing
#--------------------------------------------------------------------/
data = np.load(r'F:\Synthetic_data_no_defect\2025_11_18_sample_100x100x5mm_no_defect_isotropic_gaussian_heat_no_conv_cond_5.npz', allow_pickle=True)


data_cube = torch.tensor(data['data'][34:,:,:], dtype=torch.float32)
data_cube=(data_cube-data_cube.min())/(data_cube.max()-data_cube.min()) # Normalization of the temperature data

T, Y, X = data_cube.shape

# Initial condition at t=0
Y_ic = data_cube[0, :, :]          # shape [Y, X]
X_y, X_x = torch.meshgrid(torch.linspace(0,1,512), torch.linspace(0,1,512), indexing='ij')
X_ic = torch.cat([
    torch.zeros(Y*X,1),           # t=0
    X_y.reshape(-1,1),
    X_x.reshape(-1,1)
], dim=1)
Y_ic = Y_ic.reshape(-1,1)
N_total_ic = X_ic.shape[0]

X_ic=X_ic.to(device)
Y_ic=Y_ic.to(device)

# Boundary conditions
time = torch.linspace(0, 1, T)
y_space = torch.linspace(0, 1, Y)
x_space = torch.linspace(0, 1, X)
X_bc_left = torch.stack([
    time.repeat_interleave(Y),          # repeat each t for all y
    y_space.repeat(T),                  # each y repeated across t
    torch.zeros(T*Y)                    # x=0
], dim=1)

Y_bc_left = data_cube[:, :, 0].reshape(-1,1)  # flatten [T*Y,1]

X_bc_right = torch.stack([
    time.repeat_interleave(Y),
    y_space.repeat(T),
    torch.ones(T*Y)                      # x=1
], dim=1)

Y_bc_right = data_cube[:, :, -1].reshape(-1,1)

X_bc_bottom = torch.stack([
    time.repeat_interleave(X),
    torch.zeros(T*X),                   # y=0
    x_space.repeat(T)
], dim=1)

Y_bc_bottom = data_cube[:, 0, :].reshape(-1,1)


X_bc_top = torch.stack([
    time.repeat_interleave(X),
    torch.ones(T*X),                    # y=1
    x_space.repeat(T)
], dim=1)

Y_bc_top = data_cube[:, -1, :].reshape(-1,1)

X_bc = torch.cat([X_bc_left, X_bc_right, X_bc_bottom, X_bc_top], dim=0)
Y_bc = torch.cat([Y_bc_left, Y_bc_right, Y_bc_bottom, Y_bc_top], dim=0)

N_total_bc = X_bc.shape[0]

X_bc=X_bc.to(device)
Y_bc=Y_bc.to(device)


coll_data=DomainDataset(n_samples=n_coll,n_dim=3,method='lhs')
#--------------------------------------------------------------------/
# Logging diffusivity estimation
log_a=[]
run_iter=0

while run_iter<=N_run:

    # Definition of the network
    layers=[3,50,50,50,50,50,50,1]
    PINN=FCN(layers)
    PINN=PINN.to(device) # Moving model to GPU

    # Layers to apply GradNorm
    shared_layer=list(PINN.linears[-1].parameters())

    # Logging
    log_weights=[]
    log_loss_total=[]
    log_loss_data=[]
    log_loss_phys=[]

    # Setting up the optimizer
    optimizer_1=optim.Adam(PINN.parameters(),lr=lr) # Optimizer for the network
    scheduler = ExponentialLR(optimizer_1, gamma=0.95)
    iters=0

    PINN.train()

    for epoch in tqdm(range(N_epoch)):
        # Data points for this epoch
        indices_bc = torch.randperm(N_total_bc)[:n_bc]
        X_bc_sampled = X_bc[indices_bc]
        Y_bc_sampled = Y_bc[indices_bc]

        indices_ic = torch.randperm(N_total_ic)[:n_ic]
        X_ic_sampled = X_ic[indices_ic]
        Y_ic_sampled = Y_ic[indices_ic]

        # Collocation points for PDE residual
        x_coll=coll_data.resample()
        x_coll = x_coll.to(device)

        optimizer_1.zero_grad()
            
        # Compute losses conditionally
        loss_bc=PINN.Data_loss(X_bc_sampled,Y_bc_sampled)
        loss_ic=PINN.Data_loss(X_ic_sampled,Y_ic_sampled)
        loss_data=loss_bc+loss_ic
        loss_phys=PINN.PDE_loss(x_coll)

        losses=torch.stack([loss_data,loss_phys])
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

            # compute the weighted loss
            weighted_loss = weights @ losses
            # clear gradients of network
            # optimizer_1.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)

            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(losses)):
                dl = torch.autograd.grad(weights[i]*losses[i], shared_layer, retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)

            # compute loss ratio per task
            loss_ratio = losses.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()

            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of weights
            optimizer_2.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()

            # update model weights
            optimizer_1.step()
            # scheduler step
            if epoch % 500 ==0:
                scheduler.step() 
            # update loss weights
            optimizer_2.step()


            weight_1=weights[0].item()
            weight_2=weights[1].item()


            # renormalize weights
            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            optimizer_2 = torch.optim.Adam([weights], lr=lr2)
            # update iters
            iters += 1

        else:
            loss = (1-z)*loss_data + z*loss_phys
            if loss != 0.0:
                loss.backward()
                optimizer_1.step()
                # scheduler step
            if epoch % 250 ==0:
                scheduler.step() 

        # logging
        log_loss_total.append((loss_data.item()+loss_phys.item()))
        log_loss_data.append(loss_data.item())
        log_loss_phys.append(loss_phys.item())

        if epoch % 100 == 0 or epoch == N_epoch - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Data Loss = {weight_1*loss_data.item():.6f} | "
                f"Weight 1 = {weight_1:.6f} | "
                f"Physics Loss = {weight_2*loss_phys.item():.6f} | "
                f"Weight 2 = {weight_2:.6f} | "
                f"a = {PINN.a.item():.6e}"
            )
    if PINN.a.item()>0:
        log_a.append(PINN.a.item())
        run_iter=run_iter+1
        print(f'Starting run {run_iter+1}')

torch.save(log_a,"diffusivity_ensamble_estimation_case_cond_5.pth")




