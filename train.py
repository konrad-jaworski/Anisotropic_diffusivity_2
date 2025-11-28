import torch
import torch.optim as optim
import numpy as np
from networks import FCN
from helper_function import DomainDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
#--------------------------------------------------------------------/
N_epoch=100000
lr=1e-3
N_run=1

# Grad norm parameters
lr2=1e-2 # Learning rate for grad norm
alpha=0.26 # assymetry parameter
gradnorm_mode=True # Flag to activate GradNorm

# Weight value of the physic loss when grad norm is deactivated
z=0.5

# --------------------Data preprocessing------------------------
data_path = r'C:\Users\stone\Desktop\Synthetic_data_no_defect\2025_10_24_sample_100x100x5mm_no_defect_isotropic_gaussian_heat.npz'
data = np.load(data_path, allow_pickle=True)
data_cube = data['data'][34:, :, :]  # shape [T, Y, X]
# Normalization of the data
data_cube = (data_cube - data_cube.min()) / (data_cube.max() - data_cube.min())

# ------------------- RANDOM INTERIOR SAMPLER -------------------
def sample_random_data_points(data_cube, N_samples):
    T, Y, X = data_cube.shape
    t_idx = np.random.randint(0, T, N_samples)
    y_idx = np.random.randint(0, Y, N_samples)
    x_idx = np.random.randint(0, X, N_samples)

    t_norm = t_idx / (T - 1)
    y_norm = y_idx / (Y - 1)
    x_norm = x_idx / (X - 1)

    X_data = np.stack([t_norm, y_norm, x_norm], axis=1)
    Y_data = data_cube[t_idx, y_idx, x_idx].reshape(-1,1)
    return X_data, Y_data

N_interior = 500000 #---------------------------------------------------------------------------------------------------------------------------------------------------
X_data_rand, Y_data_rand = sample_random_data_points(data_cube, N_interior)

X_data_rand=torch.from_numpy(X_data_rand).float()
X_data_rand=X_data_rand.to(device)

Y_data_rand=torch.from_numpy(Y_data_rand).float()
Y_data_rand=Y_data_rand.to(device)


# ------------------- INITIAL CONDITION -------------------
def sample_random_data_points_ic(data_cube,N_samples):
    T, Y, X = data_cube.shape
    t_idx = np.int16(np.zeros(N_samples))
    y_idx = np.random.randint(0, Y, N_samples)
    x_idx = np.random.randint(0, X, N_samples)

   
    y_norm = y_idx / (Y - 1)
    x_norm = x_idx / (X - 1)

    X_data = np.stack([t_idx, y_norm, x_norm], axis=1)
    Y_data = data_cube[t_idx, y_idx, x_idx].reshape(-1,1)
    return X_data, Y_data

N_ic=30000 #---------------------------------------------------------------------------------------------------------------------------------------------------
X_data_ic,Y_data_ic=sample_random_data_points_ic(data_cube,N_ic)

X_data_ic=torch.from_numpy(X_data_ic).float()
X_data_ic=X_data_ic.to(device)

Y_data_ic=torch.from_numpy(Y_data_ic).float()
Y_data_ic=Y_data_ic.to(device)

# ------------------ Boundary conditions ------------------
def sample_random_data_points_bc(data_cube,N_samples=5000):
    T, Y, X = data_cube.shape

    # Lower boundary  y = 0
    t_idx_lower = np.random.randint(0, T, N_samples)
    y_idx_lower = np.int16(np.zeros(N_samples))
    x_idx_lower = np.random.randint(0, X, N_samples)

    t_norm_lower = t_idx_lower/(T-1)
    x_norm_lower = x_idx_lower / (X - 1)

    # Right boundary x = 0
    t_idx_right = np.random.randint(0, T, N_samples)
    y_idx_right = np.random.randint(0,Y,N_samples)
    x_idx_right = np.int16(np.zeros(N_samples))

    t_norm_right = t_idx_right/(T-1)
    y_norm_right = y_idx_right /(Y-1)

    # Upper boundary y = 1
    t_idx_upper = np.random.randint(0, T, N_samples)
    y_idx_upper = np.int16(np.ones(N_samples))
    x_idx_upper = np.random.randint(0, X, N_samples)

    t_norm_upper = t_idx_upper/(T-1)
    x_norm_upper = x_idx_upper / (X - 1)

    # Left boundary x = 1
    t_idx_left = np.random.randint(0, T, N_samples)
    y_idx_left = np.random.randint(0,Y,N_samples)
    x_idx_left = np.int16(np.ones(N_samples))

    t_norm_left = t_idx_left/(T-1)
    y_norm_left = y_idx_left /(Y-1)

    T_stage_one=np.concatenate([t_norm_lower,t_norm_upper,t_norm_right,t_norm_left],axis=0)
    Y_stage_one=np.concatenate([y_idx_lower,y_idx_upper,y_norm_right,y_norm_left],axis=0)
    X_stage_one=np.concatenate([x_norm_lower,x_norm_upper,x_idx_right,x_idx_left],axis=0)
    
    X_data_bc=np.stack([T_stage_one,Y_stage_one,X_stage_one],axis=1)
    
    T_idx_stage_one=np.concatenate([t_idx_lower,t_idx_upper,t_idx_right,t_idx_left],axis=0)
    Y_idx_stage_one=np.concatenate([y_idx_lower,y_idx_upper,y_idx_right,y_idx_left],axis=0)
    X_idx_stage_one=np.concatenate([x_idx_lower,x_idx_upper,x_idx_right,x_idx_left],axis=0)

    Y_data_bc=data_cube[T_idx_stage_one,Y_idx_stage_one,X_idx_stage_one].reshape(-1,1)

    return X_data_bc,Y_data_bc

X_data_bc,Y_data_bc=sample_random_data_points_bc(data_cube)

X_data_bc=torch.from_numpy(X_data_bc).float()
X_data_bc=X_data_bc.to(device)

Y_data_bc=torch.from_numpy(Y_data_bc).float()
Y_data_bc=Y_data_bc.to(device)


#---------------- Collocation points ---------------------
n_coll=500000 #---------------------------------------------------------------------------------------------------------------------------------------------------
coll_data=DomainDataset(n_samples=n_coll,n_dim=3,method='lhs')

x_coll=coll_data.resample()
x_coll = x_coll.to(device)
#--------------------------------------------------------------------/
run_iter=0
# num_epoch=1000

while run_iter<=N_run:

    # Definition of the network
    layers=[3,100,100,100,100,1]
    PINN=FCN(layers)
    PINN=PINN.to(device) # Moving model to GPU

    # Parameters for the grad norm moving average
    alpha_ema=0.01
    ema_losses=None

    # Layers to apply GradNorm (last hidden layer shared between tasks)
    shared_layer=list(PINN.linears[-2].parameters())[0]

    # Logging
    log_weights_1=[]
    log_weights_2=[]
    log_loss_total=[]
    log_loss_data=[]
    log_loss_phys=[]
    log_a=[]

    # Setting up the optimizer
    optimizer_1=optim.Adam(PINN.parameters(),lr=lr) # Optimizer for the network
    # scheduler = ExponentialLR(optimizer_1, gamma=0.95)
    iters=0

    PINN.train()

    for epoch in tqdm(range(N_epoch)):
        
            
        # Compute  individual losses
        loss_bc=PINN.Data_loss(X_data_bc,Y_data_bc)

        loss_ic=PINN.Data_loss(X_data_ic,Y_data_ic)

        loss_interior=PINN.Data_loss(X_data_rand,Y_data_rand)

        loss_phys=PINN.PDE_loss(x_coll)

        loss_data=loss_bc+loss_ic+loss_interior

        # Combination of losses for grad norm
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
                # l0=losses.detach()

                # Ema variant initialization
                ema_losses=losses.detach()

            # Moving average
            ema_losses=(1-alpha_ema)*ema_losses+alpha_ema*losses.detach()

            # compute the weighted loss
            weighted_loss = weights @ losses

            # Clean gradient of the network
            optimizer_1.zero_grad()
            
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)

            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(losses)):
                dl = torch.autograd.grad(weights[i]*losses[i], shared_layer, retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)

            # compute loss ratio per task
            # loss_ratio = losses.detach() / l0
            loss_ratio = losses.detach() / ema_losses

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
            
            # # scheduler step
            # if epoch % num_epoch ==0:
            #     scheduler.step() 

            # update loss weights
            optimizer_2.step()

            # renormalize weights
            # weights = (weights / weights.sum() * T).detach()
            # weights = torch.nn.Parameter(weights)
            # optimizer_2 = torch.optim.Adam([weights], lr=lr2)

            # Modification to keep the momentum of Adam optimizer
            with torch.no_grad():
                weights[:] = weights / weights.sum() * T

            # update iters
            iters += 1

        # else:
        #     loss = (1-z)*loss_data + z*loss_phys
        #     if loss != 0.0:
        #         loss.backward()
        #         optimizer_1.step()
        #         # scheduler step
        #     if epoch % 500 ==0:
        #         scheduler.step() 

        # logging
        log_loss_total.append((loss_data.item()+loss_phys.item()))
        log_loss_data.append(loss_data.item())
        log_loss_phys.append(loss_phys.item())
        log_weights_1.append(weights[0].item())
        log_weights_2.append(weights[1].item())
        log_a.append(PINN.a.item())

        if epoch % 100 == 0 or epoch == N_epoch - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Data Loss = {weights[0].item()*loss_data.item():.6f} | "
                f"Weight 1 = {weights[0].item():.6f} | "
                f"Physics Loss = {weights[1].item()*loss_phys.item():.6f} | "
                f"Weight 2 = {weights[1]:.6f} | "
                f"a = {PINN.a.item():.6e}"
   reor         )
        if epoch % 1000 == 0 or epoch == N_epoch - 1:
            logs = {
                        "total_loss": log_loss_total,
                        "data_loss": log_loss_data,
                        "phys_loss": log_loss_phys,
                        "weights_1": log_weights_1,
                        "weights_2": log_weights_2,
                        "a": log_a
                    }
            torch.save(logs,f"Data_network_epoch_{epoch}.pth")

        if epoch % 10000 == 0 or epoch == N_epoch - 1:
            torch.save(PINN.state_dict(), f"PINN_epoch_{epoch}.pth")

    run_iter=run_iter+1
        






