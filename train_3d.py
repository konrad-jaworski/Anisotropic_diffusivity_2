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


torch.manual_seed(0)
np.random.seed(0)

# /--------------------------------------------------------------------/
# Training parameters
N_epoch=1000
lr=1e-3 

# Parameters for the grad norm moving averages
lr2=1e-2
alpha=0.26
gradnorm_mode=True

# Weight for mode without gradnorm active
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

data_path=r'C:\Users\stone\Desktop\Synthetic_data_no_defect\2025_10_24_sample_100x100x5mm_no_defect_isotropic_gaussian_heat.npz'
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

# Logging network performance
log_loss_total=[]
log_loss_data=[]
log_loss_phys=[]

# Logging weights and loss of the additional loss
log_weights=[]
log_loss=[]

# Log of the diffusivity parameter
log_a=[]

# Setting up the optimizer
optimizer1=optim.Adam(PINN.parameters(),lr=lr) # Optimizer for the network

if decay_mode==1:
    scheduler = ExponentialLR(optimizer1, gamma=0.95)

iters=0
PINN.train()
for epoch in tqdm(range(N_epoch)):
    
    total_loss_batch=0
    phys_loss_batch=0
    data_loss_batch=0
    
    if sampling_mode==1:
        X_coll=coll_data.resample()
        X_data,Y_data=d_operator.generate() # shape [N,4] -> [T,Y,X,Z]
        dataset = TensorDataset(X_coll, X_data, Y_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
       
    for x_coll_batch,x_data_batch,y_data_batch in loader:
        # Moving data to the GPU
        x_coll_batch=x_coll_batch.to(device)
        x_data_batch=x_data_batch.to(device)
        y_data_batch=y_data_batch.to(device)

        # Compute individual losses
        loss_data=PINN.Data_loss(x_data_batch,y_data_batch)
        loss_phys=PINN.PDE_loss(x_coll_batch)

        if gradnorm_mode:
            loss=torch.stack([loss_data,loss_phys])
            if iters==0:
                # init weights
                weights = torch.ones_like(loss)
                weights = torch.nn.Parameter(weights)
                T = weights.sum().detach() # sum of weights
                # set optimizer for weights
                optimizer2 = torch.optim.Adam([weights], lr=lr2)
                # set L(0)
                l0 = loss.detach()

            # compute the weighted loss
            weighted_loss = weights @ loss
            # clear gradients of network
            optimizer1.zero_grad()   
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                dl = torch.autograd.grad(weights[i]*loss[i], shared_layer, retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)
            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of weights
            optimizer2.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()
            
            # weight for each task
            log_weights.append(weights.detach().cpu().numpy().copy())
            # task normalized loss
            log_loss.append(loss_ratio.detach().cpu().numpy().copy())    
            
            # update model weights
            optimizer1.step()
            # update loss weights
            optimizer2.step()

            # renormalize weights
            # Option A with reinitialization of the optimizer
            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            optimizer2 = torch.optim.Adam([weights], lr=lr2)

            # Option B with keeping the momentum
            # with torch.no_grad():
            #     weights *= T / weights.sum()


            # update iters
            iters += 1
            
            total_loss_batch+=(loss_data+loss_phys).item()
            data_loss_batch+=loss_data.item()
            phys_loss_batch+=loss_phys.item()


        else:
            loss=(1-z)*loss_data+z*loss_phys
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            total_loss_batch+=(loss_data+loss_phys).item()
            data_loss_batch+=loss_data.item()
            phys_loss_batch+=loss_phys.item()

    # Conditioning for the lr decay 
    if decay_mode==1 and epoch%decay_every==0 and epoch>0:
        scheduler.step()

    avg_total_loss=total_loss_batch/len(loader)
    avg_phys_loss=phys_loss_batch/len(loader)
    avg_data_loss=data_loss_batch/len(loader)

    log_loss_total.append(avg_total_loss)
    log_loss_phys.append(avg_phys_loss)
    log_loss_data.append(avg_data_loss)

    if epoch%10 == 0:
        print(f"Epoch: {epoch} | Data loss: {avg_data_loss} | Phys loss: {avg_phys_loss} | Coefficient: {PINN.a}")

