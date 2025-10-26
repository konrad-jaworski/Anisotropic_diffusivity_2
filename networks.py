import torch
import torch.nn as nn

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class fcn_isotropic(nn.Module):
    def __init__(self,layers_temp,layers_lap):
        super(fcn_isotropic, self).__init__()

        self.layers_temp=layers_temp
        self.layers_lap=layers_lap

        # Activation function
        self.activation=nn.Tanh()
        # Loss function
        self.loss_function=nn.MSELoss(reduction='mean')

        # Temperature branch
        self.linears_temp = nn.ModuleList([nn.Linear(self.layers_temp[i], self.layers_temp[i+1]) for i in range(len(self.layers_temp)-1)])
        # Laplacian branch
        self.linears_lap = nn.ModuleList([nn.Linear(self.layers_lap[i], self.layers_lap[i+1]) for i in range(len(self.layers_lap)-1)])

        # Xavier initialization for temperature branch 
        for i in range(len(self.layers_temp)-1):
            nn.init.xavier_normal_(self.linears_temp[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears_temp[i].bias.data)

        # Xavier initialization for laplacian branch
        for i in range(len(self.layers_lap)-1):
            nn.init.xavier_normal_(self.linears_lap[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears_lap[i].bias.data)

        # Initialization of the inverse problem parameters
        self.a = torch.tensor([1.0e-5], requires_grad=True).float()
        self.a = nn.Parameter(self.a)
        
    def forward(self,x):
        # Coordinates to our network comes in format [t, y, x]
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
            
        # Remain data copy for the second derivative estimation
        lap_data=x.clone()
        
        # Temperature estimation branch
        for i in range(len(self.layers_temp)-2):
            u = self.linears_temp[i](x)
            x = self.activation(u)
                
        u = self.linears_temp[-1](x)

        # Second derivative estimation branch
        for i in range(len(self.layers_lap)-2):                
            z = self.linears_lap[i](lap_data)
            lap_data = self.activation(z)
                
        u_zz = self.linears_lap[-1](lap_data)
            
        return u,u_zz
        
    def loss_data(self,x,y):
        # Network takes normalized input -1 to 1 and should outputs normalized temperature values
        temps,_=self(x)
        # y is in normalized temperature values -1 to 1
        loss_val=self.loss_function(temps,y)
        return loss_val
        
    def loss_PDE(self, x_pde,scales):
        a = self.a   # trainable diffusivity parameter

        # Clone for autograd
        g = x_pde.clone()
        g.requires_grad = True

        # Forward pass
        u, u_zz = self(g)  # u_zz is second derivative in virtual z

        # First derivatives w.r.t. normalized coordinates
        u_t_x = torch.autograd.grad(
            u, g, torch.ones_like(u).to(device),
            retain_graph=True, create_graph=True
        )[0]

        # Second derivatives w.r.t. normalized coordinates
        u_tt_xx = torch.autograd.grad(
            u_t_x, g, torch.ones_like(u_t_x).to(device),
            create_graph=True
        )[0]

        # Separate derivatives
        u_t  = u_t_x[:, 0]        # normalized t
        u_xx = u_tt_xx[:, 2]      # normalized x
        u_yy = u_tt_xx[:, 1]      # normalized y

        T_scale, y_scale, x_scale, t_scale, z_scale = scales
        # Properly scale derivatives back to physical domain
        u_t_phys  = u_t  / t_scale *T_scale 
        u_xx_phys = u_xx / x_scale**2 *T_scale
        u_yy_phys = u_yy / y_scale**2 *T_scale
        u_zz_phys = u_zz / z_scale**2 *T_scale

        # PDE residual (heat equation)
        f = u_t_phys - a * (u_xx_phys + u_yy_phys + u_zz_phys)

        loss_val = torch.mean(f**2)
        return loss_val
        
    def loss(self,x,y,x_pde,scales):
        loss_u=self.loss_data(x,y)
        loss_f=self.loss_PDE(x_pde,scales)

        loss_val=loss_u+loss_f
        return loss_val,loss_u,loss_f
    
        