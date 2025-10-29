import torch
import torch.nn as nn

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class fcn_plane(nn.Module):
    """
    Network used for prediction of thermal conductivity of the isotropic material with measurments from its surface.
    """
    def __init__(self, layers, density, specific_heat, time_scale, x_scale, y_scale):
        super(fcn_plane, self).__init__()

        # Layers structure
        self.layers = layers

        # Physical parameters
        self.q = density
        self.Cp = specific_heat

        # Normalization scales
        self.t_scale = time_scale
        self.x_scale = x_scale
        self.y_scale = y_scale

        # Activation and loss
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')

        # Define the fully connected layers
        self.linears = nn.ModuleList([
            nn.Linear(self.layers[i], self.layers[i + 1])
            for i in range(len(self.layers) - 1)
        ])

        # Xavier initialization
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Inverse parameter (thermal conductivity)
        # self.k = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        # self.k = nn.Parameter(torch.randn(1))
        self.k = nn.Parameter(torch.empty(1).uniform_(0, 5))
    def forward(self, x):
        # Input shape: [N, 3]  => [t, y, x]
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()

        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)
        return x

    def PDE_loss(self, x_pde):
        # x_pde: [N, 3] → [t, y, x]
        x_pde.requires_grad_(True)
        u = self(x_pde)

        # Compute first derivatives
        grads = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True,retain_graph=True)[0]
        u_t = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_x = grads[:, 2:3]

        # Compute second derivatives
        grads2_y = torch.autograd.grad(u_y, x_pde, torch.ones_like(u_y), create_graph=True)[0]
        grads2_x = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grads2_y[:, 1:2]
        u_xx = grads2_x[:, 2:3]

        # Scale derivatives back to physical units (assuming normalization)
        u_t_phys = u_t / self.t_scale
        u_xx_phys = u_xx / (self.x_scale ** 2)
        u_yy_phys = u_yy / (self.y_scale ** 2)

        # PDE residual: ρ·Cp·∂T/∂t - k(∂²T/∂x² + ∂²T/∂y²)
        f = self.q * self.Cp * u_t_phys - self.k * (u_xx_phys + u_yy_phys)
        loss_phys = torch.mean(f ** 2)

        return loss_phys

    def BC_loss(self, x_bc, y_bc):
        # Boundary condition loss function
        return self.loss_function(self(x_bc), y_bc)
    
    def IC_loss(self,x_ic,y_bc):
        # Initial condition loss function
        return self.loss_function(self(x_ic),y_bc)


    def Data_loss(self, x_data, y_data):
        # Data loss concerned with the actual data measurments
        return self.loss_function(self(x_data), y_data)

    def loss(self, x_data, y_data, x_bc, y_bc,x_ic,y_ic, x_pde):
        # Total loss calculations
        loss_data = self.Data_loss(x_data, y_data)
        loss_phys = self.PDE_loss(x_pde)
        loss_bc = self.BC_loss(x_bc, y_bc)
        loss_ic=self.IC_loss(x_ic,y_ic)
        total_loss = loss_data + loss_phys + loss_bc + loss_ic
        return total_loss,loss_data,loss_phys,loss_bc,loss_ic
    


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
    
        