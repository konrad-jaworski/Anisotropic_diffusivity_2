import torch
import torch.nn as nn

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FCN(nn.Module):
    """
    Network used for prediction of thermal conductivity of the isotropic material with measurments from its surface.
    """
    def __init__(self, layers, density, specific_heat, time_scale, y_scale, x_scale):
        super(FCN, self).__init__()

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
        self.k = nn.Parameter(torch.tensor([5.0], dtype=torch.float32))
    
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
        grads = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
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

    


