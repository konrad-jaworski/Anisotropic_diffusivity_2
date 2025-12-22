import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ThermalDiffusionPINN(nn.Module):
    def __init__(self, layers, mode='forward',
                 k=[2.0, 2.0, 2.0], Cp=[700.0, 700.0, 700.0], ro=[1600.0, 1600.0, 1600.0],
                 time_max=14.0, xy_max=0.1, z_max=0.005):
        """
        Args:
            time_max: Maximum time in physical units (for scaling)
            xy_max: Maximum x,y in physical units (for scaling)
            z_max: Maximum z in physical units (for scaling)
        """
        super().__init__()
        
        self.layers = layers
        self.mode = mode.lower()
        
        # Store MAX values for denormalization of derivatives
        self.time_max = time_max
        self.xy_max = xy_max
        self.z_max = z_max
        
        # Activation and loss
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        
        # Network layers (input is 4D: [t_norm, y_norm, x_norm, z_norm])
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])
        
        # Xavier initialization
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # --- Handle diffusivities ---
        if self.mode == 'forward':
            # Forward mode: known anisotropic diffusivities
            self.a_x_physical = k[0] / (Cp[0] * ro[0])
            self.a_y_physical = k[1] / (Cp[1] * ro[1])
            self.a_z_physical = k[2] / (Cp[2] * ro[2])
            
            # Compute SCALED diffusivities for PDE in normalized coordinates
            self.a_x_scaled = self.a_x_physical * time_max / (xy_max**2)
            self.a_y_scaled = self.a_y_physical * time_max / (xy_max**2)
            self.a_z_scaled = self.a_z_physical * time_max / (z_max**2)
            
        elif self.mode == 'inverse':
            # Inverse mode: learn SCALED diffusivities directly
            # Initialize scaled diffusivities to
            self.a_x_scaled = nn.Parameter(torch.tensor(3.0e-3)) # Our initial guess is that in plane diffusivity is much smaller than in depth
            self.a_y_scaled = nn.Parameter(torch.tensor(3.0e-3))
            self.a_z_scaled = nn.Parameter(torch.tensor(1.0))
            
    def forward(self, x_norm):
        """Forward pass with NORMALIZED inputs (0-1 range)"""
        for i in range(len(self.linears) - 1):
            x_norm = self.activation(self.linears[i](x_norm))
        return self.linears[-1](x_norm)
    
    def get_physical_diffusivities(self):
        """Convert scaled diffusivities to physical units"""
        if self.mode == 'forward':
            return self.a_x_physical, self.a_y_physical, self.a_z_physical
        else:  # inverse
            a_x_physical = self.a_x_scaled * (self.xy_max**2 / self.time_max)
            a_y_physical = self.a_y_scaled * (self.xy_max**2 / self.time_max)
            a_z_physical = self.a_z_scaled * (self.z_max**2 / self.time_max)
            return a_x_physical, a_y_physical, a_z_physical
    
    def PDE_loss(self, x_norm):
        """
        x_norm: [N, 4] → [t_norm, y_norm, x_norm, z_norm] where norm ∈ [0,1]
        Compute PDE loss in NORMALIZED coordinates
        """
        x_norm = x_norm.clone().detach().requires_grad_(True)
        u = self(x_norm)  # Temperature prediction
        
        # Compute first derivatives (w.r.t NORMALIZED coordinates)
        grads = torch.autograd.grad(u, x_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        u_t_norm = grads[:, 0]  # ∂u/∂t_norm
        u_y_norm = grads[:, 1]  # ∂u/∂y_norm
        u_x_norm = grads[:, 2]  # ∂u/∂x_norm
        u_z_norm = grads[:, 3]  # ∂u/∂z_norm
        
        # Compute second derivatives
        u_yy_norm = torch.autograd.grad(u_y_norm, x_norm, grad_outputs=torch.ones_like(u_y_norm), create_graph=True)[0][:, 1]
        
        u_xx_norm = torch.autograd.grad(u_x_norm, x_norm,grad_outputs=torch.ones_like(u_x_norm),create_graph=True)[0][:, 2]
        
        u_zz_norm = torch.autograd.grad(u_z_norm, x_norm, grad_outputs=torch.ones_like(u_z_norm), create_graph=True )[0][:, 3]
        
        # Heat equation in NORMALIZED coordinates:
        # ∂u/∂t_norm = a_x_scaled * ∂²u/∂x_norm² + a_y_scaled * ∂²u/∂y_norm² + a_z_scaled * ∂²u/∂z_norm²
        
        f = u_t_norm - (self.a_x_scaled * u_xx_norm + self.a_y_scaled * u_yy_norm + self.a_z_scaled * u_zz_norm)
        
        loss_phys = torch.mean(f ** 2)
        return loss_phys
    
    def Data_loss(self, x_norm, y_data):
        """Data loss with normalized inputs"""
        y_pred = self(x_norm)
        return self.loss_function(y_pred, y_data)
    

    