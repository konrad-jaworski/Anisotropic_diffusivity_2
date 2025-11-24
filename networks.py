import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FCN(nn.Module):
    """
    Network used for prediction of thermal diffusivity of the isotropic material with measurments from its surface.
    """
    def __init__(self, layers):
        super().__init__()

        # Layers structure
        self.layers = layers

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

        # Inverse parameter (thermal diffusivity)
        self.a = nn.Parameter(torch.tensor([1e-3], dtype=torch.float32))
    
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
        # Here we are using collocation points which specificic sampling in the domain
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

        # PDE residual:
        f =  u_t - self.a * (u_xx + u_yy)
        loss_phys = torch.mean(f ** 2)

        return loss_phys

    def Data_loss(self, x_data, y_data):
        # Data loss function
        return self.loss_function(self(x_data), y_data)
    




