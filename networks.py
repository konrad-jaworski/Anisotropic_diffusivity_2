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
        # Iterator for optimization 
        self.iter = 0

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
        
        self.a = torch.tensor([1.0], requires_grad=True).float()
        self.a = nn.Parameter(self.a)
        
    def forward(self,x):
        # Coordinates to our network comes in format [t, y, x]
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
            
        # Remain data copy for the second derivative estimation
        lap_data=x.clone()
            
        for i in range(len(self.layers_temp)-2):
            u = self.linears_temp[i](x)
            x = self.activation(u)
                
        u = self.linears_temp[-1](x)

        for i in range(len(self.layers_lap)-2):                
            z = self.linears_lap[i](lap_data)
            lap_data = self.activation(z)
                
        u_zz = self.linears_lap[-1](lap_data)
            
        return u,u_zz
        
    def loss_data(self,x,y):
        temps,_=self(x)
        loss_val=self.loss_function(temps,y)
        return loss_val
        
    def loss_PDE(self,x_pde):
        a=self.a

        g=x_pde.clone()
        g.requires_grad=True
            
        # Our differential function
        u,u_zz=self(g)

        # First derivative
        u_t_x=torch.autograd.grad(u,g,torch.ones_like(u).to(device),retain_graph=True,create_graph=True)[0]

        # Second derivative
        u_tt_xx=torch.autograd.grad(u_t_x,g,torch.ones_like(u_t_x).to(device),create_graph=True)[0]

        # Sepration of derivatives
        u_t=u_t_x[:,0]
        u_yy=u_tt_xx[:,1]
        u_xx=u_tt_xx[:,2]

        f=u_t-a*(u_xx+u_yy+u_zz)

        loss_val=torch.mean(f**2)

        return loss_val
        
    def loss(self,x,y,x_pde):
        loss_u=self.loss_data(x,y)
        loss_f=self.loss_PDE(x_pde)

        loss_val=loss_u+loss_f
        return loss_val,loss_u,loss_f
    
        