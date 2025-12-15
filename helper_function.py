import numpy as np
import torch

class DataGenerator:
    def __init__(self,data):
        """
        Constructor of data generot class
        data: torch tensor of shape (T,size_y,size_x) !Must be Normalized between 0 and 1 

        """
        self.data=data
    
    def get_indexes_latin(self, n_points):
        """
        Generate n_points unique 2D integer indices on a grid (size_y, size_x)
        using true stratified sampling (2D Latin Hypercube style),
        strictly inside the boundaries (not touching edges).
        """

        size_y = self.data.size(1)
        size_x = self.data.size(2)

        # Determine number of strata along each axis (roughly square)
        n_sqrt = int(torch.ceil(torch.sqrt(torch.tensor(n_points, dtype=torch.float32))))
        n_y = min(n_sqrt, size_y - 2)  # leave 1 pixel margin
        n_x = min(int(torch.ceil(torch.tensor(n_points, dtype=torch.float32) / n_y)), size_x - 2)

        # Compute stratum edges (avoid touching boundaries)
        y_edges = torch.linspace(1, size_y - 1, n_y + 1)
        x_edges = torch.linspace(1, size_x - 1, n_x + 1)

        points = []
        for i in range(n_y):
            for j in range(n_x):
                if len(points) >= n_points:
                    break

                # Pick a random point inside the stratum
                y = torch.randint(int(y_edges[i]), int(y_edges[i + 1]), (1,)).item()
                x = torch.randint(int(x_edges[j]), int(x_edges[j + 1]), (1,)).item()

                points.append((y, x))

        # Convert to tensors
        points = torch.tensor(points[:n_points])
        Y, X = points[:, 0], points[:, 1]

        return X, Y

    
    def sub_sample(self,X,Y,sub_sample=1,frame_rate=300):
        # This methods subsample the time data axis, which allows for coarse selection od data points in order to make faster training.
        T=self.data.size(0)
        temp=self.data[::sub_sample,Y,X]
        temp_len=temp.size(0)
        t_v=torch.linspace(0,T/frame_rate,temp_len) # This produce for us only physical scale of time
        return temp, t_v
    
    def network_format(self,X,Y,sub_sample=1,frame_rate=300,phys_scale=0.1):
        # Takes data coordinates and reformulate to physical scales and Network frendly format.

        # 1. Sample the data
        temp, t_v = self.sub_sample(X, Y, sub_sample, frame_rate)  # temp shape: (T, n_points)
        
        # 2. Convert indices to physical units for network 
        
        # Debugging line
        # Yf = Y.to(torch.float32)
        # Xf = X.to(torch.float32)
        
        _,N_y,N_x=self.data.size()
        Yf = Y*phys_scale/N_y
        Xf = X*phys_scale/N_x

        # 3. Use meshgrid over time and point index
        n_points=X.size(0)
        point_idx = torch.arange(n_points).to(torch.float32)
        
        # Debugging line
        # t_v=torch.linspace(0,3304,3304)

        Tt, P = torch.meshgrid(t_v, point_idx, indexing='ij')  # shape: (T, n_points)
        P = P.to(torch.long)  # index must be integer

        # 4. Gather corresponding Y, X for each point
        Yy = Yf[P]  # shape: (T, n_points)
        Xx = Xf[P]  # shape: (T, n_points)

        # 5. Flatten in row-major order to match temp.flatten()
        Tt = Tt.reshape(-1, 1)
        Yy = Yy.reshape(-1, 1)
        Xx = Xx.reshape(-1, 1)
        Zz = torch.zeros_like(Xx)

        # 6. Network input
        X_net = torch.cat([Tt, Yy, Xx, Zz], dim=1)

        # 7. Flatten target in **same order**
        Y_net = temp.reshape(-1, 1)

        # Our X_net data are in physical unit sense
        return X_net, Y_net

    def boundary_data(self,boundary_division=4):
        
        # 1. Generate boundary points
        _,size_y,size_x=self.data.size()
        x=torch.linspace(0,size_x-1,boundary_division)
        y=torch.linspace(0,size_y-1,boundary_division)

        # 2. Fill in the x axis coordinates
        P_lr=torch.ones(2*boundary_division,2)*(size_x-1)
        P_lr[0:boundary_division,0]=0.0
        j=0
        for i in range(2*boundary_division):
            if j==boundary_division:
                j=0
            P_lr[i,1]=x[j]
            j=j+1

        # 3. Fill in the y axis coordinates
        P_tb=torch.ones(2*boundary_division,2)*(size_y-1)
        P_tb[0:boundary_division,1]=0.0

        j=0
        for i in range(2*boundary_division):
            if j==boundary_division:
                j=0
            P_tb[i,0]=y[j]
            j=j+1

        # 4. Combine them
        P=torch.concatenate([P_lr,P_tb],dim=0)
        P=P.round().long()

        Px=P[:,0]
        X=Px.to(torch.int)
        Py=P[:,1]
        Y=Py.to(torch.int)

        return X, Y
    
    def lhs(self,n_samples, n_dims):
        """
        Latin Hypercube Sampling in [0,1]^n_dims.
        Returns tensor of shape (n_samples, n_dims).
        """
        # Divide into equal strata
        cut = torch.linspace(0, 1, n_samples + 1)

        # Sample one point uniformly from each stratum per dimension
        u = torch.rand(n_samples, n_dims)
        samples = cut[:-1].unsqueeze(1) + (u * (1.0 / n_samples))

        # Shuffle within each dimension
        for dim in range(n_dims):
            perm = torch.randperm(n_samples)
            samples[:, dim] = samples[perm, dim]

        return samples
    
    def set_initial_condition_points(self,n_points,phys_scales=[0.1,0.1,0.005]):

        X = self.lhs(n_samples=n_points, n_dims=3)
        phys_scales=torch.tensor(phys_scales)
        X_phys=X*phys_scales

        time_axis=torch.zeros((n_points,1))
        X_net=torch.concatenate([time_axis,X_phys],dim=1)

        Y_net=torch.tensor([0])

        return X_net, Y_net
    
    def set_collocation_points(self,N_t,N_space,phys_scales=[14.0,0.1,0.1,0.005]):
        
        phys_scales = torch.tensor(phys_scales)

        # Time points (flattened)
        time_axis = self.lhs(n_samples=N_t, n_dims=1).flatten() * phys_scales[0]

        # Space points (same for all times)
        space_points = self.lhs(n_samples=N_space, n_dims=3) * phys_scales[1:4]

        return time_axis,space_points
    
    def time_stepping(self,time_axis,space_points,index):
        proxy_time=torch.ones((space_points.size(0),1)).to(torch.float32)*time_axis[index]
        proxy_time.size()
        step=torch.concatenate([proxy_time,space_points],dim=1)

        # Monter-Carlo space time coverage witout explosion 
        # idx = torch.randint(0, N_t, (1,))
        # X = time_stepping(time_axis, space_points, idx)
        return step