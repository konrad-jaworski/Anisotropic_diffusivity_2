import numpy as np
import torch
from scipy.stats import qmc

class DomainDataset:
    """
    Provides collocation points for PINN PDE training.
    Supports Sobol, Latin Hypercube, or uniform random sampling.
    """
    def __init__(self, n_samples=1024, n_dim=3, method='lhs'):
        self.n_dim = n_dim
        self.n_samples = n_samples
        self.method = method
        self.resample()      # generate initial points

    def resample(self):
        if self.method == 'sobol':
            engine = torch.quasirandom.SobolEngine(
                dimension=self.n_dim,
                scramble=True            
            )
            samples = engine.draw(self.n_samples).float()

        elif self.method == 'lhs':
            sampler = qmc.LatinHypercube(d=self.n_dim)
            samples = sampler.random(self.n_samples)
            samples = torch.tensor(samples, dtype=torch.float32)

        elif self.method == 'random':
            samples = torch.rand(self.n_samples, self.n_dim)

        else:
            raise ValueError("method must be 'sobol', 'lhs', or 'random'")

        self.samples = samples
        return samples

class DataGeneration:
    def __init__(self,data_cube,n_interrior=None,n_initial=None,n_boundary=None):
        # With normalization of temperature in place
        self.data_cube=(data_cube-data_cube.min())/(data_cube.max()-data_cube.min())

        self.n_interior=n_interrior
        self.n_initial=n_initial
        self.n_boundary=n_boundary

    def generate(self):

        if self.n_interior != None:
            # Sample randomly data from inside of the data space
            X_data_rand,Y_data_rand=self.sample_random_data_points(self.data_cube,self.n_interior)

            # Convert to torch tensor format
            X_data_rand=torch.from_numpy(X_data_rand).float()
            Y_data_rand=torch.from_numpy(Y_data_rand).float()

        if self.n_initial != None:
            # Sample randomly data from first frame
            X_data_ic,Y_data_ic=self.sample_random_data_points_ic(self.data_cube,self.n_initial)

            # Convert to torch tensor format
            X_data_ic=torch.from_numpy(X_data_ic).float()
            Y_data_ic=torch.from_numpy(Y_data_ic).float()

        if self.n_boundary != None:
            # Sample randomly data from boundary of the data
            X_data_bc,Y_data_bc=self.sample_random_data_points_bc(self.data_cube,self.n_boundary)
            
            # Convert to torch tensor format
            X_data_bc=torch.from_numpy(X_data_bc).float()
            Y_data_bc=torch.from_numpy(Y_data_bc).float()

        if self.n_interior != None:
            X_data=torch.concatenate([X_data_rand,X_data_ic,X_data_bc],dim=0)
            Y_data=torch.concatenate([Y_data_rand,Y_data_ic,Y_data_bc],dim=0)
        elif self.n_interior == None:
            X_data=torch.concatenate([X_data_ic,X_data_bc],dim=0)
            Y_data=torch.concatenate([Y_data_ic,Y_data_bc],dim=0)

        return X_data,Y_data

    def sample_random_data_points(self,data_cube, N_samples):
        T, Y, X = data_cube.shape
        t_idx = np.random.randint(0, T, N_samples)
        y_idx = np.random.randint(0, Y, N_samples)
        x_idx = np.random.randint(0, X, N_samples)
        z_idx = np.int16(np.zeros(N_samples)).reshape(-1,1)

        t_norm = t_idx / (T - 1)
        y_norm = y_idx / (Y - 1)
        x_norm = x_idx / (X - 1)

        X_data = np.stack([t_norm, y_norm, x_norm], axis=1)
        # We are adding column of zeros to represent depth
        X_data=np.concatenate([X_data,z_idx],axis=1) 
        
        Y_data = data_cube[t_idx, y_idx, x_idx].reshape(-1,1)
        return X_data, Y_data
    

    def sample_random_data_points_ic(self,data_cube,N_samples):
        T, Y, X = data_cube.shape
        t_idx = np.int16(np.zeros(N_samples))
        y_idx = np.random.randint(0, Y, N_samples)
        x_idx = np.random.randint(0, X, N_samples)
        z_idx = np.int16(np.zeros(N_samples)).reshape(-1,1)
    
        y_norm = y_idx / (Y - 1)
        x_norm = x_idx / (X - 1)

        X_data = np.stack([t_idx, y_norm, x_norm], axis=1)
        # We are adding column of zeros to represent depth
        X_data = np.concatenate([X_data,z_idx],axis=1) 

        Y_data = data_cube[t_idx, y_idx, x_idx].reshape(-1,1)
        return X_data, Y_data
    
    def sample_random_data_points_bc(self,data_cube,N_samples=5000):
        """
        Docstring for sample_random_data_points_bc
        
        :param data_cube: our data cube of thermography data
        :param N_samples: number of sampler per each of boundary
        """
        
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

        z_idx=np.zeros(X_data_bc.shape[0]).reshape(-1,1)

        X_data_bc=np.concatenate([X_data_bc,z_idx],axis=1)
        
        T_idx_stage_one=np.concatenate([t_idx_lower,t_idx_upper,t_idx_right,t_idx_left],axis=0)
        Y_idx_stage_one=np.concatenate([y_idx_lower,y_idx_upper,y_idx_right,y_idx_left],axis=0)
        X_idx_stage_one=np.concatenate([x_idx_lower,x_idx_upper,x_idx_right,x_idx_left],axis=0)

        Y_data_bc=data_cube[T_idx_stage_one,Y_idx_stage_one,X_idx_stage_one].reshape(-1,1)

        return X_data_bc,Y_data_bc