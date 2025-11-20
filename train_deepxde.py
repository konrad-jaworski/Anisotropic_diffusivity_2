import torch
import numpy as np
import deepxde as dde

# ------------------- DEVICE -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- DATA LOADING -------------------
data_path = r'C:\Users\stone\Desktop\Synthetic_data_no_defect\2025_10_24_sample_100x100x5mm_no_defect_isotropic_gaussian_heat.npz'
data = np.load(data_path, allow_pickle=True)
data_cube = data['data'][34:, :, :]  # shape [T, Y, X]
data_cube = (data_cube - data_cube.min()) / (data_cube.max() - data_cube.min())

T, Y, X = data_cube.shape

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

N_interior = 20000
X_data_rand, Y_data_rand = sample_random_data_points(data_cube, N_interior)

# ------------------- INITIAL CONDITION -------------------
t0 = np.zeros((Y*X,1))
y_coords = np.linspace(0,1,Y)
x_coords = np.linspace(0,1,X)
Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
X_ic = np.hstack([t0, Y_grid.reshape(-1,1), X_grid.reshape(-1,1)])
Y_ic = data_cube[0,:,:].reshape(-1,1)

# ------------------- BOUNDARY CONDITION -------------------
time_coords = np.linspace(0,1,T)
y_coords = np.linspace(0,1,Y)
x_coords = np.linspace(0,1,X)

# Left boundary x=0
X_bc_left = np.array([[t,y,0.0] for t in time_coords for y in y_coords])
Y_bc_left = data_cube[:, :, 0].reshape(-1,1)

# Right boundary x=1
X_bc_right = np.array([[t,y,1.0] for t in time_coords for y in y_coords])
Y_bc_right = data_cube[:, :, -1].reshape(-1,1)

# Bottom boundary y=0
X_bc_bottom = np.array([[t,0.0,x] for t in time_coords for x in x_coords])
Y_bc_bottom = data_cube[:,0,:].reshape(-1,1)

# Top boundary y=1
X_bc_top = np.array([[t,1.0,x] for t in time_coords for x in x_coords])
Y_bc_top = data_cube[:,-1,:].reshape(-1,1)

X_bc = np.vstack([X_bc_left, X_bc_right, X_bc_bottom, X_bc_top])
Y_bc = np.vstack([Y_bc_left, Y_bc_right, Y_bc_bottom, Y_bc_top])

# ------------------- PDE VARIABLE -------------------
# Trainable thermal diffusivity
a = dde.Variable(1.0e-3)

# ------------------- PDE DEFINITION -------------------
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=0)
    dy_xx = dde.grad.hessian(y, x, i=2)
    dy_yy = dde.grad.hessian(y, x, i=1)
    return dy_t - a*(dy_xx + dy_yy)

# ------------------- GEOMETRY -------------------
geom = dde.geometry.Rectangle([0,0],[1,1])
timedomain = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ------------------- POINTSET BCs -------------------
ic = dde.PointSetBC(X_ic, Y_ic, component=0)
bc = dde.PointSetBC(X_bc, Y_bc, component=0)
interior_data = dde.PointSetBC(X_data_rand, Y_data_rand, component=0)

# ------------------- DATASET -------------------
bc_list = [ic, bc, interior_data]

data = dde.data.TimePDE(
    geomtime,
    pde,
    bc_list,         # <- third positional argument
    num_domain=20000 # collocation points
)

# ------------------- MODEL -------------------
layers = [3, 100, 100, 100, 100, 1]
net = dde.nn.FNN(layers, "tanh", "Glorot uniform")
model = dde.Model(data, net)

# ------------------- TRAINING -------------------
# Option 1: skip metrics to avoid NoneType error
model.compile("adam", lr=1e-3, external_trainable_variables=[a])
variable_callback = dde.callbacks.VariableValue(a, period=1000)

losshistory, train_state = model.train(epochs=20000, callbacks=[variable_callback])

# Optional fine-tuning with L-BFGS
model.compile("L-BFGS", external_trainable_variables=[a])
losshistory, train_state = model.train()

print("Learned thermal diffusivity a =", a.numpy())

# Save and plot results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
