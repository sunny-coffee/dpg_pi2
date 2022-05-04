import numpy as np

# start_x, start_y: the start position
duration  = 12 
std  = 0.01           
batch_size  = 30
# cost_function  
epoch      = 200
n_reuse      = 3  # number of re-used trials per update
feedback     = 0  # chooses where the position feedback must come from; 0 : virtual trajectory, 1 : actual position (simulated mass point)
n_runs       = 1  # number of times the PI2 algorithm must be run with the current parameters, for obstaining statistical results
kp0          = 20 #  
n_dim = 2
n_dim_kp = 2
n_hidden = 20
dt = 0.1
viapoints = np.array([[-7.5,-1],[-2.5,0.8],[-1.5,0.6]])
constraints = []
goal = [0,0]
plotIntermediate = 0 
stiffness_learning = 1
# n_elite_tank = 5
model = 'masspoint'
exsit_force_field = 0
feedback = 0
