###
### Specify input/model/training/testing parameters
###

###
### Input data
###
inputData = 'seqMNIST' # Keyword for input/label data
inputDir = './data'    # Storage directory for input/label data

###
### ESN properties
###
esnDir = '.' # Directory to store ESN response
alpha = 0.9  # Scale input-ESN weights
rho = 0.99   # Scale ESN-ESN weights
gamma = 0.1  # Decay rate of ESN units [0,1]
N_av = 10    # Fan-out no. from ESN units
N = 1000     # No. of ESN units

###
### Learning properties
###
batch_size = 20        # No. input samples per weight/threshold update
N_batch = 10000        # No. training batches
N_check = 50           # No. validation checks throughout training
eta = 0.01             # Learning rate
train_method = 'sparce' # Method to train output weights/thresholds [dense, sparce]

###
### Validation
###
N_val = 10000 # No. data samples for validation (taken from training set)
