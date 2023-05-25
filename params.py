###
### Specify input/model/training/testing parameters
###

###
### Input data
###
inputData = 'seqMNIST' # Keyword for input/label data
inputDir = './data/'    # Storage directory for input/label data

###
### ESN properties
###
esnDir = '.' # Directory to store ESN response
alpha = 0.1  # Decay rate of ESN units [0,1]
rho = 0.99   # Scale ESN-ESN weights
gamma = 0.1  # Scale input-ESN weights
N_av = 10    # Fan-out no. from ESN units
N_esn = 1000      # No. of ESN units
esn_tau = 1.0 / alpha / (1.0 - rho) # Maimum timesclae of ESN dynamics

###
### Metric (MET) layer properties
###
METflag = True               # Include MET layer
METclass = 'METlin'          # MET layer model class [METlin - linear, METleaky - leaky integrator]
METsave = True               # Save MET validation responses for analyses
METsaveN = 100               # No. of save-points throughout MET layer training.
N_met = 100                 # No. MET units
met_train_method = 'triplet' # Training method for metric learning [triplet; HalvagalZenke]
eta_met = 0.001               # Learning rate for ESN-MET weights

###
### Output layer and training properties
###
OUTsave = False         # Save output validation responses for analyses
batch_size = 32        # No. input samples per weight/threshold update
N_batch = 5000        # No. training batches
N_check = 50            # No. validation checks throughout training
eta_out = 0.01             # Learning rate for ESN-OUT/MET-OUT weights
out_train_method = 'dense' # Method to train output weights/thresholds [dense, sparce]
n_fb = -2 # Feedback from this layer

###
### Readout properties
###
outsPerTime = True # Specifiy whether use a readout weight per time point per class (True)
                    # or just one weight per class (False)
reportTime = 100 # Real time Accuracy/Loss report every reportTime time steps

###
### Validation
###
N_val = 10000 # No. data samples for validation (taken from training set)
