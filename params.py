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
alpha = 0.9  # Decay rate of ESN units [0,1]
rho = 0.99   # Scale ESN-ESN weights
gamma = 0.1  # Scale input-ESN weights
N_av = 10    # Fan-out no. from ESN units
N_esn = 100      # No. of ESN units

###
### Metric (MET) layer properties
###
METflag = True               # Include MET layer
METclass = 'METlin'          # MET layer model class [METlin - linear, METleaky - leaky integrator]
METsave = True               # Save MET validation responses for analyses
METsaveN = 100               # No. of save-points throughout MET layer training.
N_met = 1000                 # No. MET units
met_train_method = 'triplet' # Training method for metric learning [triplet; HalvagalZenke]
eta_met = 0.005               # Learning rate for ESN-MET weights

###
### Output layer and training properties
###
OUTsave = False         # Save output validation responses for analyses
batch_size = 4096        # No. input samples per weight/threshold update
N_batch = 20000        # No. training batches
N_check = 100            # No. validation checks throughout training
eta_out = 0.01             # Learning rate for ESN-OUT/MET-OUT weights
out_train_method = 'dense' # Method to train output weights/thresholds [dense, sparce]

###
### Readout properties
###
outsPerTime = True # Specifiy whether use a readout weight per time point per class (True)
                    # or just one weight per class (False)

###
### Validation
###
N_val = 10000 # No. data samples for validation (taken from training set)
