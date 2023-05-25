###
### Specify input/model/training/testing parameters
###
import numpy as np

###
### Input data
###
inputData = 'seqMNIST' # Keyword for input/label data
inputDir = './data/'    # Storage directory for input/label data

###
### Training properties
###
nEpochs=5000#5000 # No optimiser iterations
batch_size=40 # No. input samples per optimser iteration
eta = 0.0001 # Optimser learning rate
eta_tau = np.Inf#5000.0 # Time constant for learning rate decay
nClass=10 # No. classes in dataset
tMax = 28 # No. time steps per input sample
tauPerf = tMax / 2 # Time scale for weighting responses for accuracy/loss

###
### ESN properties
###
esnDir = '.' # Directory to store ESN response
alpha = 0.1  # Decay rate of ESN units [0,1]
rho = 0.99   # Scale ESN-ESN weights
gamma = 0.1  # Scale input-ESN weights
N_av = 10    # Fan-out no. from ESN units
N_esn = 1000      # No. of ESN units
nInputs = 28      # No. input features
esn_tau = 1.0 / alpha / (1.0 - rho) # Maimum timesclae of ESN dynamics
fbLayer = 3 # Feedback layer

###
### Metric (MET) learning properties
###
METflag = True               # Include MET layer
METclass = 'METlin'          # MET layer model class [METlin - linear, METleaky - leaky integrator]
METsave = True               # Save MET validation responses for analyses
METsaveN = 100               # No. of save-points throughout MET layer training.
metricLossType = 'tripletLoss' # Training method for metric learning [triplet; HalvagalZenke]
eta_met = 0.0001             # Learning rate for ESN-MET weights
margin=2                     # For triplet margin loss
maxLayer = 3                 # Do forward pass up to this layer (base 0)
lossLayer = 3                # Compute loss from responses in this layer (base 0)
                             #   Likely best to go no higher than maxLayer

###
### Classification learning properties
###
OUTsave = False         # Save output validation responses for analyses 
N_batch = 5000        # No. training batches
N_check = 50            # No. validation checks throughout training
eta_out = 0.01             # Learning rate for ESN-OUT/MET-OUT weights
out_train_method = 'dense' # Method to train output weights/thresholds [dense, sparce]

###
### Feedforward properties
###
Ns = [N_esn, 200, 100, nClass] # No. neurons in each layer

###
### Readout and save properties
###
outsPerTime = True # Specifiy whether use a readout weight per time point per class (True)
                    # or just one weight per class (False)
reportTime = 25 # Real time Accuracy/Loss report every reportTime time steps
saveLayers = [0, 2]
nSave = 25 # No. time points to save responses
nSaveMaxT = nEpochs #5000 # Save responses up to this iteration
nSaveSamples = 10 # No. samples to save per class
saveFlag_RESP = True # Save responses
saveFlag_FBWeights = False if not fbLayer else True # Save feedback weights


###
### Validation
###
N_val = 10000 # No. data samples for validation (taken from training set)
