import numpy as np
import torch
import matplotlib.pyplot as plt
from models import *
from data import *
#import torchvision
#import torchvision.datasets as datasets
import params as par
from os.path import exists

###
### Process inputs and output labels
###

if par.inputData == 'seqMNIST':
    il = seqMNIST(par) # Process input data and labels
print('----RUN.PY: processed inputs and labels')
###
### Precompute (or load) ESN response
###
fname_tr = par.inputDir + 'esn_train.pt'
fname_val = par.inputDir + 'esn_validate.pt'
fname_te = par.inputDir + 'esn_test.pt'
if exists(fname_tr):
    # Load ESN data
    il.Z_tr = torch.load(fname_tr)
    il.Z_val = torch.load(fname_val)
    il.Z_te = torch.load(fname_te)
else:
    # Initialise ESN
    esn = ESN(par.N, il.N_i, par.N_av, par.alpha, par.rho, par.gamma)
    # Compute and save ESN responses
    il.Z_tr = esn.ESN_response(il.X_tr) # Training data
    torch.save(il.Z_tr, fname_tr)

    il.Z_val = esn.ESN_response(il.X_val) # Validation data
    torch.save(il.Z_val, fname_val)

    il.Z_te = esn.ESN_response(il.X_te) # Test data
    torch.save(il.Z_te, fname_te)
print('----RUN.PY: generated/loaded ESN data')

###
### Train output weights (and ESN thresholds)
###

# Reshape of the datasets and concatenation of the responses across time. In this way, we use a different output weight for each 
# node AND time step, resulting in a number of features equal to N*T.
il.Z_tr = torch.reshape(il.Z_tr, [il.Z_tr.size()[0], par.N*il.T])
il.Z_val = torch.reshape(il.Z_val, [il.Z_val.size()[0], par.N*il.T])
il.Z_te = torch.reshape(il.Z_te, [il.Z_te.size()[0], par.N*il.T])

outs = Classification_ReadOuts(par.N*il.T, il.N_o, par.eta) # Notice how the input is the number of features

tr = train(par.N_batch, par.N_check, par.batch_size) # Initialise training
if par.train_method=='dense':
    outs.Dense_Initialise(par.eta) # Initialise outputs
    tr.dense_train(outs, il.Z_tr, il.Y_tr, il.Z_val, il.Y_val, il.Z_te, il.Y_te)   # Train using normal method, 'dense'

elif par.train_method=='sparce':
    outs.SpaRCe_Initialise(il.Z_tr, par.eta) # Initialise outputs
    tr.sparce_train(outs, il.Z_tr, il.Y_tr, il.Z_val, il.Y_val, il.Z_te, il.Y_te)  # Train using SpaRCe
    


