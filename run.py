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
### Precompute (or load) ESN and/or load MET responses
###
metname_tr = par.inputDir + 'met_train.pt'
metname_val = par.inputDir + 'met_validate.pt'
metname_te = par.inputDir + 'met_test.pt'
if par.METsave:
    metname_sav = par.inputDir + 'met_save.pt'

if par.METflag and exists(metname_tr):
    # Load MET data
    met_tr = torch.load(metname_tr)
    met_val = torch.load(metname_val)
    met_te = torch.load(metname_te)
    print('----RUN.PY: loaded MET data')
else:
    # Train ESN-MET weights and compute resposnes
    esnname_tr = par.inputDir + 'esn_train.pt'
    esnname_val = par.inputDir + 'esn_validate.pt'
    esnname_te = par.inputDir + 'esn_test.pt'
    if exists(esnname_tr):
        # Load ESN data
        esn_tr = torch.load(esnname_tr)
        esn_val = torch.load(esnname_val)
        esn_te = torch.load(esnname_te)
    else:
        # Initialise ESN
        esn = ESN(par.N_esn, il.N_i, par.N_av, par.alpha, par.rho, par.gamma)
        # Compute ESN responses
        esn_tr = esn.ESN_response(il.X_tr) # Training data
        esn_tr = esn.ESN_normalise(esn_tr)  # z-score each sample
        esn_val = esn.ESN_response(il.X_val) # Validation data
        esn_val = esn.ESN_normalise(esn_val)  # z-score each sample
        esn_te = esn.ESN_response(il.X_te) # Test data
        esn_te = esn.ESN_normalise(esn_te)  # z-score each sample

        # Save ESN responses
        torch.save(esn_tr, esnname_tr)
        torch.save(esn_val, esnname_val)
        torch.save(esn_te, esnname_te)

    print('----RUN.PY: generated/loaded ESN data')

###
### For both MET and output training
###
tr = train(par.N_batch, par.N_check, par.batch_size) # Initialise training

###
### Train ESN-MET weights and compute final trained MET responses
###
if par.METflag and not exists(metname_tr):
    # Initialise MET class
    if par.METclass=='METlin':
        met = METlin(par.N_esn, par.N_met, il.X_tr.shape[2], par.eta_met, par.METsave, par.N_val, par.N_check, par.N_batch)
    elif par.METclass=='METleaky':
        met = METleaky(par.N_esn, par.N_met)

    # Train MET layer
    if par.met_train_method=='triplet':
        if par.outsPerTime: # reshape to be N_batch-by-NxT
            Z_tr = torch.reshape(esn_tr, [esn_tr.size()[0], par.N_esn*il.T])
            Z_val = torch.reshape(esn_val, [esn_val.size()[0], par.N_esn*il.T])
            Z_te = torch.reshape(esn_te, [esn_te.size()[0], par.N_esn*il.T])
        tr.met_tripletloss_train(met, Z_tr, Z_val, il.label_tr, il.label_val)
    
    # Compute and save MET responses
    met_tr = met.W_esn(Z_tr)
    torch.save(met_tr, metname_tr)
    met_val = met.W_esn(Z_val)
    torch.save(met_val, metname_val)
    met_te = met.W_esn(Z_te)
    torch.save(met_te, metname_te)
    print('----RUN.PY: saved final MET responses')

    # Save MET responses throughout training
    if par.METsave:
        torch.save(met.sav, metname_sav)
        torch.save(met.wsav, par.inputDir + 'wsav.pt')
        torch.save(met.savloss, par.inputDir + 'met_loss.pt')
        dist = torch.cat((met.dpos, met.dneg),1)
        torch.save(dist, par.inputDir + 'met_distances.pt')
        print('----RUN.PY: saved MET responses throughtout training')

###
### Train output weights (and ESN thresholds)
###
# Prepare inputs to readouts (either from ESN or from MET)
# If using an output weight per time point
if par.METflag:
    Z_tr = torch.clone(met_tr)
    Z_val = torch.clone(met_val)
    Z_te = torch.clone(met_te)
    N_in = par.N_met
else:
    if par.outsPerTime: # reshape to be N_batch-by-NxT
        Z_tr = torch.reshape(esn_tr, [esn_tr.size()[0], par.N_esn*il.T])
        Z_val = torch.reshape(esn_val, [esn_val.size()[0], par.N_esn*il.T])
        Z_te = torch.reshape(esn_te, [esn_te.size()[0], par.N_esn*il.T])
        N_in = par.N_esn * il.T
    else:
        Z_tr = esn_tr
        Z_val = esn_val
        Z_te = esn_te
        N_in = par.N_esn

if par.outsPerTime:
    outs = Classification_ReadOuts(N_in, il.N_o, par.batch_size) # One weight per ESN unit and per time point
else:
    outs = Classification_ReadOuts(N_in, il.N_o, par.batch_size, par.outsPerTime, il.T) # One weight per ESN unit

#tr = train(par.N_batch, par.N_check, par.batch_size) # Initialise training
if par.out_train_method=='dense':
    outs.Dense_Initialise(par.eta_out) # Initialise outputs
    tr.dense_train(outs, Z_tr, il.Y_tr, Z_val, il.Y_val, Z_te, il.Y_te)   # Train using normal method, 'dense'

elif par.out_train_method=='sparce':
    outs.SpaRCe_Initialise(Z_tr, par.eta_out) # Initialise outputs
    tr.sparce_train(outs, Z_tr, il.Y_tr, Z_val, il.Y_val, Z_te, il.Y_te)  # Train using SpaRCe
    
###
### Test the trained model
###
with torch.no_grad():
    if par.out_train_method=='dense':
        y_out, acc, err = outs.Dense_Evaluate(Z_te, il.Y_te)
    elif par.out_train_method=='sparce':
        y_out, acc, err = outs.Sparce_Evaluate(Z_te, il.Y_te)
    print(f'----RUN.PY: Test accuracy = {acc}, Test error = {err}')

