import torch
import numpy as np
import math as ma
from torch import nn
from torch import optim


class ESN(nn.Module):

    def __init__(self, N, N_i, N_av, alpha, rho, gamma):
        super().__init__()

        self.N = N
        self.alpha = alpha
        self.rho = rho
        self.N_av = N_av
        self.N_i = N_i
        self.gamma = gamma

        dilution = 1-N_av/N
        W = np.random.uniform(-1, 1, [N, N])
        W = W*(np.random.uniform(0, 1, [N, N]) > dilution)
        eig = np.linalg.eigvals(W)
        self.W = torch.from_numpy(
            self.rho*W/(np.max(np.absolute(eig)))).float()

        self.x = []

        if self.N_i == 1:

            self.W_in = 2*np.random.randint(0, 2, [self.N_i, self.N])-1
            self.W_in = torch.from_numpy(self.W_in*self.gamma).float()

        else:

            self.W_in = np.random.randn(self.N_i, self.N)
            self.W_in = torch.from_numpy(self.gamma*self.W_in).float()

    def Reset(self, s):

        batch_size = np.shape(s)[0]
        self.x = torch.zeros([batch_size, self.N])

    def ESN_1step(self, s):

        self.x = (1-self.alpha)*self.x+self.alpha * \
            torch.tanh(torch.matmul(s, self.W_in)+torch.matmul(self.x, self.W))

    def ESN_response(self, Input):

        T = Input.shape[2]
        X = torch.zeros(Input.shape[0], self.N, T)

        self.Reset(Input[:, 0])

        batch = np.max([Input.size()[0], 10000])
        N_b = np.int(np.ceil(Input.size()[0]/batch))

        for n in range(N_b):

            for t in range(T):

                if n == N_b-1:
                    self.ESN_1step(Input[n*batch:, :, t])
                    X[n*batch:, :, t] = torch.clone(self.x)

                else:
                    self.ESN_1step(Input[n*batch:(n+1)*batch, :, t])
                    X[n*batch:(n+1)*batch, :, t] = torch.clone(self.x)

        return X


class Classification_ReadOuts:

    def __init__(self, N, N_class, batch_size, outsPerTime=True, T=0):

        self.N = N
        self.N_class = N_class
        self.batch_size = batch_size
        self.loss = []
        self.opt_theta = []
        self.outsPerTime = outsPerTime
        self.T = T
        if not self.outsPerTime:
            t_weights = torch.from_numpy(np.array([[np.exp(-np.arange(T, 0, -1) * 5.0 / T)]]))
            self.t_weights = torch.tile(t_weights,[batch_size, N])

    def Dense_Initialise(self, eta):

        self.loss = nn.BCEWithLogitsLoss()
        self.Ws = nn.Parameter((2*torch.rand([self.N, self.N_class])-1)/(self.N/10))
        self.opt = optim.Adam([{'params': self.Ws, 'lr': eta}])

    def Dense_Step(self, state, y_true):
        loss = nn.BCEWithLogitsLoss()

        if self.outsPerTime:
            y = torch.matmul(state, self.Ws)
        else:
            y = torch.matmul(torch.sum(torch.mul(state, self.t_weights), 2), self.Ws)

        error = loss(y, y_true)
        error.backward()

        self.opt.step()
        self.opt.zero_grad()

        return y, error

    def Dense_Evaluate(self, State, y_true):

        y = torch.zeros([State.size()[0], self.N_class])
        Acc = 0
        error = 0

        loss = nn.BCEWithLogitsLoss()

        batch_s = np.min([State.size()[0], 6000])
        N_batch = int(np.ceil(State.size()[0]/batch_s))

        for n in range(N_batch):
            n_end = int(np.min([(n+1)*batch_s, State.size()[0]]))
            state = torch.clone(State[n*batch_s:n_end, :])
            
            if self.outsPerTime:
                y[n*batch_s:n_end,:] = torch.matmul(state, self.Ws)
            else:
                y[n*batch_s:n_end,:] = torch.matmul(torch.sum(torch.mul(state, self.t_weights), 2), self.Ws)
            
            error = error+(loss(y[n*batch_s:n_end,:], y_true[n*batch_s:n_end,:])).detach() * \
                state.size()[0]/State.size()[0]

            Acc = Acc + torch.mean(torch.eq(torch.argmax(y[n*batch_s:n_end,:], dim=1), torch.argmax(y_true[n*batch_s:n_end,:], dim=1)).float())*state.size()[0]/State.size()[0]

        return y, Acc, error

    def SpaRCe_Initialise(self, X_tr, eta):

        Pns = 0.5
        self.loss = nn.BCEWithLogitsLoss()

        ind = np.random.default_rng().choice(X_tr.shape[0], size=1000, replace=False)
        theta_g_start = np.percentile(np.abs(X_tr[ind,:]), Pns, 0)
        self.theta_g = torch.from_numpy(theta_g_start).float()
        self.theta_i = nn.Parameter(torch.zeros([self.N]))

        self.Ws = nn.Parameter(
            (2*torch.rand([self.N, self.N_class])-1)/(self.N/10))

        self.opt = optim.Adam([{'params': self.Ws, 'lr': eta}, {
                        'params': self.theta_i, 'lr': eta/10}])

    def SpaRCe_Step(self, state, y_true):

        state_sparse = []
        y = []
        error = []

        loss = nn.BCEWithLogitsLoss()

        state_sparse = torch.sign(state) * \
            torch.relu(torch.abs(state)-self.theta_g-self.theta_i)

        if self.outsPerTime:
            y = torch.matmul(state_sparse, self.Ws)
        else:
            y = torch.matmul(torch.sum(torch.mul(state_sparse, self.t_weights), 2), self.Ws)        

        error = loss(y, y_true)

        error.backward()

        self.opt.step()
        self.opt.zero_grad()

        return y, error

    def SpaRCe_Evaluate(self, State, y_true, Return_S=False):

        if Return_S == True:
            state_sparse = torch.zeros([State.size()[0], self.N])
        y = torch.zeros([State.size()[0], self.N_class])
        Acc = 0
        error = 0
        sparsity = 0

        loss = nn.BCEWithLogitsLoss()

        N_cl = torch.sum(State != 0)

        batch_s = np.min([State.size()[0], 6000])
        N_batch = int(np.ceil(State.size()[0]/batch_s))

        if Return_S == False:
            state_sparse = False

        for n in range(N_batch):
            n_end = int(np.min([(n+1)*batch_s, State.size()[0]]))
            state = torch.clone(State[n*batch_s:n_end, :])

            S_new = (torch.sign(state)*torch.relu(torch.abs(state) -
                        self.theta_g-self.theta_i)).detach()

            if Return_S == True:
                state_sparse[n*batch_s:n_end, :] = torch.clone(S_new)

            if self.outsPerTime:
                y[n*batch_s:n_end,:] = torch.matmul(S_new, self.Ws).detach()
            else:
                y[n*batch_s:n_end,:] = torch.matmul(torch.sum(torch.mul(S_new, self.t_weights), 2), self.Ws).detach()

            error = error + (loss(y[n*batch_s:n_end,:], y_true[n*batch_s:n_end,:])).detach() * \
                state.size()[0]/State.size()[0]

            Acc = Acc + torch.mean(torch.eq(torch.argmax(y[n*batch_s:n_end, :], dim=1), torch.argmax(
                y_true[n*batch_s:n_end, :], dim=1)).float())*state.size()[0]/State.size()[0]

            sparsity = sparsity + torch.sum(S_new != 0)/N_cl

        return y, Acc, error, sparsity, state_sparse


class train:
    def __init__(self, N_batch, N_check, batch_size):
        self.N_batch = N_batch       # No. of minibatches used to train
        self.N_check = N_check       # No. of validations throughout training
        self.batch_size = batch_size  # No. samples per batch

        # Batch IDs at which we validate
        self.N_checks = np.hstack((np.arange(0, self.N_check+1,10),N_batch-2,N_batch-1)) #* int(np.floor(self.N_batch/self.N_check/100))
        print(f'N_checks = {self.N_checks}')
        # Placeholder for accuracy (Validation and testing) across training
        self.ACC = np.zeros([2, N_check])
        # Placeholder for sparsity (Validation and testing) across training
        self.CL = np.zeros([2, N_check])

        self.index_help = 0

    def dense_train(self, outs, Z_tr, Y_tr, Z_val, Y_val, Z_te, Y_te):
        for n in range(self.N_batch):
            if ma.floor(n%(self.N_batch/10))==0:
                print(f'----MODELS.PY: dense training batch {n}/{self.N_batch}')
            # Choose samples to train in this batch
            rand_ind = np.random.randint(
                0, np.shape(Z_tr)[0], (self.batch_size,))
            # Extract ESN states for this batch's samples
            state = torch.clone(Z_tr[rand_ind, :])
            # Extract data labels for this batch's samples
            labels = torch.clone(Y_tr[rand_ind, :])

            if n > 0:
                y, error = outs.Dense_Step(state, labels)
            
            if (n == self.N_checks[self.index_help]) or (n == (self.N_batch-1)):
                with torch.no_grad():
                    images = torch.clone(Z_te[:, :])
                    labels = torch.clone(Y_te[:, :])

                    Out_te, Acc_te, Error_te = outs.Dense_Evaluate(images, labels)

                    images = torch.clone(Z_val[:, :])
                    labels = torch.clone(Y_val[:, :])

                    Out_val, Acc_val, Error_val = outs.Dense_Evaluate(
                        images, labels)

                    self.ACC[0, self.index_help] = np.copy(Acc_val.detach())
                    self.ACC[1, self.index_help] = np.copy(Acc_te.detach())

                    print('Iteration: ', n, 'Dense VAL: ', Acc_val.detach(),
                        'TE: ', Acc_te.detach(), 'Error: ', Error_val.detach(), Error_te.detach())
                    
                    if self.index_help<(self.N_check-1):
                        self.index_help = self.index_help+1
            
    def sparce_train(self, outs, Z_tr, Y_tr, Z_val, Y_val, Z_te, Y_te):
        for n in range(self.N_batch):
            if ma.floor(n%(self.N_batch/10))==0:
                print(f'----MODELS.PY: SpaRCe training batch {n}/{self.N_batch}')
            rand_ind = np.random.randint(0, np.shape(Z_tr)[0],(self.batch_size,))

            state = torch.clone(Z_tr[rand_ind,:])

            labels = torch.clone(Y_tr[rand_ind,:])

            if n>0:
                y, error = outs.SpaRCe_Step(state,labels)

            if (n == self.N_checks[self.index_help]) or (n == (self.N_batch-1)):
                with torch.no_grad():
                    images = torch.clone(Z_te[:,:])
                    labels = torch.clone(Y_te[:,:])

                    Out_te, Acc_te, Error_te, Sp_te, State_sp_te = outs.SpaRCe_Evaluate(images,labels)
                    
                    images = torch.clone(Z_val[:,:])
                    labels = torch.clone(Y_val[:,:])

                    Out_val, Acc_val, Error_val, Sp_val, State_sp_val = outs.SpaRCe_Evaluate(images,labels)
                    
                    self.ACC[0, self.index_help] = np.copy(Acc_val.detach())
                    self.ACC[1, self.index_help] = np.copy(Acc_te.detach())
                    self.CL[0, self.index_help] = np.copy(Sp_val.detach())
                    self.CL[1, self.index_help] = np.copy(Sp_te.detach())
                    
                    
                    print('Iteration: ',n,'SpaRCe VAL: ', Acc_val.detach(),
                    'TE: ', Acc_te.detach(), 'Error: ',Error_val.detach(), Error_te.detach(), 'Coding ',Sp_val, Sp_te)
                    
                    if self.index_help<(self.N_check-1):
                        self.index_help = self.index_help+1
