# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:35:55 2022

@author: lucam
"""

import torch
import numpy as np
from torch import nn
from torch import optim


    

class ESN(nn.Module):
    
    def __init__(self,N,N_in,N_av,alpha,rho,gamma):
        super().__init__()
        
        
        self.N=N
        self.alpha=alpha
        self.rho=rho
        self.N_av=N_av
        self.N_in=N_in
        self.gamma=gamma
        
        diluition=1-N_av/N
        W=np.random.uniform(-1,1,[N,N])
        W=W*(np.random.uniform(0,1,[N,N])>diluition)
        eig=np.linalg.eigvals(W)
        self.W=torch.from_numpy(self.rho*W/(np.max(np.absolute(eig)))).float()
        
        
        self.x=[]
        
        if self.N_in==1:
            
            self.W_in=2*np.random.randint(0,2,[self.N_in,self.N])-1
            self.W_in=torch.from_numpy(self.W_in*self.gamma).float()
            
            
        else:
            
            self.W_in=np.random.randn(self.N_in,self.N)
            self.W_in=torch.from_numpy(self.gamma*self.W_in).float()
            
        
        
        
    def Reset(self,s):
        
        batch_size=np.shape(s)[0]
        self.x=torch.zeros([batch_size,self.N])
        
    def ESN_1step(self,s):
        
        self.x=(1-self.alpha)*self.x+self.alpha*torch.tanh(torch.matmul(s,self.W_in)+torch.matmul(self.x,self.W))
        
    def ESN_response(self,Input):
        
        T=Input.shape[2]
        X=torch.zeros(Input.shape[0],self.N,T)
        
        self.Reset(Input[:,0])
        
        batch=np.max([Input.size()[0],10000])
        N_b=np.int(np.ceil(Input.size()[0]/batch))
        
        for n in range(N_b):
            
            for t in range(T):
                
                if n==N_b-1:
                    self.ESN_1step(Input[n*batch:,:,t])
                    X[n*batch:,:,t]=torch.clone(self.x)
                    
                else:
                    self.ESN_1step(Input[n*batch:(n+1)*batch,:,t])
                    X[n*batch:(n+1)*batch,:,t]=torch.clone(self.x)
                    
                
            
            
        return X
    
            

        
class Classification_ReadOuts:
    
    def __init__(self,N,N_class,batch_size):
        
        
        self.N=N
        
        self.N_class=N_class
                
        self.batch_size=batch_size
                
        self.Ws=[]
        
        self.theta_g=[]
        
        self.theta_i=[]
        
        self.loss=[]
        self.opt=[]
        self.opt_theta=[]
        
        self.N_copies=[]
        
    
    def Initialise_Online(self,scan,N_copies,alpha_size):
        
        
        self.N_copies=N_copies
        if scan==False:
            
            self.N_copies=1
            alpha_sizes=[alpha_size]
                               
        
        if scan==True:
            
            alpha_sizes=0.01*2**(-np.linspace(0,7,self.N_copies))

        
        self.loss = nn.BCEWithLogitsLoss()
        
        for i in range(self.N_copies):
            
            
            self.Ws.append(nn.Parameter( (2*torch.rand([self.N,self.N_class])-1)/(self.N/10)))
            self.opt.append(optim.Adam([{'params': self.Ws, 'lr':alpha_sizes[i] }]))
        
        
    
    def Online_Step(self,state,y_true):
        
       
        y=[]
        error=[]
        
        loss = nn.BCEWithLogitsLoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]) )
            
            error.append(loss(y[i],y_true))
            error[i].backward()

        
            self.opt[i].step()
            self.opt[i].zero_grad()
            
            
                
        return y, error
    
    
    def Online_Evaluate(self,State,y_true):
    
        y=torch.zeros([self.N_copies,State.size()[0],self.N_class])
        Acc=torch.zeros([self.N_copies])    
        error=torch.zeros([self.N_copies])              
        
        loss = nn.BCEWithLogitsLoss()
        
        batch_s=np.min([State.size()[0],6000])
        N_batch=int(np.ceil(State.size()[0]/batch_s))
        
        for i in range(self.N_copies):
            
            for n in range(N_batch):
                
                n_end=int(np.min([(n+1)*batch_s,State.size()[0]]))
                state=torch.clone(State[n*batch_s:n_end,:])
            
                y[i,n*batch_s:n_end,:]=torch.matmul(state,self.Ws[i]).detach()

                error[i]=error[i]+(loss(y[i],y_true)).detach()*state.size()[0]/State.size()[0]

                Acc[i]=Acc[i]+torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() )*state.size()[0]/State.size()[0]

            
        return y, Acc, error
    


    def Initialise_SpaRCe(self,X_tr,alpha_size):
        
        
        Pns=[0,10,20,30,40,50,60,70,80,90]
        
        self.N_copies=np.shape(Pns)[0]
        
        self.loss = nn.BCEWithLogitsLoss()
        
        
        for i in range(self.N_copies):
            
            theta_g_start=np.percentile(np.abs(X_tr),Pns[i],0)
            
            self.theta_g.append(torch.from_numpy(theta_g_start).float())
            
            self.theta_i.append(nn.Parameter(torch.zeros([self.N])))
            
            self.Ws.append(nn.Parameter( (2*torch.rand([self.N,self.N_class])-1)/(self.N/10) ))
            
            self.opt.append(optim.Adam([{'params': self.Ws, 'lr':alpha_size },{'params': self.theta_i, 'lr':alpha_size/10 }]))
            
            
    
    def SpaRCe_Step(self,state,y_true):
        
        
        state_sparse=[]
        y=[]
        error=[]
        
        loss = nn.BCEWithLogitsLoss()

        for i in range(self.N_copies):
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     

            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            error[i].backward()

            self.opt[i].step()
            self.opt[i].zero_grad()
    
            
        return y, error
    
    def SpaRCe_Evaluate(self,State,y_true, Return_S=False):
        
        if Return_S==True:
            state_sparse=torch.zeros([self.N_copies,State.size()[0],self.N])
        y=torch.zeros([self.N_copies,State.size()[0],self.N_class])
        Acc=torch.zeros([self.N_copies])    
        error=torch.zeros([self.N_copies])              
        sparsity=torch.zeros([self.N_copies])  
        
        loss = nn.BCEWithLogitsLoss()
        
        N_cl=torch.sum(State!=0)        
        
        batch_s=np.min([State.size()[0],6000])
        N_batch=int(np.ceil(State.size()[0]/batch_s))
        
        if Return_S==False:
            state_sparse=False
        
        for i in range(self.N_copies):
            
            for n in range(N_batch):
                
                n_end=int(np.min([(n+1)*batch_s,State.size()[0]]))
                state=torch.clone(State[n*batch_s:n_end,:])
                
                S_new=(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i])).detach()     
                
                if Return_S==True:
                    
                    state_sparse[i,n*batch_s:n_end,:]=torch.clone(S_new)

                y[i,n*batch_s:n_end,:]=torch.matmul(S_new,self.Ws[i]).detach()

                error[i]=error[i]+(loss(y[i],y_true)).detach()*state.size()[0]/State.size()[0]

                Acc[i]=Acc[i]+torch.mean( torch.eq(torch.argmax(y[i,n*batch_s:n_end,:],dim=1),torch.argmax(y_true[n*batch_s:n_end,:],dim=1)).float() )*state.size()[0]/State.size()[0]
                                
                sparsity[i]=sparsity[i]+torch.sum(S_new!=0)/N_cl
        
        
        return y, Acc, error, sparsity, state_sparse
            


class Pruning:


    def __init__(self, W_Best, theta_g_Best, theta_i_Best, N_copies):
        
        
        self.N=W_Best.size()[0]
        
        self.N_class=W_Best.size()[1]
                
        self.N_copies=N_copies
        
        self.W_Best=W_Best
        
        self.Ws=[]
        self.theta_g=[]
        self.theta_i=[]
        
        for i in range(N_copies):         
            
            self.Ws.append(W_Best)
        
            self.theta_g.append(theta_g_Best)
        
            self.theta_i.append(theta_i_Best)
        
        
    def SpaRCe_Evaluate(self,state,y_true):
    
        state_sparse=[]
        y=[]
        Acc=[]    
        error=[]            
        sparsity=[]
        
        loss = nn.BCEWithLogitsLoss()
        
        N_cl=torch.sum(state!=0)
        
        for i in range(self.N_copies):
            
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     
    
            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            sparsity.append(torch.sum(state_sparse[i]!=0)/N_cl)
            
        return y, Acc, error, state_sparse
    
    
    
    def Online_Evaluate(self,state,y_true):
    
        y=[]
        Acc=[]    
        error=[]            
        
        loss = nn.BCEWithLogitsLoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            
        return y, Acc, error
            
    
    
    
    def Prune(self,  X_tr, Y_tr, X_te, Y_te, SpaRCe_True, N_cuts):
        
        
        
        images=torch.clone(X_tr[:,:])
        labels=torch.clone(Y_tr[:,:])
        
        
        if SpaRCe_True:
            
            Out_tr, Acc_tr, Error_tr, S=self.SpaRCe_Evaluate(images,labels)
            
            
            Active=torch.mean((S[0]!=0).float(),0)

            th=np.linspace(0.0,1,201)
            N_cuts=np.zeros(np.shape(th)[0])
            
            for i in range(np.shape(th)[0]):
                
                N_cuts[i]=torch.sum((Active>th[i])==0)
                
                
        else:
            
            Out_tr, Acc_tr, Error_tr=self.Online_Evaluate(images,labels)
            
            print('Provide the number of nodes to be deleted')
            print(N_cuts)
            
        
        Out_Sp=[]
        Acc_Sp=[]
        Err_Sp=[]
        
        Out_Rand=[]
        Acc_Rand=[]
        Err_Rand=[]
        
        Out_W=[]
        Acc_W=[]
        Err_W=[]
                
        
        for i in range(np.shape(N_cuts)[0]):
            
            
            images=torch.clone(X_te[:,:])
            labels=torch.clone(Y_te[:,:])
            
            N_cut=np.copy(N_cuts[i])
            
            
            if SpaRCe_True:
            
                Mask=Active>th[i]
                Mask=torch.unsqueeze(Mask,1).repeat(1,self.N_class).float()
                self.Ws[0]=torch.clone(nn.Parameter(self.W_Best*Mask))
            
                
                Out_te_Sp, Acc_te_Sp, Error_te_Sp, _=self.SpaRCe_Evaluate(images,labels)
                
            
                Out_Sp.append(Out_te_Sp[0].detach())
                Acc_Sp.append(Acc_te_Sp[0].detach())
                Err_Sp.append(Error_te_Sp[0].detach())
            
            
            for j in range(self.N_copies):
                
                Mask=torch.randint(0,self.N,[int(N_cut)])
                self.Ws[j]=torch.clone(nn.Parameter(self.W_Best))
                self.Ws[j][Mask,:]=0
            
            
            if SpaRCe_True:
            
                Out_te_Rand, Acc_te_Rand, Error_te_Rand, _=self.SpaRCe_Evaluate(images,labels)
            
            else:
                
                Out_te_Rand, Acc_te_Rand, Error_te_Rand=self.Online_Evaluate(images,labels)
                
            
            Out_Rand.append(Out_te_Rand)
            Acc_Rand.append(Acc_te_Rand)
            Err_Rand.append(Error_te_Rand)
            
            
            Fisher=torch.matmul((1-torch.sigmoid(torch.transpose(Out_tr[0],0,1))),X_tr)**2
            sort, indexes=torch.sort(torch.mean(Fisher,0))
            
                        
            Mask=indexes[0:int(N_cut)]
            self.Ws[0]=torch.clone(nn.Parameter(self.W_Best))
            self.Ws[0][Mask,:]=0
            
            if SpaRCe_True:
            
                Out_te_W, Acc_te_W, Error_te_W, _=self.SpaRCe_Evaluate(images,labels)
                
            else:
            
                Out_te_W, Acc_te_W, Error_te_W=self.Online_Evaluate(images,labels)
            
            
            Out_W.append(Out_te_W[0].detach())
            Acc_W.append(Acc_te_W[0].detach())
            Err_W.append(Error_te_W[0].detach())
            
            
        if SpaRCe_True:
            
            return Out_Sp, Acc_Sp, Err_Sp, Out_Rand, Acc_Rand, Err_Rand, Out_W, Acc_W, Err_W, N_cuts
        
        
        else:
        
            return Out_Rand, Acc_Rand, Err_Rand, Out_W, Acc_W, Err_W
        
        
        
        

        
        
        
    
    
    
    
    
    
                   
    