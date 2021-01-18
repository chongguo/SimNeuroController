import sklearn 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm
import numpy as np
import random
from sklearn.linear_model import LinearRegression, BayesianRidge, OrthogonalMatchingPursuit
from sklearn.svm import LinearSVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import explained_variance_score as EV
from scipy.stats import pearsonr as PEAR
from synthesis.synthesizer import NeuralController
from util.misc_functions import indexfun, recreate_image, save_image
import matplotlib.pyplot as plt
import pickle

from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.regression.tucker_regression import TuckerRegressor
import tensorly as tl

# use pytorch backend for fast regression
tl.set_backend('pytorch')

class EnsembleController():
    def __init__(self,nets,selected_layers,labels,device):
        # nets is a list of pytorch networks 
        # selected_layers is the respective layers selected
        # labels are the names for the respective networks
        self.models = [dict.fromkeys(['id','net','layer','dim','activations']) for i in range(len(nets))]
        self.ensemble = {'predictions': None, 'train_score': None, 'test_score': None}
        self.device = device
        self.n_models = len(self.models)
        for n in range(self.n_models):
            self.models[n]['id'] = labels[n]
            self.models[n]['net'] = nn.Sequential(*list(nets[n].children())[:selected_layers[n]])
            self.models[n]['net'].eval()
            self.models[n]['layer'] = selected_layers[n]
    
    def design(self, dataset, batch_size = 10):
        # image used for control inherit size of images used for fitting
        self.img_dim = dataset.dim
        
        # create dataloader
        self.n_data = len(dataset)
        self.batch_size = batch_size
        self.batchloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                        shuffle=False, num_workers=0)
        
        # get the output dimensions and initialized activation tensor
        with torch.no_grad():
            for n in range(self.n_models):
                # send model to cpu
                self.models[n]['net'].cpu()
                f_out = self.models[n]['net'].forward(torch.unsqueeze(dataset[0],0))
                f_out = f_out.detach()[0]
                self.models[n]['dim'] = f_out.shape
                self.models[n]['activations'] = torch.zeros((self.n_data,*f_out.shape),requires_grad = False)
        
        # generate a target indexer
        self.indexer = indexfun(self.models[-1]['dim'])
        
        # collect activation tensors from the given dataset
        print('Extracting activations')
        for n in tqdm(range(self.n_models)):
            # send model to gpu
            self.models[n]['net'].to(self.device)
            with torch.no_grad():
                for [i, x_batch] in enumerate(self.batchloader):
                    batch_idx = list(range(i*self.batch_size,(i+1)*self.batch_size))
                    with torch.no_grad():
                        x_batch = x_batch.to(self.device)
                        f_out = self.models[n]['net'].forward(x_batch)
                        f_out = f_out.detach().cpu()
                        # extract activations    
                        self.models[n]['activations'][batch_idx] = f_out
            # send model back to cpu
            self.models[n]['net'].cpu()
        
    def train_test_subset(self, n_train, n_test, n_targets, target_net = -1):
        self.n_train = n_train
        self.n_test = n_test
        self.n_targets = n_targets
        self.target_net = target_net
        self.train_idx = list(range(n_train))
        self.test_idx = list(range(n_train,n_train+n_test))
        # select a random number of units with lifetime sparsity >.1
        rand_idx = np.asarray(random.sample(range(0,len(self.indexer)),len(self.indexer)))
        tar_acts = self.models[self.target_net]['activations'][self.indexer.get_tar(rand_idx,self.test_idx)].numpy()
        tar_sparsity = np.mean(tar_acts>1e-5,axis = 0)
        valididx = rand_idx[tar_sparsity>.1]
        self.target_idx = valididx[:n_targets]
        for n in range(self.n_models):
            self.models[n]['predictions'] = torch.zeros((self.n_data,self.n_targets),requires_grad = False)
            self.models[n]['train_score'] = np.zeros(self.n_targets)
            self.models[n]['test_score'] = np.zeros(self.n_targets)
            
    def fit(self, option = 'TR', rank = 1, load = False, label = '', w_reg = 1, n_iter = 20):
        # with the following options:
        # TR - low-rank tensor regression
        # LR - linear regression
        # OMP - orthogonal matching pursuit
        # rank - rank of weight matrix along each dimension
        self.reg_opt = option
        
        if load:
            # import learned weights
            paramdict = torch.load('params\\EnsembleParams_'+option+label+'.pt')
            self.tensor_w = [w.to(self.device) for i,w in enumerate(paramdict['weights'])]
            self.tensor_b = [b.to(self.device) for i,b in enumerate(paramdict['bias'])]
            self.target_idx = paramdict['targets']
            self.n_targets = len(self.target_idx)
        else:    
            self.tensor_w = [torch.zeros(*self.models[n]['dim'],self.n_targets, device = self.device, requires_grad = False) for n in range(self.n_models)]
            self.tensor_b = [torch.zeros(self.n_targets, device = self.device, requires_grad = False) for n in range(self.n_models)]
            # extract activation from the target
            Y_train = self.models[-1]['activations'][self.indexer.get_tar(self.target_idx,self.train_idx)].detach()
            # contruct the estimator
            for n in range(self.n_models):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if option == 'TR':
                        estimator = TuckerRegressor(weight_ranks=[rank,rank,rank], tol=1e-3, n_iter_max=n_iter, reg_W=w_reg, verbose=0)
                        X_train = self.models[n]['activations'][self.train_idx].clone().to(self.device)
                        #X_norm = torch.maximum(torch.linalg.norm(X_train,ord=1,dim=0),torch.tensor(1e-3,device = self.device))
                        #X_train = f.normalize(X_train,p=1,dim=0,eps = 1e-3)
                        y_train = Y_train.clone().to(self.device)
                        #y_norm = torch.maximum(torch.linalg.norm(y_train,ord=1,dim=0),torch.tensor(1e-3,device = self.device))
                        #y_train = f.normalize(y_train,p=1,dim=0,eps = 1e-3)
                    elif option == 'OMP':
                        estimator = OrthogonalMatchingPursuit(n_nonzero_coefs = 2)
                        X_train = self.models[n]['activations'][self.train_idx].numpy().reshape(self.n_train,-1)
                        y_train = Y_train.numpy()
                    elif option == 'LR':
                        tar_ids = self.indexer.get_tar(self.target_idx)
                        estimator = LinearRegression()
                        X_train = np.transpose(self.models[n]['activations'][self.train_idx].numpy(),(1,2,3,0)).reshape((self.models[n]['dim'][0],-1)).T
                        yt1 = self.models[-1]['activations'][self.train_idx].numpy()
                        y_train = np.transpose(yt1[:,tar_ids[0]],(1,2,3,0)).reshape((self.n_targets,-1)).T
                    else:
                        print(option+' not implemented!')
                    # Fit the source to target
                    print('Mapping '+self.models[n]['id']+' --> '+self.models[-1]['id'])
                    for i, unit_id in enumerate(tqdm(self.target_idx)):
                        estimator.fit(X_train,y_train[:,i])
                        if option == 'TR':
                            self.tensor_w[n][:,:,:,i] = estimator.weight_tensor_.cpu().detach()
                        elif option == 'LR':
                            tar_id = self.indexer.get_tar(unit_id)
                            self.tensor_w[n][:,tar_id[1],tar_id[2],i] = torch.tensor(estimator.coef_)
                            self.tensor_b[n][i] = torch.tensor(estimator.intercept_,device = self.device)
                        elif option=='OMP':
                            self.tensor_w[n][:,:,:,i] = torch.tensor(estimator.coef_.reshape(*self.models[n]['dim']),device = self.device)
                    del X_train, estimator
            # save the learned weights
            paramdict = {'weights': self.tensor_w, 'bias': self.tensor_b, 'targets': self.target_idx}
            torch.save(paramdict,'params\\EnsembleParams_'+option+label+'.pt')
    
    def score_models(self,metric = 'EV'):
        # generate the predcitions and model scores
        Y_train = self.models[-1]['activations'][self.indexer.get_tar(self.target_idx,self.train_idx)].detach()
        Y_test = self.models[-1]['activations'][self.indexer.get_tar(self.target_idx,self.test_idx)].detach()
        for n in range(self.n_models):
            # extract activation from the source
            X_train = self.models[n]['activations'][self.train_idx].detach().to(self.device)
            X_test = self.models[n]['activations'][self.test_idx].detach().to(self.device)
            for i, unit_id in enumerate(self.target_idx):
                # predict and score
                Y_train_pred = (torch.einsum('nijk,ijk->n',X_train,self.tensor_w[n][:,:,:,i])+self.tensor_b[n][i]).cpu().detach()
                Y_test_pred = (torch.einsum('nijk,ijk->n',X_test,self.tensor_w[n][:,:,:,i])+self.tensor_b[n][i]).cpu().detach()
                self.models[n]['predictions'][self.train_idx,i] = Y_train_pred
                self.models[n]['predictions'][self.test_idx,i] = Y_test_pred
                self.models[n]['train_score'][i] = EV(Y_train[:,i],Y_train_pred)
                self.models[n]['test_score'][i] = EV(Y_test[:,i],Y_test_pred)
            del X_train, X_test, Y_train_pred, Y_test_pred
                    
    def score_ensembles(self, metric = 'EV'):
        Y_train = self.models[-1]['activations'][self.indexer.get_tar(self.target_idx,self.train_idx)].detach()
        Y_test = self.models[-1]['activations'][self.indexer.get_tar(self.target_idx,self.test_idx)].detach()
        self.ensemble['predictions'] = torch.zeros((self.n_data,self.n_targets,self.n_models-1),requires_grad = False)
        self.ensemble['train_score'] = np.zeros((self.n_targets,self.n_models-1,self.n_models-1))
        self.ensemble['test_score'] = np.zeros((self.n_targets,self.n_models-1,self.n_models-1))
        for N in range(self.n_models-1):
            for r in range(self.n_models-1):
                ns = [(ni+r)%(self.n_models-1) for ni in range(N+1)]
                for i in range(self.n_targets):
                    self.ensemble['predictions'][:,i,N] = 0
                    for k, n in enumerate(ns):
                        # simple averaging ensemble
                        self.ensemble['predictions'][:,i,N] += self.models[n]['predictions'][:,i]/(N+1)
                    self.ensemble['train_score'][i,r,N] = EV(Y_train[:,i],self.ensemble['predictions'][self.train_idx,i,N])
                    self.ensemble['test_score'][i,r,N] = EV(Y_test[:,i],self.ensemble['predictions'][self.test_idx,i,N])
        
    def get_surrogate(self):
        # create  to a fc layer
        for n in range(self.n_models):
            self.models[n]['surrogate'] = nn.Sequential(
                            *list(self.models[n]['net'].children()),
                            nn.Flatten(),
                            nn.Linear(self.models[n]['dim'].numel(),self.n_targets)
                        )
            # populate with learned weights
            with torch.no_grad():
                self.models[n]['surrogate'][-1].weight.shape
                self.models[n]['surrogate'][-1].weight.copy_(torch.nn.Parameter(self.tensor_w[n].reshape(self.models[n]['dim'].numel(),self.n_targets).T))
                self.models[n]['surrogate'][-1].bias.copy_(torch.nn.Parameter(self.tensor_b[n]))
    
    def design_scalar_clamp(self, clampdataset, epsilon = 10, n_units = 10, batch_size = 5):
        self.n_units = n_units
        self.epsilon = epsilon
        self.clamp_batch_size = batch_size
        self.clamp_target_idx = self.target_idx[:n_units]
        self.n_clamp_img = len(clampdataset)
        self.clamploader = torch.utils.data.DataLoader(clampdataset, batch_size=self.clamp_batch_size, shuffle=False, num_workers=0)
        for n in range(self.n_models-1):
            # target epsilon on surrogate units
            self.models[n]['sur_s_eps'] = 2*epsilon*(torch.rand((self.n_clamp_img,self.n_units))>0.5).float().to(self.device)-epsilon
            self.models[n]['sur_s_eps'].requires_grad = False
            # observed epsilon on surrogate and target units
            self.models[n]['sur_s_obs'] = torch.zeros((self.n_clamp_img,self.n_units),device = self.device, requires_grad = False)
            self.models[n]['tar_s_obs'] = torch.zeros((self.n_clamp_img,self.n_units),device = self.device, requires_grad = False)
            self.models[n]['s_dx'] = torch.zeros((self.n_clamp_img,self.n_units), requires_grad = False)
    
        # target epsilon on surrogate units
        self.ensemble['sur_s_eps'] = 2*epsilon*(torch.rand((self.n_clamp_img,self.n_units))>0.5).float().to(self.device)-epsilon
        self.ensemble['sur_s_eps'].requires_grad = False
        # observed epsilon on surrogate and target units
        self.ensemble['sur_s_obs'] = torch.zeros((self.n_clamp_img,self.n_units),device = self.device, requires_grad = False)
        self.ensemble['tar_s_obs'] = torch.zeros((self.n_clamp_img,self.n_units),device = self.device, requires_grad = False)
        self.ensemble['s_dx'] = torch.zeros((self.n_clamp_img,self.n_units), requires_grad = False)
        
    def scalar_model_clamp(self, lr = .005, wd = 1e-3, n_iter = 20):
        # control with surrogate net and measure on target net
        self.n_iter = n_iter
        self.lr = lr
        self.wd = wd
        # loss for the clamp target 
        clamp_loss = torch.nn.MSELoss()
        # A-clamp with each model
        print('Optimizing surrogate unit scalar control:')
        self.models[-1]['net'].to(self.device)
        for n in range(self.n_models-1):
            print('Scalar A-Clamping '+self.models[n]['id']+' --> '+self.models[-1]['id'])
            self.models[n]['surrogate'].to(self.device)
            for [i, x1_batch] in enumerate(tqdm(self.clamploader)):
                # pass current batch to gpu
                x0_batch = x1_batch.detach().clone().to(self.device)
                batch_idx = list(range(i*self.clamp_batch_size,(i+1)*self.clamp_batch_size))
                # get intial target activations
                with torch.no_grad():
                    f0_tar = self.models[-1]['net'].forward(x0_batch).detach()[self.indexer.get_tar(self.clamp_target_idx,list(range(self.clamp_batch_size)))].detach()
                    f0_sur = self.models[n]['surrogate'].forward(x0_batch).detach()[:,:self.n_units]
                    f1_sur_opt = f0_sur+self.models[n]['sur_s_eps'][batch_idx,:]
                for u, unit_id in enumerate(self.clamp_target_idx):
                    # reset image
                    x0_batch = x1_batch.detach().clone().to(self.device)
                    x0_batch.requires_grad = True
                    optimizer = torch.optim.Adam([x0_batch],lr=self.lr,weight_decay=self.wd)
                    # iteratively find images that steer the surrogate unit epsilon distance up or down
                    for j in range(n_iter):
                        optimizer.zero_grad()
                        f1_sur = self.models[n]['surrogate'].forward(x0_batch)
                        # compute clamp loss
                        losses = clamp_loss(f1_sur[:,u],f1_sur_opt[:,u])
                        losses.backward(retain_graph=True)
                        optimizer.step()
                        torch.clamp(x0_batch,min=-2.25,max=2.75)
                    # pass the batch of optimized clamp image to target network
                    with torch.no_grad():
                        # get magnitude of image delta
                        dx = x0_batch.detach().cpu()-x1_batch.detach()
                        self.models[n]['s_dx'][batch_idx,u] = torch.sum(dx.reshape(self.clamp_batch_size,-1)**2,axis=1)**(1/2)
                        # pass the batch of optimized clamp image to target network
                        f1_tar = self.models[-1]['net'].forward(x0_batch)
                        # record the epsilons on the target networks
                        tid = self.indexer.get_tar([unit_id],list(range(self.clamp_batch_size)));
                        self.models[n]['sur_s_obs'][batch_idx,u] = f1_sur[:,u].detach() - f0_sur[:,u]
                        self.models[n]['tar_s_obs'][batch_idx,u] = f1_tar[tid].detach().squeeze() - f0_tar[:,u]
            del f0_tar, f1_tar, f0_sur, f1_sur, f1_sur_opt, x1_batch
            # move things out of gpu 
            self.models[n]['surrogate'].cpu()
            self.models[n]['sur_s_eps'] = self.models[n]['sur_s_eps'].cpu()
            self.models[n]['sur_s_obs'] = self.models[n]['sur_s_obs'].cpu()
            self.models[n]['tar_s_obs'] = self.models[n]['tar_s_obs'].cpu()
            self.models[n]['s_dx'] = self.models[n]['s_dx'].cpu()
        self.models[-1]['net'].cpu()
    
    def scalar_ensemble_clamp(self, lr = .005, wd = 1e-3, n_iter = 20):
        # control with surrogate net and measure on target net
        self.n_iter = n_iter
        self.lr = lr
        self.wd = wd
        # loss for the clamp target 
        clamp_loss = torch.nn.MSELoss()
        
        # A-clamp with ensemble
        for n in range(self.n_models-1):
            self.models[n]['surrogate'].to(self.device)
        self.models[-1]['net'].to(self.device) 
        print(f'Scalar A-Clamping with ensemble(x{self.n_models-1})')
        for [i, x1_batch] in enumerate(tqdm(self.clamploader)):
            # pass current batch to gpu
            x0_batch = x1_batch.detach().clone().to(self.device)
            batch_idx = list(range(i*self.clamp_batch_size,(i+1)*self.clamp_batch_size))
            # get intial activations
            with torch.no_grad():
                f0_tar = self.models[-1]['net'].forward(x0_batch).detach()[self.indexer.get_tar(self.clamp_target_idx,list(range(self.clamp_batch_size)))].detach()
                f0_sur = torch.zeros((self.clamp_batch_size,self.n_units),device = self.device)
                for n in range(self.n_models-1):
                    f0_sur += self.models[n]['surrogate'].forward(x0_batch)[:,:self.n_units]/(self.n_models-1)
                f1_sur_opt = f0_sur+self.ensemble['sur_s_eps'][batch_idx,:]
            for u, unit_id in enumerate(self.clamp_target_idx):
                # reset image
                x0_batch = x1_batch.detach().clone().to(self.device)
                x0_batch.requires_grad = True
                optimizer = torch.optim.Adam([x0_batch],lr=self.lr,weight_decay=self.wd)
                # iteratively find images that steer the surrogate unit epsilon distance up or down
                for j in range(n_iter):
                    optimizer.zero_grad()
                    f1_sur = torch.zeros((self.clamp_batch_size,self.n_units),device = self.device)
                    for n in range(self.n_models-1):
                        f1_sur += self.models[n]['surrogate'].forward(x0_batch)[:,:self.n_units]/(self.n_models-1)
                    # compute clamp loss
                    losses = clamp_loss(f1_sur[:,u],f1_sur_opt[:,u])
                    losses.backward(retain_graph=True)
                    optimizer.step()
                    torch.clamp(x0_batch,min=-2.25,max=2.75)
                with torch.no_grad():
                    # get magnitude of image delta
                    dx = x0_batch.detach().cpu()-x1_batch.detach()
                    self.ensemble['s_dx'][batch_idx,u] = torch.sum(dx.reshape(self.clamp_batch_size,-1)**2,axis=1)**(1/2)
                    # pass the batch of optimized clamp image to target network
                    f1_tar = self.models[-1]['net'].forward(x0_batch)
                    # record the epsilons on the target networks
                    tid = self.indexer.get_tar([unit_id],list(range(self.clamp_batch_size)));
                    self.ensemble['sur_s_obs'][batch_idx,u] = f1_sur[:,u].detach() - f0_sur[:,u]
                    self.ensemble['tar_s_obs'][batch_idx,u] = f1_tar[tid].detach().squeeze() - f0_tar[:,u]
        del f0_tar, f1_tar, f0_sur, f1_sur, f1_sur_opt, x1_batch
        # send things back to cpu
        for n in range(self.n_models-1):
            self.models[n]['surrogate'].cpu()
        self.models[-1]['net'].cpu()
        self.ensemble['sur_s_eps'] = self.ensemble['sur_s_eps'].cpu()
        self.ensemble['sur_s_obs'] = self.ensemble['sur_s_obs'].cpu()
        self.ensemble['tar_s_obs'] = self.ensemble['tar_s_obs'].cpu()
        self.ensemble['s_dx'] = self.ensemble['s_dx'].cpu()
    
    def scalar_clamp_score(self):
        # model scores
        for n in range(self.n_models-1):
            self.models[n]['opt_s_score'] = np.empty(self.n_units)
            self.models[n]['ctr_s_score'] = np.empty(self.n_units)
            for u, unit_id in enumerate(self.clamp_target_idx):
                self.models[n]['opt_s_score'][u] = EV(self.models[n]['sur_s_eps'][:,u].numpy(),self.models[n]['sur_s_obs'][:,u].numpy())
                self.models[n]['ctr_s_score'][u] = EV(self.models[n]['sur_s_obs'][:,u].numpy(),self.models[n]['tar_s_obs'][:,u].numpy())
        # ensemble scores
        self.ensemble['opt_s_score'] = np.empty(self.n_units)
        self.ensemble['ctr_s_score'] = np.empty(self.n_units)
        for u, unit_id in enumerate(self.clamp_target_idx):
            self.ensemble['opt_s_score'][u] = EV(self.ensemble['sur_s_eps'][:,u].numpy(),self.ensemble['sur_s_obs'][:,u].numpy())
            self.ensemble['ctr_s_score'][u] = EV(self.ensemble['sur_s_obs'][:,u].numpy(),self.ensemble['tar_s_obs'][:,u].numpy())

    def design_proj_clamp(self, clampdataset, epsilon = 10, n_sets = 10, set_size = 50, batch_size = 5):
        self.n_sets = n_sets
        self.set_size = set_size
        self.epsilon = epsilon
        self.clamp_set_i = np.zeros((self.set_size,self.n_sets)).astype(int)
        self.clamp_set_idx = np.zeros((self.set_size,self.n_sets)).astype(int)
        for s in range(self.n_sets): 
            self.clamp_set_i[:,s] = np.random.choice(self.n_targets,size = self.set_size)
            self.clamp_set_idx[:,s] = self.target_idx[self.clamp_set_i[:,s]].squeeze()
        self.n_clamp_img = len(clampdataset)
        self.clamp_batch_size = batch_size
        self.clamploader = torch.utils.data.DataLoader(clampdataset, batch_size=self.clamp_batch_size, shuffle=False, num_workers=0)
        
        # target epsilon on surrogate units
        self.ensemble['sur_p_eps'] = 2*epsilon*(torch.rand((self.n_clamp_img,self.n_sets))>0.5).float()-epsilon
        self.ensemble['sur_p_eps'].requires_grad = False
        # unit-norm projection vector
        self.ensemble['sur_p_vec'] = torch.randn((self.set_size,self.n_sets))
        self.ensemble['sur_p_vec'].requires_grad = False
        for s in range(self.n_sets):
            self.ensemble['sur_p_vec'][:,s] = self.ensemble['sur_p_vec'][:,s]/torch.linalg.norm(self.ensemble['sur_p_vec'][:,s])
        # observed epsilon on surrogate and target units
        self.ensemble['sur_p_obs'] = torch.zeros((self.n_clamp_img,self.n_sets), requires_grad = False)
        self.ensemble['tar_p_obs'] = torch.zeros((self.n_clamp_img,self.n_sets), requires_grad = False)
        self.ensemble['p_dx'] = torch.zeros((self.n_clamp_img,self.n_sets), requires_grad = False)

        for n in range(self.n_models):
            # target epsilon on surrogate units
            self.models[n]['sur_p_eps'] = self.ensemble['sur_p_eps'].clone()
            self.models[n]['sur_p_eps'].requires_grad = False
            # unit-norm projection vector
            self.models[n]['sur_p_vec'] = self.ensemble['sur_p_vec'].clone()
            self.models[n]['sur_p_vec'].requires_grad = False
            #for s in range(self.n_sets):
            #    self.models[n]['sur_p_vec'][:,s] = self.models[n]['sur_p_vec'][:,s]/torch.linalg.norm(self.models[n]['sur_p_vec'][:,s])
            # observed epsilon on surrogate and target units
            self.models[n]['sur_p_obs'] = torch.zeros((self.n_clamp_img,self.n_sets), requires_grad = False)
            self.models[n]['tar_p_obs'] = torch.zeros((self.n_clamp_img,self.n_sets), requires_grad = False)
            self.models[n]['p_dx'] = torch.zeros((self.n_clamp_img,self.n_sets), requires_grad = False)
    
    def proj_model_clamp(self, lr = .005, wd = 1e-3, n_iter = 20):
        # control with surrogate net and measure on target net
        self.n_iter = n_iter
        self.lr = lr
        self.wd = wd
        # loss for the clamp target 
        clamp_loss = torch.nn.MSELoss()
        
        # A-clamp with ensemble
        self.models[-1]['net'].to(self.device)
        for n in range(self.n_models):
            print('Projected A-Clamping '+self.models[n]['id']+' --> '+self.models[-1]['id'])
            self.models[n]['surrogate'].to(self.device)
            for [i, x1_batch] in enumerate(tqdm(self.clamploader)):
                # pass current batch to gpu
                x0_batch = x1_batch.detach().clone().to(self.device)
                batch_idx = list(range(i*self.clamp_batch_size,(i+1)*self.clamp_batch_size))
                # get intial activations
                for s in range(self.n_sets):
                    u = self.clamp_set_i[:,s].tolist()
                    unit_id = self.clamp_set_idx[:,s].tolist()
                    tar_id = self.indexer.get_tar(unit_id,list(range(self.clamp_batch_size)))
                    # reset image
                    x0_batch = x1_batch.detach().clone().to(self.device)
                    with torch.no_grad():
                    # original activations
                        f0_tar = self.models[-1]['net'].forward(x0_batch).detach()[tar_id]
                        f0_sur = self.models[n]['surrogate'].forward(x0_batch).detach()[:,u]
                        # projections
                        proj_vec = self.models[n]['sur_p_vec'][:,s].to(self.device)
                        f0_tar_proj = torch.matmul(f0_tar,proj_vec).squeeze()
                        f0_tar_proj.requires_grad = False
                        f0_sur_proj = torch.matmul(f0_sur,proj_vec).squeeze()
                        f0_sur_proj.requires_grad = False
                        f1_sur_proj_opt = f0_sur_proj+self.models[n]['sur_p_eps'][batch_idx,s].to(self.device)
                        f1_sur_proj_opt.requires_grad = False
                    x0_batch.requires_grad = True
                    optimizer = torch.optim.Adam([x0_batch],lr=self.lr,weight_decay=self.wd)
                    # iteratively find images that steer the surrogate unit epsilon distance up or down
                    for j in range(n_iter):
                        optimizer.zero_grad()
                        f1_sur = self.models[n]['surrogate'].forward(x0_batch)[:,u]
                        f1_sur_proj = torch.matmul(f1_sur,proj_vec).squeeze()
                        # compute clamp loss
                        losses = clamp_loss(f1_sur_proj,f1_sur_proj_opt)
                        losses.backward(retain_graph=True)
                        optimizer.step()
                        torch.clamp(x0_batch,min=-2.25,max=2.75)
                    with torch.no_grad():
                        # get magnitude of image delta
                        dx = x0_batch.detach().cpu()-x1_batch.detach()
                        self.models[n]['p_dx'][batch_idx,s] = torch.sum(dx.reshape(self.clamp_batch_size,-1)**2,axis=1)**(1/2)
                        # pass the batch of optimized clamp image to target network
                        f1_tar = self.models[-1]['net'].forward(x0_batch).detach()[tar_id]
                        f1_tar_proj = torch.matmul(f1_tar,proj_vec).squeeze()
                        # record the epsilons on the target networks
                        self.models[n]['sur_p_obs'][batch_idx,s] = (f1_sur_proj - f0_sur_proj).cpu()
                        self.models[n]['tar_p_obs'][batch_idx,s] = (f1_tar_proj - f0_tar_proj).cpu()
            del f0_tar, f1_tar, f0_sur, f1_sur, f1_sur_proj_opt, proj_vec, f0_sur_proj, f1_tar_proj, x1_batch, x0_batch, dx, losses, optimizer
            self.models[n]['surrogate'].cpu()
        self.models[-1]['net'].cpu()

    def proj_ensemble_clamp(self, clampdataset, batch_size = 2, lr = .005, wd = 1e-3, n_iter = 20):
        # control with surrogate net and measure on target net
        self.n_iter = n_iter
        self.lr = lr
        self.wd = wd
        # loss for the clamp target 
        clamp_loss = torch.nn.MSELoss()
        # smaller batch
        self.clamp_batch_size = batch_size
        self.clamploader = torch.utils.data.DataLoader(clampdataset, batch_size=self.clamp_batch_size, shuffle=False, num_workers=0)
        
        
        # A-clamp with ensemble
        print(f'Projected A-Clamping with ensemble(x{self.n_models-1})')
        for n in range(self.n_models-1):
            self.models[n]['surrogate'].to(self.device)
        self.models[-1]['net'].to(self.device)
        for [i, x1_batch] in enumerate(tqdm(self.clamploader)):
            # pass current batch to gpu
            x0_batch = x1_batch.detach().clone().to(self.device)
            batch_idx = list(range(i*self.clamp_batch_size,(i+1)*self.clamp_batch_size))
            # get intial activations
            for s in range(self.n_sets):
                u = self.clamp_set_i[:,s].tolist()
                unit_id = self.clamp_set_idx[:,s].tolist()
                tar_id = self.indexer.get_tar(unit_id,list(range(self.clamp_batch_size)))
                # reset image
                x0_batch = x1_batch.detach().clone().to(self.device)
                with torch.no_grad():
                # original activations
                    f0_tar = self.models[-1]['net'].forward(x0_batch).detach()[tar_id]
                    f0_sur = torch.zeros((self.clamp_batch_size,self.set_size),requires_grad = False).to(self.device)
                    for n in range(self.n_models-1):
                        f0_sur += self.models[n]['surrogate'].forward(x0_batch).detach()[:,u]/(self.n_models-1)
                    # projections
                    proj_vec = self.ensemble['sur_p_vec'][:,s].to(self.device)
                    f0_tar_proj = torch.matmul(f0_tar,proj_vec).squeeze()
                    f0_tar_proj.requires_grad = False
                    f0_sur_proj = torch.matmul(f0_sur,proj_vec).squeeze()
                    f0_sur_proj.requires_grad = False
                    f1_sur_proj_opt = f0_sur_proj+self.ensemble['sur_p_eps'][batch_idx,s].to(self.device)
                    f1_sur_proj_opt.requires_grad = False
                x0_batch.requires_grad = True
                optimizer = torch.optim.Adam([x0_batch],lr=self.lr,weight_decay=self.wd)
                # iteratively find images that steer the surrogate unit epsilon distance up or down
                for j in range(n_iter):
                    optimizer.zero_grad()
                    f1_sur = torch.zeros((self.clamp_batch_size,self.set_size)).to(self.device)
                    for n in range(self.n_models-1):
                        f1_sur += self.models[n]['surrogate'].forward(x0_batch)[:,u]/(self.n_models-1)
                    f1_sur_proj = torch.matmul(f1_sur,proj_vec).squeeze()
                    # compute clamp loss
                    losses = clamp_loss(f1_sur_proj,f1_sur_proj_opt)
                    losses.backward(retain_graph=True)
                    optimizer.step()
                    torch.clamp(x0_batch,min=-2.25,max=2.75)
                with torch.no_grad():
                    # get magnitude of image delta
                    dx = x0_batch.detach().cpu()-x1_batch.detach()
                    self.ensemble['p_dx'][batch_idx,s] = torch.sum(dx.reshape(self.clamp_batch_size,-1)**2,axis=1)**(1/2)
                    # pass the batch of optimized clamp image to target network
                    f1_tar = self.models[-1]['net'].forward(x0_batch).detach()[tar_id]
                    f1_tar_proj = torch.matmul(f1_tar,proj_vec).squeeze()
                    # record the epsilons on the target networks
                    self.ensemble['sur_p_obs'][batch_idx,s] = (f1_sur_proj - f0_sur_proj).cpu()
                    self.ensemble['tar_p_obs'][batch_idx,s] = (f1_tar_proj - f0_tar_proj).cpu()
            del f0_tar, f1_tar, f0_sur, f1_sur, f1_sur_proj_opt, proj_vec, f0_sur_proj, f1_tar_proj, x1_batch, x0_batch, dx, losses, optimizer
        # send models back to cpu
        for n in range(self.n_models-1):
            self.models[n]['surrogate'].cpu()
        self.models[-1]['net'].cpu()
        
        
            
    def proj_clamp_score(self, model_eval = True, ensemble_eval = True):
        # model scores
        if model_eval:
            for n in range(self.n_models):
                self.models[n]['opt_p_score'] = np.empty(self.n_sets)
                self.models[n]['ctr_p_score'] = np.empty(self.n_sets)
                for s in range(self.n_sets):
                    self.models[n]['opt_p_score'][s] = PEAR(self.models[n]['sur_p_eps'][:,s].numpy(),self.models[n]['sur_p_obs'][:,s].numpy())[0]
                    self.models[n]['ctr_p_score'][s] = PEAR(self.models[n]['sur_p_obs'][:,s].numpy(),self.models[n]['tar_p_obs'][:,s].numpy())[0]
        
        if ensemble_eval:
            # ensemble scores
            self.ensemble['opt_p_score'] = np.empty(self.n_sets)
            self.ensemble['ctr_p_score'] = np.empty(self.n_sets)
            for s in range(self.n_sets):
                self.ensemble['opt_p_score'][s] = PEAR(self.ensemble['sur_p_eps'][:,s].numpy(),self.ensemble['sur_p_obs'][:,s].numpy())[0]
                self.ensemble['ctr_p_score'][s] = PEAR(self.ensemble['sur_p_obs'][:,s].numpy(),self.ensemble['tar_p_obs'][:,s].numpy())[0]
"""
    def proj_ensemble_clamp(self, epsilon = 10, n_sets = 100, set_size = 10, lr = .001, wd = 1e-5, n_iter = 20, load = None):
        # control with surrogate net and measure on target net
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.lr = lr
        self.wd = wd
        self.n_sets = n_sets
        self.set_size = set_size
        self.set_id = np.empty((set_size,n_sets),dtype=np.int8)
        # generate all the sets from the center 1/2 of the channel output
        for s in range(n_sets): 
            rand_set = random.sample(range(0,len(self.indexer)),len(self.indexer))
            tempidx = np.asarray(self.indexer.get_tar(rand_set),dtype=np.float)
            min1, max1 = self.tar_dim[1]*.25, self.tar_dim[1]*.75
            min2, max2 = self.tar_dim[2]*.25, self.tar_dim[2]*.75
            valididx = np.where((tempidx[1]>min1)&(tempidx[1]<max1)&(tempidx[2]>min2)&(tempidx[2]<max2))[0]
            self.set_id[:,s] = [rand_set[u] for i,u in enumerate(valididx[:set_size])]
        # target epsilon on surrogate units
        self.sur_eps = 2*epsilon*(torch.rand((self.n_train+self.n_test,n_sets),requires_grad=False)>0.5).float().to(self.device)-epsilon
        # observed epsilon on surrogate units
        self.sur_obs = torch.zeros((self.n_train+self.n_test,n_sets)).to(self.device)
        self.sur_obs.requires_grad = False
        # observed epsilon on target units
        self.tar_obs = torch.zeros((self.n_train+self.n_test,n_sets)).to(self.device)
        self.tar_obs.requires_grad = False
        # observed movement on image
        self.dx = np.zeros((self.n_train+self.n_test,n_sets))
        # loss for the clamp target 
        clamp_loss = torch.nn.MSELoss()
        
        print('Optimizing projected A-clamp over surrogate network:')
        for [i, x1_batch] in enumerate(tqdm(self.batchloader)):
            # make a copy of the batch
            x1_batch_copy = x1_batch.detach().clone()
            for s in range(n_sets):
                unit_id = self.set_id[:,s]
                # get indices for subsetting the target and surrogate batch output
                bidx = list(range(i*self.batch_size,(i+1)*self.batch_size))
                tar_bindex = self.indexer.get_tar(unit_id,bidx)
                sur_bindex = self.indexer.get_sur(unit_id,bidx)
                tar_index = self.indexer.get_sur(unit_id,list(range(self.batch_size)))
                sur_index = self.indexer.get_sur(unit_id,list(range(self.batch_size)))
                # get the clean target and surrogate activations over the set
                f0_tar = self.Xtar[tar_bindex].to(self.device)
                f0_tar.requires_grad = False
                f0_sur = self.Xsur[sur_bindex].to(self.device)
                f0_sur.requires_grad = False
                # get the target activation direction and amplitude over the set 
                f0_tar_norm = torch.linalg.norm(f0_tar,dim=1)
                f0_tar_norm.requires_grad = False
                f0_tar_vec = f0_tar/(torch.linalg.norm(f0_tar,dim = 1,keepdim=True)+1e-10)
                f0_tar_vec.requires_grad = False
                # calculate current projection of surrogate activation along target activatin vector
                f0_sur2tar_proj = torch.sum(f0_sur*f0_tar_vec,axis=1)
                # pass current batch to gpu
                x1_batch = x1_batch.to(self.device)
                x1_batch.requires_grad = True
                # optimizer for the batch
                optimizer = torch.optim.Adam([x1_batch],lr=self.lr,weight_decay=self.wd)
                # pick norm aim of current set along the target activation vector
                f1_sur_opt = f0_sur2tar_proj+self.sur_eps[bidx,s]
                # iteratively find images that steer the surrogate unit epsilon distance up or down along the observed surrogate vector
                for n in range(n_iter):
                    optimizer.zero_grad()
                    f1_sur = self.surrogate_net.forward(x1_batch)
                    f1_sur2tar_proj = torch.sum(f1_sur[sur_index]*f0_tar_vec,axis=1)
                    # compute vector clamp loss
                    losses = clamp_loss(f1_sur2tar_proj,f1_sur_opt)
                    losses.backward(retain_graph=True)
                    optimizer.step()
                    torch.clamp(x1_batch,min=-2.25,max=2.75)
                # pass the batch of optimized clamp image to target network
                f1_tar = self.target_net.forward(x1_batch)
                f1_tar_norm = torch.sum(f1_tar[tar_index]*f0_tar_vec,axis=1)
                # record the epsilons on the target networks
                self.tar_obs[bidx,s] = f1_tar_norm.detach().data - f0_tar_norm.detach().data
                self.sur_obs[bidx,s] = f1_sur2tar_proj.detach().data - f0_sur2tar_proj.detach().data 
                # record the delta on the x
                dx = x1_batch.cpu().detach().numpy()-x1_batch_copy.cpu().detach().numpy()
                self.dx[bidx,s] = np.sum(dx.reshape(self.batch_size,-1)**2,axis=1)**(1/2)
                # Save image
                self.created_image = recreate_image(x1_batch.cpu().detach()-x1_batch_copy,dataset = 'hvm')
                #im_path = 'generated\clamp+snet2snet_sid'+ str(s) +'_ssize'+str(set_size)+'_img_'+ str(i*self.batch_size) +'.jpg'
                #save_image(self.created_image, im_path)
                # reset the image
                x1_batch = x1_batch_copy.detach().clone()


    def grad_design(self,dataset, batch_size = 1, record_gradient = False):
        # image used for control is the same as the size of image used for fitting
        self.img_dim = dataset.dim
        # design extract activation (and gradient) for the entire dataset
        self.ndata = len(dataset)
        self.batch_size = batch_size
        self.batchloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                        shuffle=True, num_workers=0)
        f_samp = self.target_net.forward(iter(self.batchloader).next())
        # generate a target indexer
        self.act_dim = f_samp.shape[1:]
        self.target_indexer = indexfun(self.act_dim,source.target_unit).subset((1,self.f_subsamp,self.f_subsamp))
        # create empty list for holding data
        Xsor = []
        Xtar = []
        # create empty list for holding gradients wrt input
        if record_gradient:
            self.grad = True
            Xsor_grad = []
            Xtar_grad = []
        # assemble data matix of activation vector for the given dataset, batch for speed
        if record_gradient:
            print('Extracting activations and input jacobians')
        else:
            print('Extracting activations')
        for [i, x_batch] in enumerate(tqdm(self.batchloader)):
            # track gradient on the input
            if record_gradient:
                x_batch.requires_grad=True
            # run the two models
            f_sor = self.source_net.forward(x_batch)
            f_tar = self.target_net.forward(x_batch)
            
            for j in range(self.batch_size):
                # extract the gradient wrt each image for each unit
                if record_gradient:
                    f_sor_temp = []
                    f_tar_temp = []
                    for [u, unit_id] in enumerate(self.source_unit):
                        # compute source gradient
                        fs_act = f_sor[j,unit_id,::self.f_subsamp,::self.f_subsamp]
                        fs_act.backward(retain_graph = True)
                        f_sor_grad = x_batch.grad[j,:,::self.img_subsamp,::self.img_subsamp].cpu().data.numpy().reshape((1,-1))
                        x_batch.grad[j].data = torch.zeros_like(x_batch.grad[j].data).to(self.device)
                        # compute target gradient
                        ft_act = f_tar[j,unit_id,::self.f_subsamp,::self.f_subsamp].mean()
                        ft_act.backward(retain_graph = True)
                        f_tar_grad = x_batch.grad[j,:,::self.img_subsamp,::self.img_subsamp].cpu().data.numpy().reshape((1,-1))
                        x_batch.grad[j].data = torch.zeros_like(x_batch.grad[j].data).to(self.device)
                        # assemble jacobian for each image
                        f_sor_temp.append(f_sor_grad)
                        f_tar_temp.append(f_sor_grad)
                    f_sor_grad = np.concatenate(f_sor_temp,axis=0)
                    f_tar_grad = np.concatenate(f_tar_temp,axis=0)
                    # list of jacobians for all images    
                    Xsor_grad.append(f_sor_grad) 
                    Xtar_grad.append(f_tar_grad) 
                
                # extract activation of source and target units       
                fsor = f_sor[j,self.source_unit,::self.f_subsamp,::self.f_subsamp].cpu()
                ftar = f_tar[j,self.target_unit,::self.f_subsamp,::self.f_subsamp].cpu()
                # list of activation vectors* for all images    
                Xsor.append(fsor.detach().numpy()) 
                Xtar.append(ftar.detach().numpy()) 
            
        # save the activation patterns
        self.Xsor = Xsor
        self.Xtar = Xtar
        # save the gradients
        if record_gradient:
            self.Xsor_grad = Xsor_grad
            self.Xtar_grad = Xtar_grad
            
                            
    def grad_score(self):
        self.grad_corr = np.zeros((self.ndata,len(self.clamp_unit_id)))
        for u, unit_id in enumerate(tqdm(self.clamp_unit_id)):
            # get indices for the same random units in target and surrogate net output
            tar_index = self.indexer.get_tar(unit_id)
            sur_index = self.indexer.get_sur(unit_id)
            for [i, x0_batch] in enumerate(self.batchloader):
                # make a copy of the batch
                x1_batch = x0_batch.detach().clone()
                x0_batch = x0_batch.to(device)
                x0_batch.requires_grad = True
                x1_batch = x1_batch.to(device)
                x1_batch.requires_grad = True
                f0 = self.surrogate_net.forward(x0_batch)
                f1 = self.target_net.forward(x1_batch)
                l0 = f0[:,sur_index[0],sur_index[1],sur_index[2]].sum()
                l0.backward()
                l1 = f1[:,tar_index[0],tar_index[1],tar_index[2]].sum()
                l1.backward()
                g0 = x0_batch.grad.data.cpu().numpy()
                g1 = x1_batch.grad.data.cpu().numpy()
                g0_norm = np.sqrt(np.einsum('ichw,ichw->i',g0,g0))
                g1_norm = np.sqrt(np.einsum('ichw,ichw->i',g1,g1))
                self.grad_corr[i*10:(i+1)*10,u] = np.einsum('ichw,ichw->i',g0,g1)/(g0_norm*g1_norm)
                
        self.grad_train_corr = np.nanmedian(self.grad_corr[:self.n_train],axis=0)
        self.grad_test_corr = np.nanmedian(self.grad_corr[self.n_train:self.n_train+self.n_test],axis=0)
                   
    def gscore(self, scramble = False):
        print('Calculating test score:')
        self.test_score = [None]*self.n_tar
        for i in tqdm(range(self.n_tar)):
            if scramble:
                y_predtest = np.matmul(self.X_test,self.ws[:,i].T) + self.b[i]
            else:
                y_predtest = np.matmul(self.X_test,self.w[:,i].T) + self.b[i]
            self.test_score[i] = EV(self.y_test[:,i],y_predtest)
        
        if self.grad:
            print('Calculating test gradient score:')
            self.test_gscore = [None]*self.n_tar
            if scramble:
                yg_predtest = np.matmul(self.Xg_test,self.ws.T) 
            else:
                yg_predtest = np.matmul(self.Xg_test,self.w.T) 
            for i in tqdm(range(self.n_tar)):
                self.test_gscore[i] = EV(self.yg_test[:,i],yg_predtest[:,i])
                
                
    def grad_fit(self, n_train, n_test, option = 'LR'):
        # train-test splits for regression
        # with the following options:
        # LR - linear regression for each target unit independently
        # CCA - communicate through a smaller shared subspace
        # score functions EV or R2
        
        self.X_train = np.concatenate(self.Xsor[:n_train],axis=1).T;
        self.X_test = np.concatenate(self.Xsor[n_train:n_train+n_test],axis=1).T;
        self.y_train = np.concatenate(self.Xtar[:n_train],axis=1).T;
        self.y_test = np.concatenate(self.Xtar[n_train:n_train+n_test],axis=1).T;
        
        self.w = np.empty((self.n_sor,self.n_tar))
        self.b = np.empty(self.n_tar)
        self.train_score = [None]*self.n_tar
        
        if self.grad:
            self.Xg_train = np.concatenate(self.Xsor_grad[:n_train],axis=1).T;
            self.Xg_test = np.concatenate(self.Xsor_grad[n_train:n_train+n_test],axis=1).T;
            self.yg_train = np.concatenate(self.Xtar_grad[:n_train],axis=1).T;
            self.yg_test = np.concatenate(self.Xtar_grad[n_train:n_train+n_test],axis=1).T;

            self.train_gscore = [None]*self.n_tar
        
        print('Linear regression from source to target:')
        for i in tqdm(range(self.n_tar)):
            if option == 'LR':
                reg = LinearRegression().fit(self.X_train, self.y_train[:,i])
            else:
                print(option+' not implemented!')
            self.w[:,i] = reg.coef_
            self.b[i] = reg.intercept_
            y_pred = reg.predict(self.X_train)
            self.train_score[i] = EV(self.y_train[:,i],y_pred)
        if self.grad:
            yg_predtrain = np.matmul(self.Xg_train,self.w.T)
            for i in tqdm(range(self.n_tar)):
                self.train_gscore[i] = EV(self.yg_train[:,i],yg_predtrain[:,i])
    
    """    