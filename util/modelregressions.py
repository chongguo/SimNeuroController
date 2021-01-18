import sklearn 
import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import explained_variance_score as EV
from scipy.stats import pearsonr as PEAR
from synthesis.synthesizer import NeuralController
from util.misc_functions import indexfun, recreate_image, save_image
import matplotlib.pyplot as plt

class CNNCrossFit():
    def __init__(self,tar_net,sor_net,tar_layer,sor_layer,tar_units,sor_units,device):
        self.target_units = tar_units
        self.source_units = sor_units
        self.n_tar = len(tar_units)
        self.n_sor = len(sor_units)
        self.tar_layer = tar_layer
        self.sor_layer = sor_layer
        self.grad = None
        self.device = device
        # both nets are truncated to the selected layer and set to evaluation mode
        self.target_net = nn.Sequential(*list(tar_net.children())[:tar_layer]).to(device)
        self.source_net = nn.Sequential(*list(sor_net.children())[:sor_layer]).to(device)
        self.target_net.eval()
        self.source_net.eval()
    
    def design(self, dataset, batch_size = 1, shuffle = False):
        # image used for control inherit size of images used for fitting
        self.img_dim = dataset.dim
        # create dataloader for the whole dataset
        self.ndata = len(dataset)
        self.batch_size = batch_size
        self.batchloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                        shuffle=shuffle, num_workers=0)
        # get the output dimensions
        f_sor = self.source_net.forward(torch.unsqueeze(dataset[0],0).to(self.device))
        f_sor = f_sor.cpu().detach()[0,self.source_units]
        self.sor_dim = f_sor.shape
        f_tar = self.target_net.forward(torch.unsqueeze(dataset[0],0).to(self.device))
        f_tar = f_tar.cpu().detach()[0,self.target_units]
        self.tar_dim = f_tar.shape
        # generate a target/surrogate unit indexer
        self.indexer = indexfun(self.tar_dim,self.target_units)
        # create empty list for holding data
        self.Xsor = torch.zeros((self.ndata,*self.sor_dim))
        self.Xtar = torch.zeros((self.ndata,*self.tar_dim))
        # assemble data matix of activation vector for the given dataset, batch for speed
        print('Extracting activations')
        for [i, x_batch] in enumerate(tqdm(self.batchloader)):
            # run the two models
            x_batch = x_batch.to(self.device)
            f_sor = self.source_net.forward(x_batch)
            f_tar = self.target_net.forward(x_batch)
            f_sor = f_sor.cpu().detach()
            f_tar = f_tar.cpu().detach()
              
            # extract activation of source and target units       
            self.Xsor[i*self.batch_size:(i+1)*self.batch_size] = f_sor[:,self.source_units]
            self.Xtar[i*self.batch_size:(i+1)*self.batch_size] = f_tar[:,self.target_units]
        
        del x_batch,f_sor,f_tar
        torch.cuda.empty_cache()  
        
    def train_test_split(self, n_train, n_test):
        self.n_train = n_train
        self.n_test = n_test
        # train-test splits for regression
        self.sor_train = self.Xsor[:n_train].numpy();
        self.tar_train = self.Xtar[:n_train].numpy();
        self.sor_test = self.Xsor[n_train:n_train+n_test].numpy();
        self.tar_test = self.Xtar[n_train:n_train+n_test].numpy();
        
    def fit(self, option = 'LR', load = None):
        # with the following options:
        # LR - linear regression for each target unit independently
        # CCA - communicate through a smaller shared subspace
        # score functions EV or R2
        
        self.w = np.empty((self.n_sor,self.n_tar))
        self.b = np.empty(self.n_tar)
        
        if load is None:
            X = np.transpose(self.sor_train,(1,2,3,0)).reshape((self.n_sor,-1)).T;
            y = np.transpose(self.tar_train,(1,2,3,0)).reshape((self.n_tar,-1)).T;
            print('Linear regression from source to target:')
            for i in tqdm(range(self.n_tar)):
                if option == 'LR':
                    reg = LinearRegression().fit(X, y[:,i])
                else:
                    print(option+' not implemented!')
                self.w[:,i] = reg.coef_
                self.b[i] = reg.intercept_
        else:
            self.w = np.loadtxt('params\\' + load + 'weights.csv',delimiter = ',')
            self.b = np.loadtxt('params\\' + load + 'bias.csv',delimiter = ',')
         
    def get_surrogate(self,scramble = False):
        # join the source net to a conv2d layer
        self.surrogate_net = nn.Sequential(
                            *list(self.source_net.children()),
                            nn.Conv2d(self.n_sor,self.n_tar,(1,1),stride = 1).to(self.device)
                        )
        # populate with learned weights
        with torch.no_grad():
            if scramble:
                self.ws = self.w
                for i in range(self.n_tar):
                    self.ws[:,i] = np.random.choice(self.ws[:,i],replace = False)
                self.surrogate_net[-1].weight.copy_(torch.nn.Parameter(torch.from_numpy(self.ws.T[:,:,np.newaxis,np.newaxis])))
                self.surrogate_net[-1].bias.copy_(torch.nn.Parameter(torch.from_numpy(self.b)))
            else:
                self.surrogate_net[-1].weight.copy_(torch.nn.Parameter(torch.from_numpy(self.w.T[:,:,np.newaxis,np.newaxis])))
                self.surrogate_net[-1].bias.copy_(torch.nn.Parameter(torch.from_numpy(self.b)))
        # keep this on the gpu and set ot evaluation
        self.surrogate_net.to(self.device)
        self.surrogate_net.eval()
    
    def score(self, scramble = False, metric = 'EV'):
        self.train_score = np.empty(self.tar_dim)
        self.test_score = np.empty(self.tar_dim)
        print('Calculating test score:')
        if scramble:
            self.Xsur = np.einsum('nijk,il->nljk',self.Xsor,self.ws) + self.b.reshape(1,self.n_tar,1,1)
        else:
            self.Xsur = np.einsum('nijk,il->nljk',self.Xsor,self.w) + self.b.reshape(1,self.n_tar,1,1)
        self.tar_pred_train = self.Xsur[:self.n_train]
        self.tar_pred_test = self.Xsur[self.n_train:self.n_train+self.n_test]
        self.Xsur = torch.from_numpy(self.Xsur).float()
        for i in tqdm(range(len(self.indexer))):
            index = self.indexer.get_sur(i)
            if metric == 'EV':
                self.train_score[index] = EV(self.tar_train[:,index[0],index[1],index[2]],self.tar_pred_train[:,index[0],index[1],index[2]])
                self.test_score[index] = EV(self.tar_test[:,index[0],index[1],index[2]],self.tar_pred_test[:,index[0],index[1],index[2]])
            elif metric == 'Pearson':
                self.train_score[index] = PEAR(self.tar_train[:,index[0],index[1],index[2]],self.tar_pred_train[:,index[0],index[1],index[2]])[0]
                self.test_score[index] = PEAR(self.tar_test[:,index[0],index[1],index[2]],self.tar_pred_test[:,index[0],index[1],index[2]])[0]
            
    def scalar_control(self, epsilon = 1, n_units = 100, lr = .005, wd = 1e-3, n_iter = 20, load = None):
        # control with surrogate net and measure on target net
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.lr = lr
        self.wd = wd
        self.n_clamp_units = n_units
        # get a random subset of units from the center 1/3 of the channel output
        rand_set = random.sample(range(0,len(self.indexer)),len(self.indexer))
        tempidx = np.asarray(self.indexer.get_tar(rand_set),dtype=np.float)
        min1, max1 = self.tar_dim[1]*.25, self.tar_dim[1]*.75
        min2, max2 = self.tar_dim[2]*.25, self.tar_dim[2]*.75
        valididx = np.where((tempidx[1]>min1)&(tempidx[1]<max1)&(tempidx[2]>min2)&(tempidx[2]<max2))[0]
        self.clamp_unit_id = [rand_set[u] for i,u in enumerate(valididx[:n_units])]
        # target epsilon on surrogate units
        self.sur_eps = 2*epsilon*(torch.rand((self.n_train+self.n_test,n_units),requires_grad=False)>0.5).float().to(self.device)-epsilon
        self.sur_eps.requires_grad = False
        # observed epsilon on surrogate units
        self.sur_obs = torch.zeros((self.n_train+self.n_test,n_units)) 
        self.sur_obs.requires_grad = False
        # observed epsilon on target units
        self.tar_obs = torch.zeros((self.n_train+self.n_test,n_units)) 
        self.tar_obs.requires_grad = False
        # loss for the clamp target 
        clamp_loss = torch.nn.MSELoss()
        
        print('Optimizing surrogate unit scalar control:')
        for [i, x1_batch] in enumerate(tqdm(self.batchloader)):
            # make a copy of the batch
            x1_batch_copy = x1_batch.detach().clone()
            for u, unit_id in enumerate(self.clamp_unit_id):
                # get indices for the same random units in target and surrogate net output
                tar_index = self.indexer.get_tar(unit_id)
                sur_index = self.indexer.get_sur(unit_id)
                # get the clean target and surrogate activations
                f0_tar = self.Xtar[i*self.batch_size:(i+1)*self.batch_size,tar_index[0],tar_index[1],tar_index[2]].to(self.device)
                f0_tar.requires_grad = False
                f0_sur = self.Xsur[i*self.batch_size:(i+1)*self.batch_size,sur_index[0],sur_index[1],sur_index[2]].to(self.device)
                f0_sur.requires_grad = False
                # pass current batch to gpu
                x1_batch = x1_batch.to(self.device)
                x1_batch.requires_grad = True
                # optimizer for the batch
                optimizer = torch.optim.Adam([x1_batch],lr=self.lr,weight_decay=self.wd)
                # pick target of current unit
                f1_sur_opt = f0_sur+self.sur_eps[i*self.batch_size:(i+1)*self.batch_size,u]
                # iteratively find images that steer the surrogate unit epsilon distance up or down
                for n in range(n_iter):
                    optimizer.zero_grad()
                    f1_sur = self.surrogate_net.forward(x1_batch)
                    # compute clamp loss
                    losses = clamp_loss(f1_sur[:,sur_index[0],sur_index[1],sur_index[2]],f1_sur_opt)
                    losses.backward(retain_graph=True)
                    optimizer.step()
                    torch.clamp(x1_batch,min=-2.25,max=2.75)
                # pass the batch of optimized clamp image to target network
                f1_tar = self.target_net.forward(x1_batch)
                # record the epsilons on the target networks
                self.tar_obs[i*self.batch_size:(i+1)*self.batch_size,u] = f1_tar[:,tar_index[0],tar_index[1],tar_index[2]].detach().data - f0_tar
                self.sur_obs[i*self.batch_size:(i+1)*self.batch_size,u] = f1_sur[:,sur_index[0],sur_index[1],sur_index[2]].detach().data - f0_sur          
                # reset the image
                x1_batch = x1_batch_copy.detach().clone()
    
    
    def control_score(self, metric = 'EV'):
        self.train_sub_score = np.empty(self.n_clamp_units)
        self.test_sub_score = np.empty(self.n_clamp_units)
        self.train_sur_score = np.empty(self.n_clamp_units)
        self.test_sur_score = np.empty(self.n_clamp_units)
        self.train_ctr_score = np.empty(self.n_clamp_units)
        self.test_ctr_score = np.empty(self.n_clamp_units)
        
        self.sur_eps_train = self.sur_eps[:self.n_train].cpu().numpy()
        self.sur_eps_test = self.sur_eps[self.n_train:self.n_train+self.n_test].cpu().numpy()
        self.sur_obs_train = self.sur_obs[:self.n_train].cpu().numpy()
        self.sur_obs_test = self.sur_obs[self.n_train:self.n_train+self.n_test].cpu().numpy()
        self.tar_obs_train = self.tar_obs[:self.n_train].cpu().numpy()
        self.tar_obs_test = self.tar_obs[self.n_train:self.n_train+self.n_test].cpu().numpy()
        
        for u, unit_id in enumerate(self.clamp_unit_id):
            index = self.indexer.get_sur(unit_id)
            if metric == 'EV':
                self.train_sur_score[u] = EV(self.sur_eps_train[:,u],self.sur_obs_train[:,u])
                self.test_sur_score[u] = EV(self.sur_eps_test[:,u],self.sur_obs_test[:,u])
                self.train_ctr_score[u] = EV(self.sur_obs_train[:,u],self.tar_obs_train[:,u])
                self.test_ctr_score[u] = EV(self.sur_obs_test[:,u],self.tar_obs_test[:,u])
                self.train_sub_score[u] = self.train_score[index]
                self.test_sub_score[u] = self.test_score[index]
            elif metric == 'Pearson':
                self.train_sur_score[u] = PEAR(self.sur_eps_train[:,u],self.sur_obs_train[:,u])
                self.test_sur_score[u] = PEAR(self.sur_eps_test[:,u],self.sur_obs_test[:,u])
                self.train_ctr_score[u] = PEAR(self.sur_obs_train[:,u],self.tar_obs_train[:,u])
                self.test_ctr_score[u] = PEAR(self.sur_obs_test[:,u],self.tar_obs_test[:,u])
                self.train_sub_score[u] = self.train_score[index]
                self.test_sub_score[u] = self.test_score[index]
                
    def vec_control_score(self, metric = 'EV'):
        
        self.train_sub_score = np.empty(self.n_sets)
        self.test_sub_score = np.empty(self.n_sets)
        self.train_sur_score = np.empty(self.n_sets)
        self.test_sur_score = np.empty(self.n_sets)
        self.train_ctr_score = np.empty(self.n_sets)
        self.test_ctr_score = np.empty(self.n_sets)
        
        self.sur_eps_train = self.sur_eps[:self.n_train].cpu().numpy()
        self.sur_eps_test = self.sur_eps[self.n_train:self.n_train+self.n_test].cpu().numpy()
        self.sur_obs_train = self.sur_obs[:self.n_train].cpu().numpy()
        self.sur_obs_test = self.sur_obs[self.n_train:self.n_train+self.n_test].cpu().numpy()
        self.tar_obs_train = self.tar_obs[:self.n_train].cpu().numpy()
        self.tar_obs_test = self.tar_obs[self.n_train:self.n_train+self.n_test].cpu().numpy()
        
        for s in range(self.n_sets):
            unit_id = self.set_id[:,s]
            index = self.indexer.get_sur(unit_id)
            if metric == 'EV':
                self.train_sur_score[s] = EV(self.sur_eps_train[:,s],self.sur_obs_train[:,s])
                self.test_sur_score[s] = EV(self.sur_eps_test[:,s],self.sur_obs_test[:,s])
                self.train_ctr_score[s] = EV(self.sur_obs_train[:,s],self.tar_obs_train[:,s])
                self.test_ctr_score[s] = EV(self.sur_obs_test[:,s],self.tar_obs_test[:,s])
                self.train_sub_score[s] = self.train_score[index].mean()
                self.test_sub_score[s] = self.test_score[index].mean()
                
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

    def vector_control(self, epsilon = 10, n_sets = 100, set_size = 10, lr = .001, wd = 1e-5, n_iter = 20, load = None):
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
     
    """    
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