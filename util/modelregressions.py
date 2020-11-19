import sklearn 
import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score as EV
from synthesis.synthesizer import NeuralController
import matplotlib.pyplot as plt

class CNNCrossFit():
    def __init__(self,target_net,source_net,cnn_layer,target_units,source_units,device):
        self.target_unit = target_units
        self.source_unit = source_units
        self.n_tar = len(target_units)
        self.n_sor = len(source_units)
        self.cnn_layer = cnn_layer
        self.grad = None
        self.f_subsamp = 3
        self.img_subsamp = 10
        self.device = device
        # this is only for the size of image used for control, model could be fit with any input size
        self.img_dim = [256,256,3]
        # both nets are truncated to the selected layer and set to evaluation mode
        self.target_net = target_net[:cnn_layer]
        self.source_net = source_net[:cnn_layer]
        self.target_net.eval()
        self.source_net.eval()
    
    def design(self,dataset, batch_size = 1, record_gradient = False):
        # design extract activation (and gradient) for the entire dataset
        self.ndata = len(dataset)
        self.batch_size = batch_size
        self.batchloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                        shuffle=True, num_workers=0)
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
                        fs_act = f_sor[j,unit_id,::self.f_subsamp,::self.f_subsamp].mean()
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
                fsor = f_sor[j,self.source_unit,::self.f_subsamp,::self.f_subsamp].cpu().reshape((self.n_sor,-1))
                ftar = f_tar[j,self.target_unit,::self.f_subsamp,::self.f_subsamp].cpu().reshape((self.n_tar,-1))
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
        
    def fit(self, n_train, n_test, option = 'LR'):
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
                
    def score(self, scramble = False):
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
    
    def make_centaur(self,scramble = False):
        # conjure the mysterious half-man half-beast by glueing the source net to a conv2d layer
        self.centaur_net = nn.Sequential(
                            self.source_net,
                            nn.Conv2d(self.n_sor,self.n_tar,(1,1),stride = 1).to(self.device)
                        )
        # populate with learned weights
        with torch.no_grad():
            if scramble:
                self.ws = self.w
                for i in range(self.n_tar):
                    self.ws[:,i] = np.random.choice(self.ws[:,i],replace = False)
                self.centaur_net[-1].weight.copy_(torch.tensor(self.ws.T).unsqueeze(2).unsqueeze(3))
                self.centaur_net[-1].bias.copy_(torch.tensor(self.b))
            else:
                self.centaur_net[-1].weight.copy_(torch.tensor(self.w.T).unsqueeze(2).unsqueeze(3))
                self.centaur_net[-1].bias.copy_(torch.tensor(self.b))
        
    def control(self, subset=None, n_iter = 20):
        self.n_iter = n_iter
        # implement control with either source or target net
        self.target_act = np.zeros(self.n_tar)
        self.source_act = np.zeros(self.n_tar)
        self.ctr_score = np.zeros(self.n_tar)
        
        if subset==None:
            self.target_subset = self.target_unit
            self.n_ctr = self.n_tar
        else:
            self.target_subset = self.target_unit[:subset]
            self.n_ctr = subset
        print('Optimizing source to target control:')
        for i in tqdm(range(self.n_ctr)):
            # target control
            target_controller = NeuralController(self.target_net,self.target_unit[i],self.img_dim,self.device)
            target_controller.visualize(niter=self.n_iter,label = 'snet_tar_l'+str(self.cnn_layer)+'_u'+str(self.target_unit[i]))
            self.target_act[i] = target_controller.act
            # source control
            centaur_controller = NeuralController(self.centaur_net,self.target_unit[i],self.img_dim,self.device)
            centaur_controller.visualize(niter=self.n_iter,label = 'snet_sor_l'+str(self.cnn_layer)+'_u'+str(self.target_unit[i]))
            # calculate fraction control
            x = self.target_net.forward(centaur_controller.processed_image)
            self.source_act[i] = torch.mean(x[0,self.target_unit[i]])
            self.ctr_score[i] = self.source_act[i]/self.target_act[i]
        
                